"""Autonomous analysis agent (v0.5).

Replaces the old linear "one question -> one query -> one answer" path with a
plan / execute / reflect / synthesize loop:

    plan_analysis -> [run_query]* -> reflect -> ( [run_query]* -> reflect )* -> synthesize_report

1. ``plan_analysis``    turns the question into hypotheses + ordered sub-questions.
2. ``run_query``        answers one sub-question: retrieve -> generate -> validate
                        -> (dry_run if BigQuery) -> execute -> record a finding.
                        Reuses the existing read-only SQL guardrails.
3. ``reflect``          decides whether the evidence is sufficient or a few bounded
                        follow-up sub-questions are warranted.
4. ``synthesize_report``drafts findings + recommendations, citing which query
                        produced which number, with explicit uncertainty.

The number of executed sub-queries (planned + follow-ups) is capped by
``config.max_sub_queries`` so runtime and cost stay bounded.
"""

from __future__ import annotations

import json
from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph

from analytics_agent.config import AgentConfig, load_config
from analytics_agent.llm import build_chat_model
from analytics_agent.retriever import create_retriever
from analytics_agent.runners import build_runner
from analytics_agent.sql import (
    SQLValidationError,
    ensure_default_limit,
    strip_sql_fences,
    validate_sql,
)


class AgentState(TypedDict, total=False):
    question: str
    # Plan
    hypotheses: list[str]
    sub_questions: list[str]
    plan: str
    # Loop bookkeeping
    cursor: int
    iterations: int
    findings: list[dict[str, Any]]
    reflections: list[str]
    # Output
    report: str
    answer: str
    # Backward-compat surface for the single-box UI (reflects the last query)
    sql: str
    rows: list[dict[str, Any]]
    bytes_processed: int
    context_documents: list[Document]
    context_text: str


_DIALECT = {
    "bigquery": "BigQuery Standard SQL",
    "duckdb": "DuckDB SQL",
}

# Total SQL attempts per sub-question: the first draft plus bounded self-repair
# retries that feed the validation/database error back to the LLM.
_MAX_SQL_ATTEMPTS = 2


def _format_context(documents: list[Document]) -> str:
    chunks = []
    for doc in documents:
        source = doc.metadata.get("model_name") or doc.metadata.get("name") or "unknown"
        doc_type = doc.metadata.get("doc_type", "document")
        chunks.append(f"Source: {source} ({doc_type})\n{doc.page_content}")
    return "\n\n---\n\n".join(chunks)


def _format_rows(rows: list[dict[str, Any]], max_rows: int = 20) -> str:
    if not rows:
        return "No rows returned."
    preview = rows[:max_rows]
    return "\n".join(str(row) for row in preview)


def _format_bullets(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- (none)"


def _format_plan(hypotheses: list[str], sub_questions: list[str]) -> str:
    return (
        f"Hypotheses:\n{_format_bullets(hypotheses)}\n\n"
        f"Sub-questions:\n{_format_bullets(sub_questions)}"
    )


def _format_findings_digest(findings: list[dict[str, Any]]) -> str:
    if not findings:
        return "No queries have been run yet."
    blocks = []
    for finding in findings:
        header = f"[{finding['id']}] {finding['sub_question']}"
        if finding.get("error"):
            blocks.append(f"{header}\nERROR: {finding['error']}")
        else:
            blocks.append(
                f"{header}\nSQL: {finding['sql']}\n"
                f"Rows ({finding['row_count']}):\n{_format_rows(finding['rows'])}"
            )
    return "\n\n".join(blocks)


def _extract_json(text: str) -> dict | None:
    """Tolerantly pull the first JSON object out of an LLM response."""
    if not text:
        return None
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except (ValueError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


# --- Prompts -----------------------------------------------------------------


def _plan_prompt(question: str) -> list:
    return [
        SystemMessage(
            content=(
                "You are a senior data scientist planning an end-to-end analysis. "
                "Given a business question, produce (1) a few concrete hypotheses "
                "worth testing and (2) an ordered list of specific, self-contained "
                "analytical sub-questions, each answerable by a single read-only SQL "
                "query over a dbt warehouse. Keep sub-questions focused and "
                "non-redundant; order them so earlier answers inform later ones. "
                'Respond with ONLY a JSON object of the form '
                '{"hypotheses": ["..."], "sub_questions": ["..."]}.'
            )
        ),
        HumanMessage(content=f"Business question:\n{question}"),
    ]


def _generate_sql_prompt(config: AgentConfig, question: str, context: str) -> list:
    table_list = "\n".join(
        f"- {name}: {fqtn}" for name, fqtn in config.fully_qualified_tables.items()
    )
    dialect = _DIALECT.get(config.backend, "standard SQL")
    return [
        SystemMessage(
            content=(
                f"You are an analytics engineer. Generate exactly one read-only "
                f"{dialect} query. Return only SQL, no markdown. "
                "Use only the allowed table identifiers exactly as written below "
                "(do not rename, requalify, or strip backticks). "
                "Do not query raw sources. Prefer aggregate queries for metric questions. "
                "Use clear aliases. If showing detail rows, include a LIMIT."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Allowed tables:\n{table_list}\n\n"
                f"Retrieved dbt and metric context:\n{context}"
            )
        ),
    ]


def _repair_sql_prompt(
    config: AgentConfig,
    question: str,
    context: str,
    failed_sql: str,
    error: str,
) -> list:
    table_list = "\n".join(
        f"- {name}: {fqtn}" for name, fqtn in config.fully_qualified_tables.items()
    )
    dialect = _DIALECT.get(config.backend, "standard SQL")
    return [
        SystemMessage(
            content=(
                f"You are an analytics engineer fixing a broken {dialect} query. "
                "The previous attempt failed with the error shown. Return a single "
                "corrected read-only query and nothing else (no markdown). Use only "
                "the allowed table identifiers exactly as written, and only columns "
                "that exist — if the error lists candidate column names, use those. "
                "Do not invent columns."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{question}\n\n"
                f"Allowed tables:\n{table_list}\n\n"
                f"Retrieved dbt and metric context:\n{context}\n\n"
                f"Failed SQL:\n{failed_sql}\n\n"
                f"Error:\n{error}"
            )
        ),
    ]


def _reflect_prompt(question: str, plan: str, findings: list[dict[str, Any]]) -> list:
    return [
        SystemMessage(
            content=(
                "You are reviewing partial analysis results to decide whether the "
                "evidence already answers the original question or whether a few "
                "targeted follow-up queries are needed. Only request follow-ups that "
                "are clearly justified by gaps, errors, or surprises in the results; "
                "avoid redundant or speculative queries. Each follow-up must be a "
                "self-contained sub-question answerable by one SQL query. "
                'Respond with ONLY a JSON object of the form '
                '{"sufficient": true/false, "reasoning": "...", '
                '"follow_up_sub_questions": ["..."]}.'
            )
        ),
        HumanMessage(
            content=(
                f"Original question:\n{question}\n\n"
                f"Plan:\n{plan}\n\n"
                f"Findings so far:\n{_format_findings_digest(findings)}"
            )
        ),
    ]


def _synthesize_prompt(
    question: str, hypotheses: list[str], findings: list[dict[str, Any]]
) -> list:
    return [
        SystemMessage(
            content=(
                "You are a senior analyst writing the final report. Using ONLY the "
                "query results provided, synthesize clear findings and actionable "
                "recommendations that answer the original question. Cite the query id "
                "(e.g. [Q1]) next to each specific number or claim it supports. "
                "State uncertainty explicitly and note where data was missing or a "
                "query failed. Do not invent numbers that are not in the results."
            )
        ),
        HumanMessage(
            content=(
                f"Original question:\n{question}\n\n"
                f"Hypotheses:\n{_format_bullets(hypotheses)}\n\n"
                f"Evidence (each query and its result):\n"
                f"{_format_findings_digest(findings)}"
            )
        ),
    ]


def build_graph(config: AgentConfig):
    llm = build_chat_model(config)
    retriever = create_retriever(config)
    runner = build_runner(config)

    def _answer_sub_question(
        sub_question: str, index: int
    ) -> tuple[dict[str, Any], list[Document], str]:
        """Run the full guarded query pipeline for one sub-question.

        Errors are captured into the finding rather than raised, so a single bad
        sub-question never aborts the whole analysis.
        """
        finding: dict[str, Any] = {
            "id": f"Q{index}",
            "sub_question": sub_question,
            "sql": "",
            "rows": [],
            "row_count": 0,
            "bytes_processed": None,
            "error": None,
        }
        try:
            documents = retriever.invoke(sub_question)
        except Exception as exc:  # retrieval is best-effort
            finding["error"] = f"retrieval failed: {exc}"
            return finding, [], ""

        context_text = _format_context(documents)
        # First draft, then bounded self-repair: each failure (validation or a
        # database error) is fed back to the LLM so it can correct the query.
        messages = _generate_sql_prompt(config, sub_question, context_text)
        last_error: str | None = None
        for attempt in range(_MAX_SQL_ATTEMPTS):
            response = llm.invoke(messages)
            sql = ensure_default_limit(
                strip_sql_fences(response.content), config.default_result_limit
            )
            try:
                result = validate_sql(
                    sql,
                    allowed_table_names=config.allowed_table_names,
                    fully_qualified_tables=set(config.fully_qualified_tables.values()),
                )
            except SQLValidationError as exc:
                last_error = f"sql validation failed: {exc}"
                finding["sql"] = sql
                messages = _repair_sql_prompt(
                    config, sub_question, context_text, sql, str(exc)
                )
                continue

            finding["sql"] = result.sql
            try:
                if config.backend == "bigquery":
                    finding["bytes_processed"] = runner.dry_run(result.sql)
                rows = runner.execute(
                    result.sql, max_results=config.default_result_limit
                )
            except Exception as exc:
                last_error = f"query failed: {exc}"
                messages = _repair_sql_prompt(
                    config, sub_question, context_text, result.sql, str(exc)
                )
                continue

            finding["rows"] = rows
            finding["row_count"] = len(rows)
            finding["error"] = None
            return finding, documents, context_text

        finding["error"] = last_error
        return finding, documents, context_text

    def plan_analysis(state: AgentState) -> AgentState:
        response = llm.invoke(_plan_prompt(state["question"]))
        parsed = _extract_json(response.content) or {}
        hypotheses = _str_list(parsed.get("hypotheses"))
        sub_questions = _str_list(parsed.get("sub_questions"))
        if not sub_questions:
            # Fall back to answering the question directly as a single sub-question.
            sub_questions = [state["question"]]
        sub_questions = sub_questions[: config.max_sub_queries]
        return {
            **state,
            "hypotheses": hypotheses,
            "sub_questions": sub_questions,
            "plan": _format_plan(hypotheses, sub_questions),
            "cursor": 0,
            "iterations": 0,
            "findings": [],
            "reflections": [],
        }

    def run_query(state: AgentState) -> AgentState:
        cursor = state["cursor"]
        sub_question = state["sub_questions"][cursor]
        finding, documents, context_text = _answer_sub_question(
            sub_question, len(state["findings"]) + 1
        )
        update: AgentState = {
            **state,
            "cursor": cursor + 1,
            "iterations": state["iterations"] + 1,
            "findings": state["findings"] + [finding],
            "context_documents": documents,
            "context_text": context_text,
        }
        # Surface the most recent successful query for the legacy single-box UI.
        if not finding["error"]:
            update["sql"] = finding["sql"]
            update["rows"] = finding["rows"]
            if finding["bytes_processed"] is not None:
                update["bytes_processed"] = finding["bytes_processed"]
        return update

    def reflect(state: AgentState) -> AgentState:
        remaining = config.max_sub_queries - state["iterations"]
        if remaining <= 0:
            return {
                **state,
                "reflections": state["reflections"]
                + ["Reached the sub-query budget; proceeding to synthesis."],
            }
        response = llm.invoke(
            _reflect_prompt(state["question"], state["plan"], state["findings"])
        )
        parsed = _extract_json(response.content) or {}
        reasoning = str(parsed.get("reasoning", "")).strip()
        sufficient = bool(parsed.get("sufficient", True))
        follow_ups = _str_list(parsed.get("follow_up_sub_questions"))[:remaining]

        update: AgentState = {
            **state,
            "reflections": state["reflections"] + ([reasoning] if reasoning else []),
        }
        if not sufficient and follow_ups:
            update["sub_questions"] = state["sub_questions"] + follow_ups
        return update

    def synthesize_report(state: AgentState) -> AgentState:
        response = llm.invoke(
            _synthesize_prompt(
                state["question"], state.get("hypotheses", []), state["findings"]
            )
        )
        report = response.content
        return {**state, "report": report, "answer": report}

    def _has_pending_work(state: AgentState) -> bool:
        return (
            state["cursor"] < len(state["sub_questions"])
            and state["iterations"] < config.max_sub_queries
        )

    def _dispatch(state: AgentState) -> str:
        # After planning or a query: run the next sub-question, else reflect.
        return "run_query" if _has_pending_work(state) else "reflect"

    def _after_reflect(state: AgentState) -> str:
        # Reflection may have enqueued bounded follow-ups; otherwise synthesize.
        return "run_query" if _has_pending_work(state) else "synthesize_report"

    graph = StateGraph(AgentState)
    graph.add_node("plan_analysis", plan_analysis)
    graph.add_node("run_query", run_query)
    graph.add_node("reflect", reflect)
    graph.add_node("synthesize_report", synthesize_report)

    graph.set_entry_point("plan_analysis")
    graph.add_conditional_edges(
        "plan_analysis", _dispatch, {"run_query": "run_query", "reflect": "reflect"}
    )
    graph.add_conditional_edges(
        "run_query", _dispatch, {"run_query": "run_query", "reflect": "reflect"}
    )
    graph.add_conditional_edges(
        "reflect",
        _after_reflect,
        {"run_query": "run_query", "synthesize_report": "synthesize_report"},
    )
    graph.add_edge("synthesize_report", END)
    return graph.compile()


def run_analytics_question(question: str, config: AgentConfig | None = None) -> AgentState:
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    resolved_config = config or load_config()
    graph = build_graph(resolved_config)
    # Generous ceiling: plan + (run_query + reflect) rounds + synthesize stay well
    # under this, while max_sub_queries is the real bound on work done.
    recursion_limit = resolved_config.max_sub_queries * 2 + 10
    return graph.invoke(
        {"question": question.strip()},
        config={"recursion_limit": recursion_limit},
    )


# Stage names emitted by ``stream_analytics_events`` so UIs can render the
# agent's reasoning live instead of only its final answer.
STAGE_PLAN = "plan"
STAGE_QUERY = "query"
STAGE_REFLECT = "reflect"
STAGE_SYNTHESIZE = "synthesize"


def stream_analytics_events(question: str, config: AgentConfig | None = None):
    """Run the agent and yield one event per completed graph node.

    Each event is a dict with a ``stage`` key (one of the ``STAGE_*`` constants)
    plus stage-specific payload. The final ``synthesize`` event also carries the
    full terminal ``state`` so callers can recover rows/sql/sources for the UI.
    This lets a frontend show planning -> querying -> reflecting -> synthesizing
    as it happens. ``run_analytics_question`` remains for non-streaming callers.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    resolved_config = config or load_config()
    graph = build_graph(resolved_config)
    recursion_limit = resolved_config.max_sub_queries * 2 + 10

    emitted_findings = 0
    emitted_reflections = 0
    for chunk in graph.stream(
        {"question": question.strip()},
        config={"recursion_limit": recursion_limit},
        stream_mode="updates",
    ):
        for node, state in chunk.items():
            if node == "plan_analysis":
                yield {
                    "stage": STAGE_PLAN,
                    "hypotheses": state.get("hypotheses", []),
                    "sub_questions": state.get("sub_questions", []),
                }
            elif node == "run_query":
                findings = state.get("findings", [])
                # Emit only findings produced since the last event.
                for finding in findings[emitted_findings:]:
                    yield {
                        "stage": STAGE_QUERY,
                        "finding": finding,
                        "index": finding["id"],
                        "planned_total": len(state.get("sub_questions", [])),
                    }
                emitted_findings = len(findings)
            elif node == "reflect":
                reflections = state.get("reflections", [])
                new_reflections = reflections[emitted_reflections:]
                emitted_reflections = len(reflections)
                yield {
                    "stage": STAGE_REFLECT,
                    "reflection": new_reflections[-1] if new_reflections else "",
                    "sub_questions": state.get("sub_questions", []),
                }
            elif node == "synthesize_report":
                yield {
                    "stage": STAGE_SYNTHESIZE,
                    "report": state.get("report", ""),
                    "state": state,
                }
