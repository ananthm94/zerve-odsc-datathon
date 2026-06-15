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

from analytics_agent import dashboard
from analytics_agent.core.config import AgentConfig, load_config
from analytics_agent.core.contracts import DashboardSpec, MetricDefinition, ReportRecord
from analytics_agent.core.llm import build_chat_model
from analytics_agent.library.memory import ReportMemory
from analytics_agent.ask.model_router import (
    TASK_CONDENSE,
    TASK_PLAN,
    TASK_REFLECT,
    TASK_REPAIR,
    TASK_SQL,
    TASK_SYNTHESIZE,
    build_model_for_task,
    estimate_call_cost,
    messages_to_text,
    select_model,
    usage_summary,
)
from analytics_agent.core.retriever import create_retriever
from analytics_agent.core.runners import build_runner
from analytics_agent.ask.semantic_layer import (
    format_for_prompt,
    load_semantic_layer,
    lookup_metrics,
)
from analytics_agent.ask.snapshots import SnapshotStore, compare_to_snapshot
from analytics_agent.core.sql import (
    SQLValidationError,
    ensure_default_limit,
    strip_sql_fences,
    validate_sql,
)


class AgentState(TypedDict, total=False):
    question: str
    # Agentic orchestrator extensions
    intent: str
    metric_definitions: list[MetricDefinition]
    recalled_reports: list[ReportRecord]
    snapshot_check: dict[str, Any]
    dashboard_worthy: bool
    dashboard_spec: DashboardSpec
    response_mode: str
    final_response: str
    model_usage: dict[str, Any]
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
                "You are a senior analyst writing the final report. Choose the right "
                "answer depth from the user's question and the evidence: quick metric "
                "questions should be compact, diagnostic questions should get a "
                "standard executive summary, and explicit deep-dive requests can be "
                "longer. Always use a pyramid structure: put the most important "
                "takeaway first, then key numbers, then explanation, then caveats "
                "only if useful. Avoid repeating the same number in multiple sections. "
                "Cite the query id (e.g. [Q1]) next to each specific number or claim "
                "it supports. State uncertainty explicitly and note where data was "
                "missing or a query failed. Do not invent numbers that are not in the "
                "results."
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
    plan_llm = _llm_for_task(config, TASK_PLAN)
    sql_llm = _llm_for_task(config, TASK_SQL)
    repair_llm = _llm_for_task(config, TASK_REPAIR)
    reflect_llm = _llm_for_task(config, TASK_REFLECT)
    synthesize_llm = _llm_for_task(config, TASK_SYNTHESIZE)
    retriever = create_retriever(config)
    runner = build_runner(config)
    usage_estimates = []

    def _invoke_llm(task: str, llm, messages: list):
        response = llm.invoke(messages)
        usage_estimates.append(
            estimate_call_cost(
                config,
                task,
                messages_to_text(messages),
                response,
                selection=select_model(config, task),
            )
        )
        return response

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
        active_sql_llm = sql_llm
        active_task = TASK_SQL
        for attempt in range(_MAX_SQL_ATTEMPTS):
            response = _invoke_llm(active_task, active_sql_llm, messages)
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
                active_sql_llm = repair_llm
                active_task = TASK_REPAIR
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
                active_sql_llm = repair_llm
                active_task = TASK_REPAIR
                continue

            finding["rows"] = rows
            finding["row_count"] = len(rows)
            finding["error"] = None
            return finding, documents, context_text

        finding["error"] = last_error
        return finding, documents, context_text

    def plan_analysis(state: AgentState) -> AgentState:
        response = _invoke_llm(TASK_PLAN, plan_llm, _plan_prompt(state["question"]))
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
        response = _invoke_llm(
            TASK_REFLECT,
            reflect_llm,
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
        response = _invoke_llm(
            TASK_SYNTHESIZE,
            synthesize_llm,
            _synthesize_prompt(
                state["question"], state.get("hypotheses", []), state["findings"]
            )
        )
        report = response.content
        return {
            **state,
            "report": report,
            "answer": report,
            "model_usage": usage_summary(usage_estimates),
        }

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


def _llm_for_task(config: AgentConfig, task: str):
    if config.model_routing_enabled:
        return build_model_for_task(config, task)
    return build_chat_model(config)


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
STAGE_CONDENSE = "condense"
STAGE_INTENT = "intent"
STAGE_SEMANTIC = "semantic_layer"
STAGE_MEMORY = "memory"
STAGE_RESPONSE_MODE = "response_mode"
STAGE_SNAPSHOT = "snapshot_check"
STAGE_FINAL = "final_response"

INTENT_METRIC_DEFINITION = "metric_definition_question"
INTENT_SIMPLE_METRIC = "simple_metric_question"
INTENT_DASHBOARD_REQUEST = "dashboard_request"
INTENT_ANALYSIS = "analysis"

RESPONSE_DEFINITION = "definition"
RESPONSE_QUICK = "quick_answer"
RESPONSE_STANDARD = "standard_analysis"
RESPONSE_DEEP = "deep_dive"
RESPONSE_DASHBOARD = "dashboard_spec"


def _condense_prompt(question: str, history: list[dict[str, str]]) -> list:
    turns = "\n".join(
        f"User: {turn.get('question', '')}\nAssistant: {turn.get('answer', '')}"
        for turn in history
    )
    return [
        SystemMessage(
            content=(
                "You rewrite a follow-up question into a standalone question using "
                "the prior conversation. Resolve pronouns and implicit references "
                "(e.g. 'what about for mobile users?') into a fully self-contained "
                "question that keeps the original metric/subject. If the follow-up is "
                "already standalone, return it unchanged. Output ONLY the rewritten "
                "question, with no preamble, quotes, or explanation."
            )
        ),
        HumanMessage(
            content=(
                f"Conversation so far:\n{turns}\n\n"
                f"Follow-up question:\n{question}\n\n"
                "Standalone question:"
            )
        ),
    ]


def _condense_question(
    question: str,
    history: list[dict[str, str]] | None,
    config: AgentConfig,
) -> str:
    """Rewrite a follow-up into a standalone question using prior turns.

    No history means nothing to resolve, so the question is returned verbatim and
    no LLM call is made. On any failure we fall back to the original question so a
    condense hiccup never blocks the answer.
    """
    if not history:
        return question
    try:
        llm = build_model_for_task(config, TASK_CONDENSE)
        response = llm.invoke(_condense_prompt(question, history))
        rewritten = str(getattr(response, "content", "") or "").strip()
    except Exception:
        return question
    return rewritten or question


def build_orchestrator_graph(config: AgentConfig):
    """Return a lightweight orchestrator handle for agentic entry points.

    The legacy LangGraph analysis graph remains the execution engine for
    one-time analysis. This wrapper owns intent routing, semantic grounding,
    memory lookup, snapshot hooks, and final response assembly.
    """
    return {"config": config}


def run_agentic_question(
    question: str,
    config: AgentConfig | None = None,
    history: list[dict[str, str]] | None = None,
) -> AgentState:
    final: AgentState = {}
    for event in stream_agentic_events(question, config=config, history=history):
        if event["stage"] == STAGE_FINAL:
            final = event.get("state", {})
    return final


def stream_agentic_events(
    question: str,
    config: AgentConfig | None = None,
    history: list[dict[str, str]] | None = None,
):
    """Stream trust-loop events while preserving the legacy analysis path.

    ``history`` is an optional list of prior ``{"question", "answer"}`` turns. When
    present, the incoming question is first condensed into a standalone question so
    follow-ups ("what about for mobile users?") resolve against the conversation.
    """
    if not question.strip():
        raise ValueError("Question cannot be empty.")
    resolved_config = config or load_config()
    build_orchestrator_graph(resolved_config)
    original_question = question.strip()

    # Resolve follow-ups into a standalone question before any routing/grounding.
    clean_question = _condense_question(original_question, history, resolved_config)
    if clean_question != original_question:
        yield {
            "stage": STAGE_CONDENSE,
            "original_question": original_question,
            "standalone_question": clean_question,
        }

    intent = _classify_intent(clean_question)
    response_mode = _choose_response_mode(clean_question, intent)
    state: AgentState = {
        "question": clean_question,
        "intent": intent,
        "response_mode": response_mode,
    }
    yield {"stage": STAGE_INTENT, "intent": intent}
    yield {
        "stage": STAGE_RESPONSE_MODE,
        "response_mode": response_mode,
        "description": _response_mode_description(response_mode),
    }

    layer = load_semantic_layer(resolved_config)
    metric_defs = lookup_metrics(clean_question, layer, llm=None)
    state["metric_definitions"] = metric_defs
    yield {
        "stage": STAGE_SEMANTIC,
        "metric_definitions": metric_defs,
        "formatted": format_for_prompt(metric_defs),
    }

    if intent == INTENT_METRIC_DEFINITION:
        memory = ReportMemory(resolved_config)
        recalled = memory.recall_reports(clean_question)
        state["recalled_reports"] = recalled
        yield {"stage": STAGE_MEMORY, "recalled_reports": recalled}

        response = _metric_definition_response(metric_defs)
        state["final_response"] = response
        state["answer"] = response
        state["report"] = response
        yield {
            "stage": STAGE_FINAL,
            "final_response": response,
            "state": state,
        }
        return

    if intent == INTENT_SIMPLE_METRIC:
        response, rows = _simple_metric_response(
            clean_question, resolved_config, response_mode
        )
        state["rows"] = rows
        state["final_response"] = response
        state["answer"] = response
        state["report"] = response
        yield {
            "stage": STAGE_FINAL,
            "final_response": response,
            "state": state,
        }
        return

    if intent == INTENT_DASHBOARD_REQUEST:
        spec = _dashboard_spec_from_metrics(clean_question, metric_defs)
        state["dashboard_spec"] = spec
        state["dashboard_worthy"] = True
        response = _dashboard_request_response(spec)
        state["final_response"] = response
        state["answer"] = response
        state["report"] = response
        yield {
            "stage": STAGE_FINAL,
            "final_response": response,
            "dashboard_spec": spec,
            "state": state,
        }
        return

    terminal_state: AgentState = {}
    for event in stream_analytics_events(clean_question, config=resolved_config):
        yield event
        if event["stage"] == STAGE_SYNTHESIZE:
            terminal_state = event.get("state", {})

    metric_name, value = _first_numeric_finding(terminal_state.get("findings", []))
    snapshot_check: dict[str, Any] = {
        "baseline": None,
        "pct_diff": None,
        "flag": False,
    }
    if metric_name is not None and value is not None:
        snapshot_check = compare_to_snapshot(
            metric_name,
            value,
            SnapshotStore(resolved_config),
            resolved_config.snapshot_drift_threshold,
        )
    terminal_state["snapshot_check"] = snapshot_check
    terminal_state["response_mode"] = response_mode
    terminal_state["final_response"] = terminal_state.get("report", "")
    yield {"stage": STAGE_SNAPSHOT, "snapshot_check": snapshot_check}
    yield {
        "stage": STAGE_FINAL,
        "final_response": terminal_state.get("final_response", ""),
        "state": terminal_state,
    }


def _classify_intent(question: str) -> str:
    lowered = question.lower()
    definition_markers = (
        "what is",
        "what's",
        "define",
        "definition",
        "meaning of",
        "how is",
    )
    metric_markers = (
        "metric",
        "daily active users",
        "dau",
        "ai adoption",
        "adoption rate",
        "tokens",
        "credits",
        "exceptions",
        "latency",
    )
    if any(marker in lowered for marker in definition_markers) and any(
        marker in lowered for marker in metric_markers
    ):
        return INTENT_METRIC_DEFINITION
    if (
        any(marker in lowered for marker in ("how many", "number of", "count of"))
        and "user" in lowered
        and any(marker in lowered for marker in ("ai", "a.i.", "artificial intelligence"))
    ):
        return INTENT_SIMPLE_METRIC
    if any(
        marker in lowered
        for marker in ("dashboard", "chart", "visualize", "monitor", "track")
    ):
        return INTENT_DASHBOARD_REQUEST
    return INTENT_ANALYSIS


def _choose_response_mode(question: str, intent: str) -> str:
    lowered = question.lower()
    if intent == INTENT_METRIC_DEFINITION:
        return RESPONSE_DEFINITION
    if intent == INTENT_SIMPLE_METRIC:
        return RESPONSE_QUICK
    if intent == INTENT_DASHBOARD_REQUEST:
        return RESPONSE_DASHBOARD
    if any(
        marker in lowered
        for marker in (
            "deep dive",
            "in detail",
            "detailed",
            "comprehensive",
            "explain everything",
        )
    ):
        return RESPONSE_DEEP
    return RESPONSE_STANDARD


def _response_mode_description(response_mode: str) -> str:
    descriptions = {
        RESPONSE_DEFINITION: "Definition: answer with the official metric contract.",
        RESPONSE_QUICK: "Quick answer: lead with the number, then brief context.",
        RESPONSE_STANDARD: "Standard analysis: takeaway, key numbers, explanation, caveats.",
        RESPONSE_DEEP: "Deep dive: fuller analysis because the question asks for depth.",
        RESPONSE_DASHBOARD: "Dashboard spec: produce a reusable monitoring view.",
    }
    return descriptions.get(response_mode, "Standard analysis.")


def _simple_metric_response(
    question: str, config: AgentConfig, response_mode: str = RESPONSE_QUICK
) -> tuple[str, list[dict[str, Any]]]:
    lowered = question.lower()
    if "user" in lowered and "ai" in lowered:
        kpis = dashboard.kpis(config)
        ai_users = float(kpis.get("ai_users", 0) or 0)
        total_users = float(kpis.get("total_users", 0) or 0)
        adoption = (ai_users / total_users) if total_users else 0.0
        response = _format_ai_users_answer(
            ai_users, total_users, adoption, response_mode
        )
        return response, [
            {
                "metric": "ai_users",
                "ai_users": ai_users,
                "total_users": total_users,
                "ai_adoption_rate": adoption,
            }
        ]
    return (
        "I can answer that as a metric, but I do not have a direct fast-path for "
        "this phrasing yet. Try asking for AI users, total users, or daily active users.",
        [],
    )


def _format_ai_users_answer(
    ai_users: float, total_users: float, adoption: float, response_mode: str
) -> str:
    top_line = (
        f"**{ai_users:,.0f} users have used AI**, "
        f"which is {adoption * 100:.1f}% of {total_users:,.0f} total users."
    )
    if response_mode == RESPONSE_QUICK:
        return (
            f"{top_line}\n\n"
            "**Context**\n"
            "- This is a user-level adoption count, not the number of AI generations.\n"
            "- A user counts once if they have at least one `$ai_generation` event.\n"
            "- For comparison across cohorts, use the adoption rate rather than the raw count."
        )
    return (
        f"{top_line}\n\n"
        "**Key numbers**\n"
        f"- AI users: `{ai_users:,.0f}`\n"
        f"- Total users: `{total_users:,.0f}`\n"
        f"- AI adoption rate: `{adoption * 100:.1f}%`\n\n"
        "**How to read it**\n"
        "This measures whether a user has ever generated AI in the modeled dataset. "
        "It does not measure frequency, token volume, or credit consumption."
    )


def _metric_definition_response(defs: list[MetricDefinition]) -> str:
    if not defs:
        return (
            "I could not match that to an official semantic-layer metric yet. "
            "Try naming the metric, for example daily active users or AI adoption rate."
        )
    blocks = []
    for definition in defs[:1]:
        dimensions = (
            ", ".join(definition.allowed_dimensions)
            if definition.allowed_dimensions
            else "none declared"
        )
        blocks.append(
            "\n".join(
                [
                    f"**{definition.metric}**",
                    definition.definition or "No description is defined.",
                    f"- Type: `{definition.metric_type or 'unknown'}`",
                    f"- Source table: `{definition.source_table or 'unknown'}`",
                    f"- Measure: `{definition.measure or 'unknown'}`",
                    f"- Time column: `{definition.time_column or 'none'}`",
                    f"- Allowed dimensions: {dimensions}",
                ]
            )
        )
    return "\n\n".join(blocks)


def _dashboard_spec_from_metrics(
    question: str, defs: list[MetricDefinition]
) -> DashboardSpec:
    primary = defs[0] if defs else None
    metric = primary.metric if primary else "daily_active_users"
    theme = "Metric Monitoring"
    return {
        "dashboard_title": f"{metric.replace('_', ' ').title()} Monitor",
        "purpose": question,
        "theme": theme,
        "charts": [
            {
                "title": metric.replace("_", " ").title(),
                "type": "line",
                "metric": metric,
                "dimensions": [primary.time_column] if primary else ["event_date"],
                "filters": {},
            }
        ],
        "refresh_frequency": "manual",
        "recommended_alerts": [],
    }


def _dashboard_request_response(spec: DashboardSpec) -> str:
    return (
        f"I drafted a dashboard spec for **{spec['dashboard_title']}**. "
        "A live preview renders below — review it, edit the JSON if needed, then "
        "approve to save it to the Library."
    )


def _first_numeric_finding(
    findings: list[dict[str, Any]]
) -> tuple[str | None, float | None]:
    for finding in findings:
        for row in finding.get("rows", []):
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    return str(key), float(value)
    return None, None


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
