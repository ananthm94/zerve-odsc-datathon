"""The "Ask" tab — a conversational chat over the agentic trust loop.

Routes intent -> grounds in the semantic layer -> plans -> queries -> reflects
-> synthesizes -> verifies against snapshots. Each answer leads; the reasoning
streams into a status that auto-collapses, so the thought process is one click
away. Follow-ups are condensed against the conversation. Answers can be saved to
the Library and generated dashboards approved.
"""

from __future__ import annotations

import json
from uuid import uuid4

import pandas as pd
import streamlit as st

from analytics_agent import dashboard
from analytics_agent.ask.graph import (
    STAGE_CONDENSE,
    STAGE_FINAL,
    STAGE_INTENT,
    STAGE_MEMORY,
    STAGE_PLAN,
    STAGE_QUERY,
    STAGE_REFLECT,
    STAGE_RESPONSE_MODE,
    STAGE_SEMANTIC,
    STAGE_SNAPSHOT,
    STAGE_SYNTHESIZE,
    stream_agentic_events,
)
from analytics_agent.ask.snapshots import SnapshotStore, promote_snapshot
from analytics_agent.core.config import ConfigError
from analytics_agent.core.contracts import ReportRecord, assign_theme
from analytics_agent.core.sql import SQLValidationError
from analytics_agent.dashboard.render import render_spec, save_spec
from analytics_agent.library.memory import ReportMemory
from analytics_agent.ui import common
from analytics_agent.ui.helpers import example_prompts


def _source_label(doc) -> str:
    name = doc.metadata.get("model_name") or doc.metadata.get("name") or "unknown"
    return f"{name} ({doc.metadata.get('doc_type', 'document')})"


def _render_result_preview(rows: list[dict]) -> None:
    if not rows:
        return
    df = pd.DataFrame(rows)
    time_cols = [
        col
        for col in df.columns
        if any(token in col.lower() for token in ("date", "month", "week", "period"))
    ]
    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col not in time_cols
    ]
    if time_cols and numeric_cols and len(df) > 1:
        x_col = time_cols[0]
        y_cols = numeric_cols[:3]
        st.line_chart(df, x=x_col, y=y_cols, height=240)
    with st.expander("Result table", expanded=not (time_cols and numeric_cols)):
        st.dataframe(df, use_container_width=True, hide_index=True)


def _metric_and_value_from_state(state: dict) -> tuple[str | None, float | None]:
    if state.get("rows"):
        for row in state["rows"]:
            for key, value in row.items():
                if isinstance(value, (int, float)) and key != "total_users":
                    return str(key), float(value)
    for finding in state.get("findings", []) or []:
        for row in finding.get("rows", []) or []:
            for key, value in row.items():
                if isinstance(value, (int, float)):
                    return str(key), float(value)
    return None, None


def _report_record_from_state(state: dict) -> ReportRecord:
    metric_defs = state.get("metric_definitions", []) or []
    metrics = [getattr(metric, "metric", "") for metric in metric_defs if getattr(metric, "metric", "")]
    findings = state.get("findings", []) or []
    sql_used = [finding.get("sql", "") for finding in findings if finding.get("sql")]
    source_tables = []
    for metric in metric_defs:
        table = getattr(metric, "source_table", "")
        if table:
            source_tables.append(table)
    for finding in findings:
        for table in finding.get("referenced_tables", []) or []:
            source_tables.append(table)
    summary = state.get("final_response") or state.get("report") or state.get("answer") or ""
    return ReportRecord(
        id=f"report-{uuid4().hex[:12]}",
        question=state.get("question", ""),
        summary=summary,
        metrics_used=metrics,
        sql_used=sql_used,
        date_range="",
        source_tables=sorted(set(source_tables)),
        approval_status="approved",
        tags=[],
        theme=assign_theme(metrics),
        created_at=common.now(),
    )


def _render_model_usage(state: dict) -> None:
    usage = state.get("model_usage")
    if not usage:
        return
    with st.expander("Model usage and estimated cost"):
        st.metric(
            "Estimated model cost",
            f"${usage.get('estimated_cost_usd', 0.0):.6f}",
        )
        st.caption(
            f"Input tokens: {usage.get('input_tokens', 0):,} · "
            f"Output tokens: {usage.get('output_tokens', 0):,}"
        )
        calls = usage.get("calls", [])
        if calls:
            st.dataframe(
                pd.DataFrame(calls),
                use_container_width=True,
                hide_index=True,
            )


def _render_reasoning_event(event: dict) -> None:
    """Render one thought-process stage. Called inside a status/expander container.

    Shared by the live stream and the history replay so a turn looks identical
    whether it was just produced or is being re-rendered on a later rerun.
    """
    stage = event["stage"]
    if stage == STAGE_CONDENSE:
        st.markdown(f"**Interpreted as:** {event['standalone_question']}")
        st.caption(f"From your follow-up: “{event['original_question']}”")
    elif stage == STAGE_INTENT:
        st.markdown(f"**Intent:** {event['intent'].replace('_', ' ').title()}")
    elif stage == STAGE_RESPONSE_MODE:
        st.markdown(
            f"**Response depth:** {event['response_mode'].replace('_', ' ').title()}"
        )
        if event.get("description"):
            st.caption(event["description"])
    elif stage == STAGE_SEMANTIC:
        st.markdown("**Semantic grounding**")
        metric_defs = event.get("metric_definitions", [])
        if metric_defs:
            for metric in metric_defs[:3]:
                st.markdown(f"- **{metric.metric}** · `{metric.source_table}`")
                st.caption(metric.definition)
        else:
            st.caption("No official metric match found.")
    elif stage == STAGE_MEMORY:
        st.markdown("**Report memory**")
        recalled = event.get("recalled_reports", [])
        if recalled:
            for report in recalled:
                st.markdown(f"- {report.question}")
        else:
            st.caption("No approved prior reports recalled.")
    elif stage == STAGE_PLAN:
        if event.get("hypotheses"):
            st.markdown("**Hypotheses**")
            for hypothesis in event["hypotheses"]:
                st.markdown(f"- {hypothesis}")
        st.markdown("**Sub-questions**")
        for i, sub_q in enumerate(event.get("sub_questions", []), 1):
            st.markdown(f"{i}. {sub_q}")
    elif stage == STAGE_QUERY:
        finding = event["finding"]
        failed = bool(finding.get("error"))
        st.markdown(
            f"{'⚠️' if failed else '🔍'} **[{finding['id']}]** {finding['sub_question']}"
        )
        if failed:
            st.error(finding["error"])
        else:
            st.caption(f"{finding['row_count']} row(s)")
            if finding["rows"]:
                _render_result_preview(finding["rows"])
            with st.expander("SQL"):
                st.code(finding["sql"], language="sql")
    elif stage == STAGE_REFLECT:
        reflection = event.get("reflection") or "Proceeding to synthesis."
        st.markdown(f"**Reflection:** {reflection}")
    elif stage == STAGE_SNAPSHOT:
        check = event.get("snapshot_check", {})
        st.markdown("**Snapshot verification**")
        if check.get("baseline") is None:
            st.caption("No trusted snapshot baseline found yet.")
        else:
            drift = check.get("pct_diff")
            drift_label = f"{drift * 100:.1f}%" if drift is not None else "n/a"
            st.write(f"Baseline `{check['baseline']}`; drift {drift_label}.")
            if check.get("flag"):
                st.warning("Snapshot drift exceeded the threshold.")


def _history_pairs(chat_history: list[dict]) -> list[dict]:
    """Build {question, answer} pairs from completed turns for follow-up context.

    The just-added user turn has no assistant reply yet, so it is naturally
    excluded. Capped to the most recent few exchanges to bound token use.
    """
    pairs: list[dict] = []
    pending_question = None
    for turn in chat_history:
        if turn["role"] == "user":
            pending_question = turn["content"]
        elif turn["role"] == "assistant" and pending_question is not None:
            pairs.append(
                {"question": pending_question, "answer": turn.get("answer", "")}
            )
            pending_question = None
    return pairs[-6:]


def _save_report_controls(final_state: dict, idx: int) -> None:
    if not (final_state.get("final_response") or final_state.get("report")):
        return
    save_col, snapshot_col = st.columns([1, 3])
    if save_col.button("Save report", key=f"save_report_{idx}"):
        try:
            record = _report_record_from_state(final_state)
            ReportMemory(common.config()).save_report(record)
        except Exception as exc:
            st.error(f"Could not save report: {exc}")
        else:
            st.success("Saved report to Library.")
            metric, value = _metric_and_value_from_state(final_state)
            if metric and value is not None:
                promote_snapshot(metric, value, SnapshotStore(common.config()))
                snapshot_col.caption(
                    f"Promoted `{metric}` snapshot from this report."
                )


def _render_assistant_turn(turn: dict, idx: int) -> None:
    """Re-render a stored assistant turn: answer first, reasoning collapsed."""
    st.markdown(turn.get("answer") or "_No answer was produced._")
    reasoning = turn.get("reasoning", [])
    if reasoning:
        with st.status("🧠 Show reasoning", state="complete", expanded=False):
            for event in reasoning:
                _render_reasoning_event(event)
    final_state = turn.get("final_state", {}) or {}
    if final_state:
        _render_model_usage(final_state)
        _save_report_controls(final_state, idx)
        with st.expander("Sources (retrieved semantic-layer context)"):
            docs = final_state.get("context_documents", [])
            if not docs:
                st.caption("No sources retrieved.")
            for doc in docs:
                st.markdown(f"**{_source_label(doc)}**")
                st.text(doc.page_content)


def _run_live_turn(question: str, history: list[dict]) -> dict | None:
    """Stream one agent run live, then return the assistant turn for the thread.

    Reasoning streams inside an ``st.status`` that auto-collapses on completion,
    so the answer leads and the thought process is one click away.
    """
    reasoning_events: list[dict] = []
    answer = ""
    final_state: dict = {}
    dashboard_spec = None
    try:
        with st.status("Thinking…", expanded=True) as status:
            for event in stream_agentic_events(question, history=history or None):
                stage = event["stage"]
                if stage == STAGE_SYNTHESIZE:
                    final_state = event.get("state", {}) or final_state
                    answer = event.get("report", "") or answer
                elif stage == STAGE_FINAL:
                    final_state = event.get("state", {}) or final_state
                    if event.get("final_response"):
                        answer = event["final_response"]
                    if event.get("dashboard_spec"):
                        dashboard_spec = event["dashboard_spec"]
                else:
                    reasoning_events.append(event)
                    _render_reasoning_event(event)
            status.update(label="🧠 Show reasoning", state="complete", expanded=False)
    except ConfigError as exc:
        st.error(str(exc))
        return None
    except SQLValidationError as exc:
        st.error(f"SQL validation failed: {exc}")
        return None
    except Exception as exc:  # surface any agent/runtime error to the user
        st.error(f"Analysis failed: {exc}")
        return None

    st.markdown(answer or "_No answer was produced._")
    if dashboard_spec:
        st.session_state["pending_dashboard_spec"] = dashboard_spec
    return {
        "role": "assistant",
        "answer": answer,
        "reasoning": reasoning_events,
        "final_state": final_state,
        "dashboard_spec": dashboard_spec,
    }


def render() -> None:
    head_col, clear_col = st.columns([4, 1])
    head_col.subheader("Ask the data")
    st.session_state.setdefault("chat_history", [])
    if st.session_state["chat_history"]:
        if clear_col.button("Clear conversation"):
            st.session_state["chat_history"] = []
            st.session_state.pop("pending_dashboard_spec", None)
            st.rerun()

    if not st.session_state["chat_history"]:
        st.markdown(
            """
            <div class="za-guide">
              <strong>Start with a product question.</strong><br>
              Ask in plain English. The agent will choose trusted metrics, run
              read-only SQL, and keep reasoning, sources, and cost details
              available without putting them in your way.
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption("Try one of these:")
        prompt_cols = st.columns(2)
        for i, example in enumerate(example_prompts()):
            if prompt_cols[i % 2].button(example, key=f"example_prompt_{i}"):
                st.session_state["queued_prompt"] = example
                st.rerun()

    # Replay the conversation so far (oldest first).
    for turn_index, turn in enumerate(st.session_state["chat_history"]):
        if turn["role"] == "user":
            with st.chat_message("user"):
                st.markdown(turn["content"])
        else:
            with st.chat_message("assistant"):
                _render_assistant_turn(turn, turn_index)

    # New question.
    queued_prompt = st.session_state.pop("queued_prompt", None)
    typed_prompt = st.chat_input("Ask about the data...")
    prompt = queued_prompt or typed_prompt
    if prompt and prompt.strip():
        question = prompt.strip()
        st.session_state["chat_history"].append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        history = _history_pairs(st.session_state["chat_history"])
        with st.chat_message("assistant"):
            assistant_turn = _run_live_turn(question, history)

        if assistant_turn is None:
            # The run failed; drop the dangling user turn and keep the error visible.
            st.session_state["chat_history"].pop()
        else:
            st.session_state["chat_history"].append(assistant_turn)
            st.rerun()

    pending_spec = st.session_state.get("pending_dashboard_spec")
    if pending_spec:
        st.divider()
        st.markdown("### Generated dashboard preview")
        st.caption(
            "Review the generated charts before saving them to the Library."
        )
        try:
            render_spec(pending_spec, common.config(), dashboard.DashboardFilters())
        except Exception as exc:
            st.error(f"Could not render generated dashboard: {exc}")
        with st.expander("Advanced: edit dashboard JSON"):
            edited_json = st.text_area(
                "Dashboard spec",
                value=json.dumps(pending_spec, indent=2),
                height=220,
                label_visibility="collapsed",
            )
            if st.button("Apply dashboard edits"):
                try:
                    st.session_state["pending_dashboard_spec"] = json.loads(edited_json)
                except ValueError as exc:
                    st.error(f"Invalid JSON: {exc}")
                else:
                    st.success("Dashboard edits applied.")
                    st.rerun()
        c_save, c_reject, c_clear = st.columns([1, 1, 3])
        if c_save.button("Approve and save", type="primary"):
            try:
                approved = dict(st.session_state["pending_dashboard_spec"])
                approved["approval_status"] = "approved"
                approved.setdefault("created_at", common.now())
                path = save_spec(approved, common.config())
            except Exception as exc:
                st.error(f"Could not save dashboard: {exc}")
            else:
                st.success(f"Saved dashboard spec to {path}")
        if c_reject.button("Reject"):
            st.session_state.pop("pending_dashboard_spec", None)
            st.info("Rejected dashboard preview.")
            st.rerun()
        if c_clear.button("Clear preview"):
            st.session_state.pop("pending_dashboard_spec", None)
            st.rerun()
