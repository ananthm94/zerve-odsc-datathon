"""Zerve Analytics Agent — productized multi-tab UI (v0.7).

Tabs:
  * Ask         — autonomous agent with its reasoning streamed live
                  (plan -> query per sub-question -> reflect -> synthesize).
  * Dashboard   — pre-built metrics and charts over the dbt marts.
  * Experiments — observational cohort comparisons with a ship/no-ship narrative.
  * About       — architecture and status.

The Ask tab consumes ``graph.stream_analytics_events`` so the user watches the
agent think rather than staring at a spinner.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from analytics_agent import dashboard
from analytics_agent.config import ConfigError, load_config
from analytics_agent.experiments import (
    CohortSpec,
    format_comparison,
    narrate_experiment,
    run_continuous_experiment,
    run_proportion_experiment,
)
from analytics_agent.graph import (
    STAGE_PLAN,
    STAGE_QUERY,
    STAGE_REFLECT,
    STAGE_SYNTHESIZE,
    stream_analytics_events,
)
from analytics_agent.sql import SQLValidationError

load_dotenv()

st.set_page_config(page_title="Zerve Analytics Agent", layout="wide")
st.title("Zerve Analytics Agent")
st.caption("Autonomous analysis over a dbt semantic layer · DuckDB · RAG · LangGraph")


# --- Cached config + data --------------------------------------------------


@st.cache_resource
def _config():
    return load_config()


# Cached dashboard data. Each wrapper takes a hashable ``key`` (the filters'
# cache_key) plus the filters object passed with a leading underscore so
# Streamlit keys the cache on ``key`` and skips hashing the dataclass itself.


@st.cache_data(show_spinner=False)
def _kpis(key, _filters):
    return dashboard.kpis(_config(), _filters)


@st.cache_data(show_spinner=False)
def _daily_activity(key, _filters):
    return dashboard.daily_activity_df(_config(), _filters)


@st.cache_data(show_spinner=False)
def _tokens_by_model(key, _filters):
    return dashboard.ai_tokens_by_model_df(_config(), _filters)


@st.cache_data(show_spinner=False)
def _tokens_by_provider(key, _filters):
    return dashboard.ai_tokens_by_provider_df(_config(), _filters)


@st.cache_data(show_spinner=False)
def _activity_segments(key, _filters):
    return dashboard.activity_segment_df(_config(), _filters)


@st.cache_data(show_spinner=False)
def _users_by_role(key, _filters):
    return dashboard.users_by_role_df(_config(), _filters)


@st.cache_data(show_spinner=False)
def _exceptions_by_os(key, _filters):
    return dashboard.exceptions_by_os_df(_config(), _filters)


@st.cache_data(show_spinner=False)
def _feature_usage(key, _filters):
    return dashboard.feature_usage_df(_config(), _filters)


@st.cache_data(show_spinner=False)
def _feature_trend(key, _filters, categories):
    return dashboard.feature_trend_df(_config(), _filters, categories)


@st.cache_data(show_spinner=False)
def _retention_matrix(key, _filters, max_offset):
    return dashboard.retention_matrix_df(_config(), _filters, max_offset)


@st.cache_data(show_spinner=False)
def _retention_curve(key, _filters, max_offset):
    return dashboard.retention_curve_df(_config(), _filters, max_offset)


@st.cache_data(show_spinner=False)
def _funnel(key, _filters, steps, ordered):
    return dashboard.funnel_df(_config(), _filters, steps, ordered)


@st.cache_data(show_spinner=False)
def _distinct(table: str, column: str):
    return dashboard.distinct_values(table, column, _config())


@st.cache_data(show_spinner=False)
def _date_bounds():
    return dashboard.date_bounds(_config())


def _build_filters() -> dashboard.DashboardFilters:
    """Render the sidebar filter controls and return the selected filters.

    These apply to the Dashboard tab. Selectors are populated from the warehouse.
    """
    st.sidebar.header("Dashboard filters")
    st.sidebar.caption("Apply to the Dashboard tab.")

    lo, hi = _date_bounds()
    date_start = date_end = None
    if lo and hi:
        picked = st.sidebar.date_input(
            "Date range", value=(lo, hi), min_value=lo, max_value=hi
        )
        if isinstance(picked, (list, tuple)) and len(picked) == 2:
            date_start, date_end = picked

    def _ms(label, table, column):
        return tuple(
            st.sidebar.multiselect(label, _distinct(table, column))
        )

    filters = dashboard.DashboardFilters(
        date_start=date_start,
        date_end=date_end,
        user_roles=_ms("User role", "dim_users", "user_role"),
        activity_segments=_ms("Activity segment", "user_summary", "activity_segment"),
        continents=_ms("Continent", "dim_users", "continent"),
        countries=_ms("Country", "dim_users", "country"),
        device_types=_ms("Device type", "dim_users", "device_type"),
        operating_systems=_ms("Operating system", "dim_users", "os"),
    )
    if filters.has_any:
        st.sidebar.success("Filters active")
    return filters


def _source_label(doc) -> str:
    name = doc.metadata.get("model_name") or doc.metadata.get("name") or "unknown"
    return f"{name} ({doc.metadata.get('doc_type', 'document')})"


def _fmt(n: float) -> str:
    return f"{n:,.0f}"


# Funnel presets: ordered (label, event_names) steps grounded in the event feed.
_FUNNEL_PRESETS = {
    "Activation": (
        ("Sign in", ("sign_in",)),
        ("Onboarding", ("skip_onboarding_form",)),
        ("Run block", ("run_block",)),
        ("AI generation", ("$ai_generation",)),
        ("Use credits", ("credits_used",)),
    ),
    "Agent workflow": (
        ("Agent created", ("agent_worker_created",)),
        (
            "Tool call",
            (
                "agent_tool_call_create_block_tool",
                "agent_tool_call_run_block_tool",
                "agent_tool_call_get_block_tool",
            ),
        ),
        ("Finish ticket", ("agent_tool_call_finish_ticket_tool",)),
    ),
}


ask_tab, dash_tab, exp_tab, about_tab = st.tabs(
    ["💬 Ask", "📊 Dashboard", "🧪 Experiments", "ℹ️ About"]
)


# === Ask ====================================================================

with ask_tab:
    st.subheader("Ask a question")
    st.write(
        "The agent plans an analysis, runs read-only SQL for each sub-question, "
        "reflects on the evidence, then synthesizes a cited report. Its stages "
        "stream below as they happen."
    )

    with st.form("ask_form"):
        question = st.text_input(
            "Question",
            placeholder="What's driving AI credit consumption?",
        )
        submitted = st.form_submit_button("Ask", type="primary")

    if submitted:
        if not question.strip():
            st.warning("Enter a question.")
            st.stop()

        final_state: dict = {}
        try:
            # Each event corresponds to one completed graph node; render it live.
            for event in stream_analytics_events(question):
                stage = event["stage"]

                if stage == STAGE_PLAN:
                    with st.status("🧭 Planning the analysis…", state="complete"):
                        if event["hypotheses"]:
                            st.markdown("**Hypotheses**")
                            for h in event["hypotheses"]:
                                st.markdown(f"- {h}")
                        st.markdown("**Sub-questions**")
                        for i, q in enumerate(event["sub_questions"], 1):
                            st.markdown(f"{i}. {q}")

                elif stage == STAGE_QUERY:
                    finding = event["finding"]
                    failed = bool(finding.get("error"))
                    label = (
                        f"{'⚠️' if failed else '🔍'} [{finding['id']}] "
                        f"{finding['sub_question']}"
                    )
                    with st.status(label, state="error" if failed else "complete"):
                        if failed:
                            st.error(finding["error"])
                        else:
                            st.code(finding["sql"], language="sql")
                            st.caption(f"{finding['row_count']} row(s)")
                            if finding["rows"]:
                                st.dataframe(
                                    pd.DataFrame(finding["rows"]),
                                    use_container_width=True,
                                    hide_index=True,
                                )

                elif stage == STAGE_REFLECT:
                    with st.status("🤔 Reflecting on the evidence…", state="complete"):
                        st.write(event["reflection"] or "Proceeding to synthesis.")

                elif stage == STAGE_SYNTHESIZE:
                    final_state = event.get("state", {})
                    st.markdown("### Report")
                    st.markdown(event["report"] or "No report produced.")

        except ConfigError as exc:
            st.error(str(exc))
            st.stop()
        except SQLValidationError as exc:
            st.error(f"SQL validation failed: {exc}")
            st.stop()
        except Exception as exc:  # surface any agent/runtime error to the user
            st.error(f"Analysis failed: {exc}")
            st.stop()

        if final_state:
            with st.expander("Sources (retrieved semantic-layer context)"):
                docs = final_state.get("context_documents", [])
                if not docs:
                    st.caption("No sources retrieved.")
                for doc in docs:
                    st.markdown(f"**{_source_label(doc)}**")
                    st.text(doc.page_content)


# === Dashboard ==============================================================

with dash_tab:
    st.subheader("Product analytics overview")

    try:
        filters = _build_filters()
        fkey = filters.cache_key
        k = _kpis(fkey, filters)
        daily = _daily_activity(fkey, filters)
        seg = _activity_segments(fkey, filters)
        by_model = _tokens_by_model(fkey, filters)
        by_provider = _tokens_by_provider(fkey, filters)
        by_role = _users_by_role(fkey, filters)
        exc_os = _exceptions_by_os(fkey, filters)
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not load dashboard data: {exc}")
        st.stop()

    if filters.has_any:
        st.caption("Showing a filtered view — see the sidebar.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Users", _fmt(k["total_users"]))
    c2.metric("Events", _fmt(k["total_events"]))
    c3.metric("AI generations", _fmt(k["ai_generations"]))
    c4.metric("AI adoption", f"{k['ai_adoption_rate'] * 100:.1f}%")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("AI tokens", _fmt(k["total_ai_tokens"]))
    c6.metric("Credits used", _fmt(k["total_credits_used"]))
    c7.metric("Exceptions", _fmt(k["total_exceptions"]))
    c8.metric("AI users", _fmt(k["ai_users"]))

    st.divider()

    st.markdown("#### Daily active users")
    if not daily.empty:
        st.line_chart(daily, x="event_date", y="active_users", height=260)

        st.markdown("#### Daily event mix")
        st.area_chart(
            daily,
            x="event_date",
            y=["ai_generations", "exceptions", "pageviews", "credit_events"],
            height=260,
        )

    left, right = st.columns(2)
    with left:
        st.markdown("#### AI tokens by model")
        if not by_model.empty:
            st.bar_chart(by_model, x="ai_model", y="total_tokens", height=300)
        st.markdown("#### Users by role")
        if not by_role.empty:
            st.bar_chart(by_role, x="user_role", y="users", height=300)
    with right:
        st.markdown("#### AI tokens by provider")
        if not by_provider.empty:
            st.bar_chart(by_provider, x="ai_provider", y="total_tokens", height=300)
        st.markdown("#### Exceptions by OS")
        if not exc_os.empty:
            st.bar_chart(exc_os, x="os", y="exceptions", height=300)

    st.markdown("#### Activity segments")
    if not seg.empty:
        st.dataframe(seg, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Feature usage")
    feat = _feature_usage(fkey, filters)
    if feat.empty:
        st.caption("No feature events in scope.")
    else:
        fl, fr = st.columns([3, 2])
        with fl:
            st.markdown("#### Events by feature")
            st.bar_chart(feat, x="feature_category", y="events", height=320)
        with fr:
            st.markdown("#### Reach & adoption")
            display = feat.assign(
                adoption=(feat["adoption_rate"] * 100).round(1)
            )[["feature_category", "reach", "adoption"]]
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "reach": st.column_config.NumberColumn("Users reached"),
                    "adoption": st.column_config.NumberColumn(
                        "Adoption %", format="%.1f%%"
                    ),
                },
            )

        st.markdown("#### Usage trend")
        all_cats = feat["feature_category"].tolist()
        default_cats = all_cats[:4]
        chosen = st.multiselect(
            "Feature areas", all_cats, default=default_cats, key="feature_trend_cats"
        )
        if chosen:
            trend = _feature_trend(fkey, filters, tuple(chosen))
            if not trend.empty:
                st.line_chart(
                    trend, x="event_date", y="events", color="feature_category",
                    height=300,
                )

    st.divider()
    st.markdown("### Weekly cohort retention")
    st.caption(
        "Cohort = the user's first active week. Retention at week N is the share "
        "of the cohort active N weeks later. Week 0 is 100% by definition."
    )
    max_offset = st.slider("Weeks tracked", 4, 24, 12, key="retention_weeks")
    curve = _retention_curve(fkey, filters, max_offset)
    matrix = _retention_matrix(fkey, filters, max_offset)

    if curve.empty:
        st.caption("No retention data in scope.")
    else:
        ret = curve.set_index("week_offset")["retention"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Week 1", f"{ret.get(1, 0) * 100:.1f}%")
        m2.metric("Week 4", f"{ret.get(4, 0) * 100:.1f}%")
        m3.metric(
            f"Week {max_offset}", f"{ret.get(max_offset, 0) * 100:.1f}%"
        )

        st.markdown("#### Retention curve")
        curve_plot = curve.assign(retention_pct=(curve["retention"] * 100).round(2))
        st.line_chart(curve_plot, x="week_offset", y="retention_pct", height=240)

        if not matrix.empty:
            st.markdown("#### Retention triangle")
            heat = matrix.assign(
                cohort=matrix["cohort_week"].dt.strftime("%Y-%m-%d"),
                retention_pct=(matrix["retention"] * 100).round(1),
            )
            chart = (
                alt.Chart(heat)
                .mark_rect()
                .encode(
                    x=alt.X("week_offset:O", title="Weeks since first active"),
                    y=alt.Y("cohort:O", title="Cohort (first week)", sort="descending"),
                    color=alt.Color(
                        "retention_pct:Q",
                        title="Retention %",
                        scale=alt.Scale(scheme="blues"),
                    ),
                    tooltip=[
                        alt.Tooltip("cohort:N", title="Cohort"),
                        alt.Tooltip("week_offset:O", title="Week"),
                        alt.Tooltip("active_users:Q", title="Active users"),
                        alt.Tooltip("cohort_size:Q", title="Cohort size"),
                        alt.Tooltip("retention_pct:Q", title="Retention %"),
                    ],
                )
                .properties(height=22 * heat["cohort"].nunique() + 40)
            )
            st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.markdown("### Conversion funnel")
    fc1, fc2 = st.columns([2, 1])
    preset_name = fc1.selectbox("Funnel", list(_FUNNEL_PRESETS))
    ordered = fc2.toggle(
        "Ordered", value=True, help="Require steps to occur in sequence (first-touch)."
    )
    steps = _FUNNEL_PRESETS[preset_name]
    fn = _funnel(fkey, filters, steps, ordered)

    if fn.empty or fn["users"].iloc[0] == 0:
        st.caption("No users enter this funnel in the current scope.")
    else:
        top = fn["users"].iloc[0]
        fn = fn.assign(
            of_start=(fn["users"] / top * 100).round(1),
            step_conv=(fn["users"] / fn["users"].shift(1) * 100).round(1),
        )
        funnel_chart = (
            alt.Chart(fn)
            .mark_bar()
            .encode(
                y=alt.Y("label:N", sort=fn["label"].tolist(), title=None),
                x=alt.X("users:Q", title="Users"),
                tooltip=[
                    alt.Tooltip("label:N", title="Step"),
                    alt.Tooltip("users:Q", title="Users", format=","),
                    alt.Tooltip("of_start:Q", title="% of start"),
                    alt.Tooltip("step_conv:Q", title="% of previous"),
                ],
            )
            .properties(height=55 * len(fn) + 20)
        )
        st.altair_chart(funnel_chart, use_container_width=True)
        st.dataframe(
            fn[["step", "label", "users", "of_start", "step_conv"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "of_start": st.column_config.NumberColumn("% of start", format="%.1f%%"),
                "step_conv": st.column_config.NumberColumn(
                    "% of previous", format="%.1f%%"
                ),
            },
        )


# === Experiments ============================================================

with exp_tab:
    st.subheader("Observational cohort comparison")
    st.info(
        "The dataset has **no experiment-assignment column**, so these are "
        "**quasi-experiments**: cohorts are defined by an existing user attribute, "
        "and differences may be confounded by self-selection. Read the narrative's "
        "caveats before acting.",
        icon="⚠️",
    )

    test_kind = st.radio(
        "Comparison type",
        ["Continuous metric", "Proportion / rate"],
        horizontal=True,
    )

    # Cohorts are defined by a user attribute on user_summary or dim_users.
    dimension_table = st.selectbox(
        "Cohort dimension table", ["user_summary", "dim_users"]
    )
    dim_options = {
        "user_summary": ["activity_segment", "is_ai_user"],
        "dim_users": ["user_role", "cloud_provider", "continent", "device_type"],
    }[dimension_table]
    group_column = st.selectbox("Cohort dimension", dim_options)

    try:
        values = _distinct(dimension_table, group_column)
    except Exception as exc:
        st.error(f"Could not load cohort values: {exc}")
        st.stop()

    if len(values) < 2:
        st.warning("Need at least two distinct values in this dimension.")
        st.stop()

    cc1, cc2 = st.columns(2)
    cohort_a = cc1.selectbox("Cohort A", values, index=0)
    cohort_b = cc2.selectbox("Cohort B", values, index=1)

    if test_kind == "Continuous metric":
        metric_column = st.selectbox(
            "Metric (per user)",
            [
                "total_ai_tokens",
                "total_event_count",
                "total_credits_used",
                "avg_ai_latency_seconds",
                "exception_count",
                "user_lifetime_days",
            ],
        )
    else:
        success_label = st.selectbox(
            "Success definition",
            ["is_ai_user", "total_credits_used > 0", "exception_count > 0"],
        )

    run = st.button("Run comparison", type="primary")

    if run:
        if cohort_a == cohort_b:
            st.warning("Pick two different cohorts.")
            st.stop()
        try:
            cfg = _config()
            # Cohort attributes live on user_summary or dim_users; the per-user
            # metric always lives on user_summary, joined on user_id.
            metric_table = "user_summary"
            group_table = dimension_table
            with st.spinner("Sampling cohorts and running statistics…"):
                if test_kind == "Continuous metric":
                    spec = CohortSpec(
                        metric_table=metric_table,
                        group_table=group_table,
                        group_column=group_column,
                        cohort_a=str(cohort_a),
                        cohort_b=str(cohort_b),
                        metric_column=metric_column,
                    )
                    result = run_continuous_experiment(cfg, spec)
                else:
                    spec = CohortSpec(
                        metric_table=metric_table,
                        group_table=group_table,
                        group_column=group_column,
                        cohort_a=str(cohort_a),
                        cohort_b=str(cohort_b),
                        success_expr=success_label,
                    )
                    result = run_proportion_experiment(cfg, spec)
                narrative = narrate_experiment(cfg, result)
        except Exception as exc:
            st.error(f"Experiment failed: {exc}")
            st.stop()

        st.markdown("#### Recommendation")
        st.markdown(narrative)
        with st.expander("Statistical detail", expanded=True):
            st.code(format_comparison(result), language="text")


# === About ==================================================================

with about_tab:
    st.subheader("About")
    st.markdown(
        """
This is an **AI-native analytics platform** over the Zerve product-analytics
dataset (~3.5M events, ~17.5k users).

**Pipeline:** raw events → dbt staging + marts + a MetricFlow **semantic layer**
→ embedded into **Qdrant** → an autonomous **LangGraph** agent
(plan → query → reflect → synthesize) with read-only SQL guardrails, querying
**DuckDB** by default (BigQuery optional).

**Status:** v0.1–v0.6 complete (multi-provider LLM, DuckDB backend, semantic
layer, autonomous agent, experiment analysis); v0.7 is this productized UI.
See the README for the full architecture and roadmap.
        """
    )
