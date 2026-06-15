"""The "Experiments" tab — observational cohort comparisons with a ship/no-ship
narrative. These are observational, not randomized A/B tests; the UI preserves
that caveat throughout.
"""

from __future__ import annotations

import streamlit as st

from analytics_agent.experiments import (
    CohortSpec,
    format_comparison,
    narrate_experiment,
    run_continuous_experiment,
    run_proportion_experiment,
)
from analytics_agent.ui import common


def render() -> None:
    st.subheader("Observational cohort comparison")
    st.markdown(
        """
        <div class="za-guide">
          <strong>Use this to compare existing user cohorts.</strong><br>
          These are observational comparisons, not randomized A/B tests. Treat
          the recommendation as directional and read the statistical detail
          before making a product decision.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### Step 1 - Choose comparison type")
    test_kind = st.radio(
        "Comparison type",
        ["Continuous metric", "Proportion / rate"],
        horizontal=True,
    )

    # Cohorts are defined by a user attribute on user_summary or dim_users.
    st.markdown("#### Step 2 - Define cohorts")
    dimension_table = st.selectbox(
        "Cohort dimension table", ["user_summary", "dim_users"]
    )
    dim_options = {
        "user_summary": ["activity_segment", "is_ai_user"],
        "dim_users": ["user_role", "cloud_provider", "continent", "device_type"],
    }[dimension_table]
    group_column = st.selectbox("Cohort dimension", dim_options)

    try:
        values = common.distinct(dimension_table, group_column)
    except Exception as exc:
        st.error(f"Could not load cohort values: {exc}")
        st.stop()

    if len(values) < 2:
        st.warning("Need at least two distinct values in this dimension.")
        st.stop()

    cc1, cc2 = st.columns(2)
    cohort_a = cc1.selectbox("Cohort A", values, index=0)
    cohort_b = cc2.selectbox("Cohort B", values, index=1)

    st.markdown("#### Step 3 - Choose outcome")
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

    st.markdown("#### Step 4 - Run comparison")
    run = st.button("Run comparison", type="primary")

    if run:
        if cohort_a == cohort_b:
            st.warning("Pick two different cohorts.")
            st.stop()
        try:
            cfg = common.config()
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
        with st.expander("Statistical detail", expanded=False):
            st.code(format_comparison(result), language="text")
