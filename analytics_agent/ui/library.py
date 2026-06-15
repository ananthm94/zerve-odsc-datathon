"""The "Library" tab — saved reports and approved dashboards, grouped for reuse."""

from __future__ import annotations

import streamlit as st

from analytics_agent import dashboard
from analytics_agent.dashboard.render import list_specs, render_spec
from analytics_agent.library.memory import ReportMemory
from analytics_agent.ui import common
from analytics_agent.ui.helpers import (
    dashboard_to_item,
    filter_items,
    format_item_date,
    group_items,
    library_counts,
    library_item_label,
    report_to_item,
)


def render() -> None:
    st.subheader("Library")
    st.caption("Approved reports and generated dashboards you can reuse later.")

    try:
        cfg = common.config()
        memory = ReportMemory(cfg)
        reports = [report_to_item(report) for report in memory.list_reports()]
        dashboards = [dashboard_to_item(spec) for spec in list_specs(cfg)]
    except Exception as exc:
        st.error(f"Could not load library: {exc}")
        st.stop()

    items = reports + dashboards
    report_count, dashboard_count, total_count = library_counts(items)
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Saved reports", report_count)
    sc2.metric("Dashboards", dashboard_count)
    sc3.metric("Total items", total_count)

    lc1, lc2 = st.columns([2, 1])
    query = lc1.text_input("Search library", placeholder="AI, credits, activation...")
    group_by = lc2.segmented_control(
        "Group by", ["theme", "month", "type"], default="theme"
    )

    grouped = group_items(filter_items(items, query), group_by)
    if not grouped:
        st.markdown(
            """
            <div class="za-guide">
              <strong>No saved work yet.</strong><br>
              Save an answer from Ask or approve a generated dashboard, and it
              will appear here for reuse.
            </div>
            """,
            unsafe_allow_html=True,
        )
    for group, group_items_list in grouped.items():
        st.markdown(f"### {group} ({len(group_items_list)})")
        for item in group_items_list:
            label = library_item_label(item)
            with st.expander(label):
                st.caption(
                    f"{item.status.title()} · {format_item_date(item.created_at)} "
                    f"· {item.theme or 'Other'}"
                )
                if item.summary:
                    st.write(item.summary)
                if item.metrics:
                    st.markdown(" ".join(f"`{metric}`" for metric in item.metrics))
                if item.item_type == "dashboard":
                    try:
                        render_spec(item.source, cfg, dashboard.DashboardFilters())
                    except Exception as exc:
                        st.error(f"Could not render dashboard: {exc}")
                else:
                    report = item.source
                    st.markdown(report.summary)
                    if report.sql_used:
                        with st.expander("SQL"):
                            for sql in report.sql_used:
                                st.code(sql, language="sql")
