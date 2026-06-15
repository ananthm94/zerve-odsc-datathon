"""Render and persist generated dashboard specs.

The generated dashboard spec stays declarative. This module maps each chart in a
spec to trusted, author-written dashboard helpers instead of letting generated
specs create arbitrary SQL.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from analytics_agent import dashboard
from analytics_agent.core.config import AgentConfig
from analytics_agent.core.contracts import DashboardSpec
from analytics_agent.core.llm import build_chat_model


MAX_CHART_REPAIRS = 2

_DEFAULT_FUNNEL_STEPS = (
    ("Sign in", ("sign_in",)),
    ("Onboarding", ("skip_onboarding_form",)),
    ("Run block", ("run_block",)),
    ("AI generation", ("$ai_generation",)),
    ("Use credits", ("credits_used",)),
)


@dataclass(frozen=True)
class ResolvedChart:
    chart: dict[str, Any]
    dataframe: Any
    render_type: str
    x: str | None = None
    y: str | list[str] | None = None
    note: str = ""


def resolve_chart_data(
    chart: dict[str, Any],
    config: AgentConfig,
    filters: dashboard.DashboardFilters | None = None,
) -> ResolvedChart:
    """Resolve one chart spec to a trusted dashboard helper result."""
    filters = filters or dashboard.DashboardFilters()
    chart_type = str(chart.get("type", "")).lower()
    metric = str(chart.get("metric", "")).lower()
    dimensions = [str(dim).lower() for dim in chart.get("dimensions", []) or []]

    if chart_type == "line":
        return _resolve_line(chart, metric, dimensions, config, filters)
    if chart_type == "bar":
        return _resolve_bar(chart, metric, dimensions, config, filters)
    if chart_type == "funnel":
        return _resolve_funnel(chart, config, filters)
    if chart_type == "heatmap":
        return _resolve_heatmap(chart, config, filters)
    raise ValueError(f"Unsupported chart type: {chart_type or '(missing)'}")


def render_spec(
    spec: DashboardSpec,
    config: AgentConfig,
    filters: dashboard.DashboardFilters | None = None,
) -> None:
    """Render a dashboard spec in Streamlit."""
    import streamlit as st

    filters = filters or dashboard.DashboardFilters()
    st.markdown(f"### {spec.get('dashboard_title', 'Generated dashboard')}")
    if spec.get("purpose"):
        st.caption(spec["purpose"])

    for chart in spec.get("charts", []) or []:
        repaired = chart
        resolved = None
        last_error = ""
        for _ in range(MAX_CHART_REPAIRS + 1):
            try:
                resolved = resolve_chart_data(repaired, config, filters)
                break
            except Exception as exc:
                last_error = str(exc)
                next_chart = _repair_chart_spec(repaired, last_error, config)
                if next_chart == repaired:
                    break
                repaired = next_chart

        title = repaired.get("title") or chart.get("title") or "Chart"
        st.markdown(f"#### {title}")
        if resolved is None:
            st.warning(f"Could not render this chart: {last_error}")
            continue
        _render_resolved_chart(resolved, st)


def save_spec(spec: DashboardSpec, config: AgentConfig) -> str:
    """Persist a dashboard spec as JSON and return the file path."""
    dashboards_dir = Path(config.agent_store_dir) / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    saved = dict(spec)
    saved.setdefault("created_at", _now())
    saved.setdefault("id", _slug(saved.get("dashboard_title", "dashboard")))
    path = dashboards_dir / f"{saved['id']}.json"
    path.write_text(json.dumps(saved, indent=2, sort_keys=True))
    return str(path)


def list_specs(config: AgentConfig) -> list[dict]:
    """List saved dashboard specs from the local store."""
    dashboards_dir = Path(config.agent_store_dir) / "dashboards"
    if not dashboards_dir.exists():
        return []
    specs = []
    for path in sorted(dashboards_dir.glob("*.json")):
        try:
            item = json.loads(path.read_text())
        except ValueError:
            continue
        item["_path"] = str(path)
        specs.append(item)
    return sorted(
        specs,
        key=lambda item: (item.get("created_at", ""), item.get("dashboard_title", "")),
        reverse=True,
    )


def _resolve_line(chart, metric, dimensions, config, filters) -> ResolvedChart:
    if _has_any(metric, "daily_active_users", "active_users", "total_events", "events"):
        df = dashboard.daily_activity_df(config, filters)
        y = _first_existing(
            df,
            [
                _metric_to_daily_column(metric),
                "active_users",
                "total_events",
            ],
        )
        return ResolvedChart(chart, df, "line", x="event_date", y=y)
    if _has_any(metric, "feature", "usage") or "feature_category" in dimensions:
        categories = tuple(chart.get("filters", {}).get("categories", ()) or ())
        df = dashboard.feature_trend_df(config, filters, categories=categories)
        return ResolvedChart(chart, df, "line", x="event_date", y="events")
    if _has_any(metric, "retention") or "week_offset" in dimensions:
        df = dashboard.retention_curve_df(config, filters)
        return ResolvedChart(chart, df, "line", x="week_offset", y="retention")
    raise ValueError(f"No line-chart mapping for metric {metric!r}.")


def _resolve_bar(chart, metric, dimensions, config, filters) -> ResolvedChart:
    if _has_any(metric, "token", "ai_tokens", "total_ai_tokens"):
        df = dashboard.ai_tokens_by_model_df(config, filters)
        return ResolvedChart(chart, df, "bar", x="ai_model", y="total_tokens")
    if "role" in " ".join(dimensions) or _has_any(metric, "users_by_role"):
        df = dashboard.users_by_role_df(config, filters)
        return ResolvedChart(chart, df, "bar", x="user_role", y="users")
    if _has_any(metric, "feature", "usage"):
        df = dashboard.feature_usage_df(config, filters)
        return ResolvedChart(chart, df, "bar", x="feature_category", y="events")
    raise ValueError(f"No bar-chart mapping for metric {metric!r}.")


def _resolve_funnel(chart, config, filters) -> ResolvedChart:
    raw_steps = chart.get("filters", {}).get("steps")
    steps = _coerce_steps(raw_steps) if raw_steps else _DEFAULT_FUNNEL_STEPS
    ordered = bool(chart.get("filters", {}).get("ordered", True))
    df = dashboard.funnel_df(config, filters, steps=steps, ordered=ordered)
    return ResolvedChart(chart, df, "bar", x="label", y="users")


def _resolve_heatmap(chart, config, filters) -> ResolvedChart:
    max_offset = int(chart.get("filters", {}).get("max_offset", 12))
    df = dashboard.retention_matrix_df(config, filters, max_offset=max_offset)
    return ResolvedChart(
        chart,
        df,
        "heatmap",
        x="week_offset",
        y="cohort_week",
        note="retention",
    )


def _render_resolved_chart(resolved: ResolvedChart, st) -> None:
    df = resolved.dataframe
    if getattr(df, "empty", False):
        st.caption("No data for this chart in the current scope.")
        return
    if resolved.render_type == "line":
        st.line_chart(df, x=resolved.x, y=resolved.y, height=260)
    elif resolved.render_type == "bar":
        st.bar_chart(df, x=resolved.x, y=resolved.y, height=300)
    elif resolved.render_type == "heatmap":
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.dataframe(df, use_container_width=True, hide_index=True)


def _repair_chart_spec(
    chart: dict[str, Any], error: str, config: AgentConfig
) -> dict[str, Any]:
    try:
        llm = build_chat_model(config)
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        "Repair one dashboard chart spec so it maps to known helpers. "
                        "Return only JSON with title,type,metric,dimensions,filters. "
                        "Allowed types: line, bar, funnel, heatmap. Common metrics: "
                        "daily_active_users, total_ai_tokens, feature_usage, retention, funnel."
                    )
                ),
                HumanMessage(
                    content=f"Broken chart:\n{json.dumps(chart)}\n\nError:\n{error}"
                ),
            ]
        )
        parsed = _extract_json_object(str(response.content))
    except Exception:
        return chart
    if not isinstance(parsed, dict):
        return chart
    repaired = dict(chart)
    repaired.update(parsed)
    return repaired


def _extract_json_object(text: str) -> dict | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return None
    try:
        parsed = json.loads(text[start : end + 1])
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _coerce_steps(raw_steps) -> tuple[tuple[str, tuple[str, ...]], ...]:
    steps = []
    for item in raw_steps:
        if isinstance(item, dict):
            label = str(item.get("label", "Step"))
            events = tuple(str(event) for event in item.get("events", []) or [])
        else:
            label, events = item
            events = tuple(str(event) for event in events)
        if events:
            steps.append((label, events))
    return tuple(steps) or _DEFAULT_FUNNEL_STEPS


def _metric_to_daily_column(metric: str) -> str:
    return {
        "daily_active_users": "active_users",
        "active_users": "active_users",
        "total_events": "total_events",
        "ai_generations": "ai_generations",
        "exceptions": "exceptions",
        "pageviews": "pageviews",
        "credits": "credit_events",
    }.get(metric, metric)


def _first_existing(df, columns: list[str]) -> str:
    for column in columns:
        if column in getattr(df, "columns", []):
            return column
    raise ValueError(f"None of these columns are available: {columns}")


def _has_any(text: str, *needles: str) -> bool:
    return any(needle in text for needle in needles)


def _slug(value: str) -> str:
    base = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "dashboard"
    return f"{base}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "MAX_CHART_REPAIRS",
    "ResolvedChart",
    "list_specs",
    "render_spec",
    "resolve_chart_data",
    "save_spec",
]
