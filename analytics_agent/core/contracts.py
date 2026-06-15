"""Shared contracts for the agentic analytics feature units.

These lightweight dataclasses and aliases are the stable boundary between the
semantic layer, report memory, snapshots, dashboard generation, and the future
orchestrator graph. Keeping them here lets each feature unit evolve without
copying shape definitions across modules.

Future orchestrator ``AgentState`` extensions from the build plan:

- ``intent``: routed user intent such as metric definition, dashboard request,
  one-time analysis, root-cause analysis, SQL generation, or experiment.
- ``metric_definitions``: structured metric definitions selected from the dbt
  semantic layer.
- ``recalled_reports``: prior approved reports relevant to the current question.
- ``snapshot_check``: comparison between current numbers and trusted snapshots.
- ``dashboard_worthy``: whether the answer should produce a dashboard spec.
- ``dashboard_spec``: generated dashboard configuration matching
  ``DashboardSpec``.
- ``final_response``: final user-facing response after routing, memory,
  verification, and optional dashboard generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class MetricDefinition:
    metric: str
    definition: str
    grain: str
    time_column: str
    allowed_dimensions: list[str] = field(default_factory=list)
    source_table: str = ""
    join_keys: list[str] = field(default_factory=list)
    measure: str = ""
    metric_type: str = ""


@dataclass(frozen=True)
class ReportRecord:
    id: str
    question: str
    summary: str
    metrics_used: list[str] = field(default_factory=list)
    sql_used: list[str] = field(default_factory=list)
    date_range: str = ""
    source_tables: list[str] = field(default_factory=list)
    approval_status: str = "pending"
    tags: list[str] = field(default_factory=list)
    theme: str = "Other"
    created_at: str = ""


@dataclass(frozen=True)
class SnapshotRecord:
    metric: str
    filters_key: str
    period: str
    value: float
    source: str
    captured_at: str


# DashboardSpec is intentionally a plain dict contract because Streamlit and
# JSON persistence both work naturally with dictionaries. Expected shape:
#
# {
#   "dashboard_title": str,
#   "purpose": str,
#   "theme": str,
#   "charts": [
#       {
#           "title": str,
#           "type": str,
#           "metric": str,
#           "dimensions": list[str],
#           "filters": dict[str, Any],
#       },
#   ],
#   "refresh_frequency": str,
#   "recommended_alerts": list[str],
# }
DashboardSpec = dict[str, Any]


_THEME_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "AI Generation",
        ("ai", "token", "latency", "generation", "model", "provider"),
    ),
    ("Agent", ("agent", "tool")),
    ("Credits", ("credit", "billing")),
    ("Errors", ("exception", "error", "failure")),
    (
        "Onboarding/Activation",
        ("activation", "onboarding", "sign_in", "signup", "retention", "funnel"),
    ),
    ("Web/Pageviews", ("pageview", "web", "browser", "device", "os", "geo")),
    ("Deployments", ("deploy", "deployment")),
    ("Block Execution", ("block", "execution", "run_block")),
    ("System/Identity", ("user", "identity", "auth")),
)


def assign_theme(metrics: list[str]) -> str:
    """Assign a stable analysis theme from the dominant metric name.

    The first metric is treated as dominant. Names are matched by keyword against
    product feature categories already used in ``stg_feature_events`` so reports
    and generated dashboards can be grouped consistently in the future Library.
    """
    dominant = next((metric for metric in metrics if metric), "")
    normalized = dominant.lower().replace("-", "_").replace(" ", "_")
    if not normalized:
        return "Other"

    for theme, keywords in _THEME_KEYWORDS:
        if any(keyword in normalized for keyword in keywords):
            return theme
    return "Other"


__all__ = [
    "DashboardSpec",
    "MetricDefinition",
    "ReportRecord",
    "SnapshotRecord",
    "assign_theme",
]
