"""Testable non-Streamlit helpers for Library and approval UI."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from analytics_agent.core.contracts import ReportRecord, assign_theme


@dataclass(frozen=True)
class LibraryItem:
    id: str
    item_type: str
    title: str
    summary: str
    theme: str
    status: str
    created_at: str
    tags: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    source: Any = None


def report_to_item(report: ReportRecord) -> LibraryItem:
    return LibraryItem(
        id=report.id,
        item_type="report",
        title=report.question,
        summary=report.summary,
        theme=report.theme or assign_theme(report.metrics_used),
        status=report.approval_status,
        created_at=report.created_at,
        tags=report.tags,
        metrics=report.metrics_used,
        source=report,
    )


def dashboard_to_item(spec: dict) -> LibraryItem:
    charts = spec.get("charts", []) or []
    metrics = [str(chart.get("metric", "")) for chart in charts if chart.get("metric")]
    return LibraryItem(
        id=str(spec.get("id") or spec.get("_path") or spec.get("dashboard_title")),
        item_type="dashboard",
        title=str(spec.get("dashboard_title") or "Untitled dashboard"),
        summary=str(spec.get("purpose") or ""),
        theme=str(spec.get("theme") or assign_theme(metrics)),
        status=str(spec.get("approval_status") or spec.get("status") or "approved"),
        created_at=str(spec.get("created_at") or ""),
        tags=list(spec.get("tags") or []),
        metrics=metrics,
        source=spec,
    )


def filter_items(items: list[LibraryItem], query: str = "") -> list[LibraryItem]:
    terms = [term.lower() for term in query.split() if term.strip()]
    if not terms:
        return sorted_items(items)
    filtered = []
    for item in items:
        haystack = " ".join(
            [
                item.title,
                item.summary,
                item.theme,
                item.item_type,
                item.status,
                " ".join(item.tags),
                " ".join(item.metrics),
            ]
        ).lower()
        if all(term in haystack for term in terms):
            filtered.append(item)
    return sorted_items(filtered)


def group_items(items: list[LibraryItem], group_by: str = "theme") -> dict[str, list[LibraryItem]]:
    groups: dict[str, list[LibraryItem]] = {}
    for item in sorted_items(items):
        if group_by == "month":
            key = _month_key(item.created_at)
        elif group_by == "type":
            key = item.item_type.title()
        else:
            key = item.theme or "Other"
        groups.setdefault(key, []).append(item)
    return groups


def sorted_items(items: list[LibraryItem]) -> list[LibraryItem]:
    return sorted(items, key=lambda item: (item.created_at, item.title), reverse=True)


def example_prompts() -> tuple[str, ...]:
    return (
        "What changed in weekly active users over the dataset?",
        "Which user segments adopt AI features most?",
        "Where are exceptions concentrated by device or OS?",
        "Create a dashboard for AI usage and activation.",
    )


def library_counts(items: list[LibraryItem]) -> tuple[int, int, int]:
    reports = sum(1 for item in items if item.item_type == "report")
    dashboards = sum(1 for item in items if item.item_type == "dashboard")
    return reports, dashboards, len(items)


def library_item_label(item: LibraryItem) -> str:
    return f"{item.item_type.title()} - {item.title}"


def format_item_date(created_at: str) -> str:
    if not created_at:
        return "No saved date"
    try:
        parsed = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
    except ValueError:
        return created_at[:10] if len(created_at) >= 10 else "No saved date"
    return parsed.strftime("%b %d, %Y").replace(" 0", " ")


def drift_badge(snapshot_check: dict | None) -> str:
    if not snapshot_check or snapshot_check.get("baseline") is None:
        return "No baseline"
    pct = snapshot_check.get("pct_diff")
    if pct is None:
        return "Baseline unavailable"
    arrow = "up" if pct > 0 else "down" if pct < 0 else "flat"
    marker = "alert" if snapshot_check.get("flag") else "ok"
    return f"{arrow} {pct * 100:.1f}% ({marker})"


def auto_theme(metrics: list[str]) -> str:
    return assign_theme(metrics)


def _month_key(created_at: str) -> str:
    if not created_at:
        return "Unknown month"
    try:
        return datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime(
            "%Y-%m"
        )
    except ValueError:
        return created_at[:7] if len(created_at) >= 7 else "Unknown month"


__all__ = [
    "LibraryItem",
    "auto_theme",
    "dashboard_to_item",
    "drift_badge",
    "example_prompts",
    "filter_items",
    "format_item_date",
    "group_items",
    "library_counts",
    "library_item_label",
    "report_to_item",
    "sorted_items",
]
