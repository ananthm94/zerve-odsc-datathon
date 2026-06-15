"""Metric snapshots and drift checks for trusted historical numbers."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from analytics_agent import dashboard
from analytics_agent.core.config import AgentConfig
from analytics_agent.core.contracts import SnapshotRecord


SNAPSHOTS_DB = "snapshots.db"


class SnapshotStore:
    """SQLite-backed metric snapshot store."""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.store_dir = Path(config.agent_store_dir)
        self.db_path = self.store_dir / SNAPSHOTS_DB
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self._ensure_schema()

    def record_snapshot(self, record: SnapshotRecord) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO snapshots (
                    metric, filters_key, period, value, source, captured_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.metric,
                    record.filters_key,
                    record.period,
                    record.value,
                    record.source,
                    record.captured_at,
                ),
            )

    def latest_snapshot(
        self, metric: str, filters_key: str = "", period: str | None = None
    ) -> SnapshotRecord | None:
        predicates = ["metric = ?", "filters_key = ?"]
        params: list[str] = [metric, filters_key]
        if period is not None:
            predicates.append("period = ?")
            params.append(period)
        where = " AND ".join(predicates)
        with self._connect() as con:
            row = con.execute(
                f"""
                SELECT metric, filters_key, period, value, source, captured_at
                FROM snapshots
                WHERE {where}
                ORDER BY captured_at DESC, rowid DESC
                LIMIT 1
                """,
                params,
            ).fetchone()
        if row is None:
            return None
        return SnapshotRecord(
            metric=row[0],
            filters_key=row[1],
            period=row[2],
            value=float(row[3]),
            source=row[4],
            captured_at=row[5],
        )

    @contextmanager
    def _connect(self):
        con = sqlite3.connect(self.db_path)
        try:
            yield con
            con.commit()
        finally:
            con.close()

    def _ensure_schema(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS snapshots (
                    metric TEXT NOT NULL,
                    filters_key TEXT NOT NULL,
                    period TEXT NOT NULL,
                    value REAL NOT NULL,
                    source TEXT NOT NULL,
                    captured_at TEXT NOT NULL
                )
                """
            )
            con.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_snapshots_lookup
                ON snapshots(metric, filters_key, period, captured_at)
                """
            )


def seed_snapshots(config: AgentConfig) -> int:
    """Seed current dashboard KPI snapshots for the default unfiltered scope."""
    store = SnapshotStore(config)
    kpis = dashboard.kpis(config)
    now = _now()
    period = "current"
    count = 0
    for metric, value in kpis.items():
        if isinstance(value, (int, float)):
            store.record_snapshot(
                SnapshotRecord(
                    metric=metric,
                    filters_key="",
                    period=period,
                    value=float(value),
                    source="seed",
                    captured_at=now,
                )
            )
            count += 1
    return count


def compare_to_snapshot(
    metric: str, value: float, store: SnapshotStore, threshold: float
) -> dict:
    """Compare a value to the latest snapshot and flag threshold drift."""
    baseline = store.latest_snapshot(metric)
    if baseline is None:
        return {
            "baseline": None,
            "pct_diff": None,
            "flag": False,
            "source": None,
            "captured_at": None,
        }

    if baseline.value == 0:
        pct_diff = 0.0 if value == 0 else float("inf")
    else:
        pct_diff = (float(value) - baseline.value) / abs(baseline.value)
    return {
        "baseline": baseline.value,
        "pct_diff": pct_diff,
        "flag": abs(pct_diff) > threshold,
        "source": baseline.source,
        "captured_at": baseline.captured_at,
    }


def promote_snapshot(
    metric: str,
    value: float,
    store: SnapshotStore,
    *,
    filters_key: str = "",
    period: str = "current",
    captured_at: str | None = None,
) -> SnapshotRecord:
    """Write an approved snapshot after human approval."""
    record = SnapshotRecord(
        metric=metric,
        filters_key=filters_key,
        period=period,
        value=float(value),
        source="approved",
        captured_at=captured_at or _now(),
    )
    store.record_snapshot(record)
    return record


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


__all__ = [
    "SNAPSHOTS_DB",
    "SnapshotStore",
    "compare_to_snapshot",
    "promote_snapshot",
    "seed_snapshots",
]
