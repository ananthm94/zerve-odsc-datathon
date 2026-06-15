"""Query backend abstraction.

The agent runs against DuckDB by default (local, free, reproducible) and can
optionally target BigQuery. Both backends expose the same ``QueryRunner``
interface so the rest of the agent is backend-agnostic.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from analytics_agent.core.bigquery_client import BigQueryRunner
from analytics_agent.core.config import AgentConfig, ConfigError


@runtime_checkable
class QueryRunner(Protocol):
    def dry_run(self, sql: str) -> int:
        """Validate/estimate a query without returning rows. Returns bytes
        processed where the backend reports it (BigQuery), else 0 (DuckDB)."""

    def execute(self, sql: str, max_results: int = 100) -> list[dict[str, Any]]:
        """Run a read-only query and return up to ``max_results`` rows as dicts."""


class DuckDBRunner:
    """Runs queries against the local DuckDB database built by dbt.

    Connections are opened read-only per query, so the agent can never mutate
    the warehouse even if validation were bypassed.
    """

    def __init__(self, config: AgentConfig):
        self.config = config

    def _connect(self):
        import duckdb

        return duckdb.connect(self.config.duckdb_path, read_only=True)

    def dry_run(self, sql: str) -> int:
        # DuckDB queries are free; EXPLAIN confirms the plan compiles. No byte cost.
        con = self._connect()
        try:
            con.execute(f"EXPLAIN {sql}")
        finally:
            con.close()
        return 0

    def execute(self, sql: str, max_results: int = 100) -> list[dict[str, Any]]:
        con = self._connect()
        try:
            cursor = con.execute(sql)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchmany(max_results)
        finally:
            con.close()
        return [dict(zip(columns, row)) for row in rows]


def build_runner(config: AgentConfig) -> QueryRunner:
    if config.backend == "duckdb":
        return DuckDBRunner(config)
    if config.backend == "bigquery":
        return BigQueryRunner(config)
    raise ConfigError(f"Unsupported backend: {config.backend!r}")


__all__ = ["QueryRunner", "DuckDBRunner", "BigQueryRunner", "build_runner"]
