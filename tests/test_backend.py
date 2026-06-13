import unittest
from dataclasses import replace

from analytics_agent.config import AgentConfig, ConfigError
from analytics_agent.runners import BigQueryRunner, DuckDBRunner, build_runner
from analytics_agent.sql import validate_sql


def _config(**overrides) -> AgentConfig:
    base = AgentConfig(
        openai_api_key="x",
        qdrant_url="x",
        qdrant_api_key="x",
        qdrant_collection_name="c",
        bigquery_project="my-zerve-project",
        bigquery_dataset="zerveevents",
    )
    return replace(base, **overrides)


class BackendIdentifierTests(unittest.TestCase):
    def test_duckdb_identifiers(self):
        cfg = _config(backend="duckdb")
        self.assertEqual(
            cfg.fully_qualified_tables["user_summary"], "main.user_summary"
        )

    def test_bigquery_identifiers(self):
        cfg = _config(backend="bigquery")
        self.assertEqual(
            cfg.fully_qualified_tables["user_summary"],
            "`my-zerve-project.zerveevents.user_summary`",
        )

    def test_validate_accepts_duckdb_query(self):
        cfg = _config(backend="duckdb")
        result = validate_sql(
            "select count(*) as n from main.user_summary",
            allowed_table_names=cfg.allowed_table_names,
            fully_qualified_tables=set(cfg.fully_qualified_tables.values()),
        )
        self.assertEqual(result.referenced_tables, {"user_summary"})


class BuildRunnerTests(unittest.TestCase):
    def test_duckdb_runner(self):
        self.assertIsInstance(build_runner(_config(backend="duckdb")), DuckDBRunner)

    def test_bigquery_runner(self):
        # BigQueryRunner constructs a bigquery.Client; selection should still
        # return the right type. Construction may require creds, so guard import.
        runner = build_runner(_config(backend="bigquery"))
        self.assertIsInstance(runner, BigQueryRunner)

    def test_unknown_backend_raises(self):
        # build_runner guards even if an AgentConfig is hand-built with a bad value.
        with self.assertRaises(ConfigError):
            build_runner(_config(backend="snowflake"))


if __name__ == "__main__":
    unittest.main()
