import os
import unittest
from unittest import mock

from analytics_agent.core.config import (
    DEFAULT_AGENT_STORE_DIR,
    DEFAULT_SEMANTIC_MANIFEST_PATH,
    DEFAULT_SNAPSHOT_DRIFT_THRESHOLD,
    AgentConfig,
    load_config,
)
from analytics_agent.core.contracts import (
    MetricDefinition,
    ReportRecord,
    SnapshotRecord,
    assign_theme,
)


class ConfigContractTests(unittest.TestCase):
    def test_agent_config_defaults_resolve(self):
        cfg = AgentConfig(
            openai_api_key="openai",
            qdrant_url="http://qdrant",
            qdrant_api_key="qdrant",
            qdrant_collection_name="collection",
        )

        self.assertEqual(cfg.semantic_manifest_path, DEFAULT_SEMANTIC_MANIFEST_PATH)
        self.assertEqual(cfg.agent_store_dir, DEFAULT_AGENT_STORE_DIR)
        self.assertEqual(
            cfg.snapshot_drift_threshold, DEFAULT_SNAPSHOT_DRIFT_THRESHOLD
        )

    def test_load_config_reads_new_env_overrides(self):
        env = {
            "OPENAI_API_KEY": "openai",
            "QDRANT_URL": "http://qdrant",
            "QDRANT_API_KEY": "qdrant",
            "QDRANT_COLLECTION_NAME": "collection",
            "DBT_SEMANTIC_MANIFEST_PATH": "custom/semantic_manifest.json",
            "AGENT_STORE_DIR": "custom_store",
            "SNAPSHOT_DRIFT_THRESHOLD": "0.25",
        }
        with mock.patch.dict(os.environ, env, clear=True), mock.patch(
            "analytics_agent.core.config.load_dotenv", return_value=None
        ):
            cfg = load_config()

        self.assertEqual(cfg.semantic_manifest_path, "custom/semantic_manifest.json")
        self.assertEqual(cfg.agent_store_dir, "custom_store")
        self.assertEqual(cfg.snapshot_drift_threshold, 0.25)


class DataclassContractTests(unittest.TestCase):
    def test_metric_definition_instantiates(self):
        definition = MetricDefinition(
            metric="daily_active_users",
            definition="Distinct users active on a day.",
            grain="day",
            time_column="event_date",
            allowed_dimensions=["event_name", "event_country"],
            source_table="stg_events",
            join_keys=["user_id"],
            measure="active_users",
            metric_type="simple",
        )

        self.assertEqual(definition.metric, "daily_active_users")
        self.assertEqual(definition.source_table, "stg_events")

    def test_report_record_instantiates(self):
        record = ReportRecord(
            id="report-1",
            question="What changed?",
            summary="Activity increased.",
            metrics_used=["daily_active_users"],
            sql_used=["select 1"],
            date_range="2026-01-01 to 2026-01-31",
            source_tables=["agg_daily_activity"],
            approval_status="approved",
            tags=["activation"],
            theme="Onboarding/Activation",
            created_at="2026-06-14T00:00:00",
        )

        self.assertEqual(record.approval_status, "approved")
        self.assertEqual(record.metrics_used, ["daily_active_users"])

    def test_snapshot_record_instantiates(self):
        snapshot = SnapshotRecord(
            metric="total_ai_tokens",
            filters_key="",
            period="2026-04",
            value=123.45,
            source="approved",
            captured_at="2026-06-14T00:00:00",
        )

        self.assertEqual(snapshot.metric, "total_ai_tokens")
        self.assertEqual(snapshot.value, 123.45)

    def test_assign_theme_uses_dominant_metric(self):
        self.assertEqual(assign_theme(["total_ai_tokens"]), "AI Generation")
        self.assertEqual(assign_theme(["total_credits_used"]), "Credits")
        self.assertEqual(assign_theme([]), "Other")


if __name__ == "__main__":
    unittest.main()
