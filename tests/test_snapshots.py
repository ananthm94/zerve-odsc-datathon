import tempfile
import unittest

from analytics_agent.core.config import AgentConfig
from analytics_agent.core.contracts import SnapshotRecord
from analytics_agent.ask.snapshots import (
    SnapshotStore,
    compare_to_snapshot,
    promote_snapshot,
)


def _config(store_dir: str) -> AgentConfig:
    return AgentConfig(
        openai_api_key="openai",
        qdrant_url="http://qdrant",
        qdrant_api_key="qdrant",
        qdrant_collection_name="collection",
        agent_store_dir=store_dir,
    )


class SnapshotStoreTests(unittest.TestCase):
    def test_record_latest_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(_config(tmp))
            store.record_snapshot(
                SnapshotRecord(
                    metric="daily_active_users",
                    filters_key="",
                    period="2026-04",
                    value=100,
                    source="seed",
                    captured_at="2026-06-14T10:00:00",
                )
            )
            store.record_snapshot(
                SnapshotRecord(
                    metric="daily_active_users",
                    filters_key="",
                    period="2026-04",
                    value=125,
                    source="approved",
                    captured_at="2026-06-14T11:00:00",
                )
            )

            latest = store.latest_snapshot("daily_active_users", period="2026-04")

        self.assertIsNotNone(latest)
        self.assertEqual(latest.value, 125)
        self.assertEqual(latest.source, "approved")

    def test_compare_to_snapshot_flags_threshold_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(_config(tmp))
            store.record_snapshot(
                SnapshotRecord(
                    metric="total_ai_tokens",
                    filters_key="",
                    period="current",
                    value=100,
                    source="seed",
                    captured_at="2026-06-14T10:00:00",
                )
            )

            result = compare_to_snapshot("total_ai_tokens", 130, store, 0.15)

        self.assertEqual(result["baseline"], 100)
        self.assertAlmostEqual(result["pct_diff"], 0.30)
        self.assertTrue(result["flag"])

    def test_promote_snapshot_writes_approved_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = SnapshotStore(_config(tmp))
            promoted = promote_snapshot(
                "total_users",
                42,
                store,
                captured_at="2026-06-14T12:00:00",
            )
            latest = store.latest_snapshot("total_users")

        self.assertEqual(promoted.source, "approved")
        self.assertEqual(latest.source, "approved")
        self.assertEqual(latest.value, 42)


if __name__ == "__main__":
    unittest.main()
