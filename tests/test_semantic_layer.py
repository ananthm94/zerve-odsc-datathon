import json
import tempfile
import unittest
from pathlib import Path

from analytics_agent.core.config import AgentConfig
from analytics_agent.ask.semantic_layer import (
    format_for_prompt,
    load_semantic_layer,
    lookup_metrics,
)


def _semantic_manifest() -> dict:
    return {
        "semantic_models": [
            {
                "name": "events",
                "defaults": {"agg_time_dimension": "event_date"},
                "node_relation": {"alias": "stg_events"},
                "entities": [{"name": "user", "type": "foreign", "expr": "user_id"}],
                "dimensions": [
                    {"name": "event_date", "type": "time"},
                    {"name": "event_name", "type": "categorical"},
                    {"name": "event_country", "type": "categorical", "expr": "country"},
                ],
                "measures": [
                    {
                        "name": "active_users",
                        "agg": "count_distinct",
                        "description": "Distinct active users.",
                        "expr": "user_id",
                    }
                ],
            },
            {
                "name": "user_summary",
                "defaults": {"agg_time_dimension": "first_event_date"},
                "node_relation": {"alias": "user_summary"},
                "entities": [{"name": "user", "type": "primary", "expr": "user_id"}],
                "dimensions": [
                    {"name": "first_event_date", "type": "time"},
                    {"name": "activity_segment", "type": "categorical"},
                ],
                "measures": [
                    {
                        "name": "ai_user_count_measure",
                        "agg": "sum",
                        "description": "Count of AI users.",
                        "expr": "case when is_ai_user then 1 else 0 end",
                    },
                    {
                        "name": "total_users_measure",
                        "agg": "count_distinct",
                        "description": "Count of unique users.",
                        "expr": "user_id",
                    },
                ],
            },
        ],
        "metrics": [
            {
                "name": "daily_active_users",
                "description": "Distinct active users (group by day for DAU).",
                "type": "simple",
                "type_params": {
                    "measure": {"name": "active_users"},
                    "input_measures": [{"name": "active_users"}],
                },
            },
            {
                "name": "ai_adoption_rate",
                "description": "Share of users who have made at least one AI generation.",
                "type": "ratio",
                "type_params": {
                    "numerator": {"name": "ai_users"},
                    "denominator": {"name": "total_users"},
                    "input_measures": [
                        {"name": "ai_user_count_measure"},
                        {"name": "total_users_measure"},
                    ],
                },
            },
        ],
    }


def _config(path: str) -> AgentConfig:
    return AgentConfig(
        openai_api_key="openai",
        qdrant_url="http://qdrant",
        qdrant_api_key="qdrant",
        qdrant_collection_name="collection",
        semantic_manifest_path=path,
    )


class SemanticLayerTests(unittest.TestCase):
    def test_parse_daily_active_users_metric_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic_manifest.json"
            path.write_text(json.dumps(_semantic_manifest()))

            layer = load_semantic_layer(_config(str(path)))

        metric = layer["daily_active_users"]
        self.assertEqual(metric.metric, "daily_active_users")
        self.assertEqual(metric.metric_type, "simple")
        self.assertEqual(metric.grain, "day")
        self.assertEqual(metric.time_column, "event_date")
        self.assertEqual(metric.source_table, "stg_events")
        self.assertEqual(metric.measure, "active_users")
        self.assertIn("event_name", metric.allowed_dimensions)
        self.assertEqual(metric.join_keys, ["user_id"])

    def test_parse_ratio_metric_definition(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic_manifest.json"
            path.write_text(json.dumps(_semantic_manifest()))

            layer = load_semantic_layer(_config(str(path)))

        metric = layer["ai_adoption_rate"]
        self.assertEqual(metric.metric_type, "ratio")
        self.assertEqual(metric.source_table, "user_summary")
        self.assertEqual(
            metric.measure, "ai_user_count_measure, total_users_measure"
        )
        self.assertIn("activity_segment", metric.allowed_dimensions)

    def test_lookup_term_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic_manifest.json"
            path.write_text(json.dumps(_semantic_manifest()))
            layer = load_semantic_layer(_config(str(path)))

        matches = lookup_metrics("What is DAU by country?", layer, llm=None)

        self.assertGreaterEqual(len(matches), 1)
        self.assertEqual(matches[0].metric, "daily_active_users")

    def test_format_for_prompt_includes_query_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "semantic_manifest.json"
            path.write_text(json.dumps(_semantic_manifest()))
            layer = load_semantic_layer(_config(str(path)))

        formatted = format_for_prompt([layer["daily_active_users"]])

        self.assertIn("Metric: daily_active_users", formatted)
        self.assertIn("Source table: stg_events", formatted)
        self.assertIn("Allowed dimensions: event_date, event_name", formatted)


if __name__ == "__main__":
    unittest.main()
