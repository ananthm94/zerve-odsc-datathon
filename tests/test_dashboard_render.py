import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from analytics_agent.core.config import AgentConfig
from analytics_agent.dashboard import render as dashboard_render
from analytics_agent.dashboard import DashboardFilters


def _config(store_dir: str) -> AgentConfig:
    return AgentConfig(
        openai_api_key="openai",
        qdrant_url="http://qdrant",
        qdrant_api_key="qdrant",
        qdrant_collection_name="collection",
        agent_store_dir=store_dir,
    )


class DashboardRenderMappingTests(unittest.TestCase):
    def test_line_daily_active_users_maps_to_daily_activity(self):
        chart = {
            "title": "DAU",
            "type": "line",
            "metric": "daily_active_users",
            "dimensions": ["event_date"],
            "filters": {},
        }
        df = pd.DataFrame(
            [{"event_date": "2026-01-01", "active_users": 10, "total_events": 20}]
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            dashboard_render.dashboard, "daily_activity_df", return_value=df
        ) as helper:
            resolved = dashboard_render.resolve_chart_data(
                chart, _config(tmp), DashboardFilters()
            )

        helper.assert_called_once()
        self.assertEqual(resolved.render_type, "line")
        self.assertEqual(resolved.x, "event_date")
        self.assertEqual(resolved.y, "active_users")

    def test_bar_tokens_maps_to_ai_tokens_by_model(self):
        chart = {
            "title": "Tokens",
            "type": "bar",
            "metric": "total_ai_tokens",
            "dimensions": ["ai_model"],
            "filters": {},
        }
        df = pd.DataFrame([{"ai_model": "gpt", "total_tokens": 100}])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            dashboard_render.dashboard, "ai_tokens_by_model_df", return_value=df
        ) as helper:
            resolved = dashboard_render.resolve_chart_data(
                chart, _config(tmp), DashboardFilters()
            )

        helper.assert_called_once()
        self.assertEqual(resolved.render_type, "bar")
        self.assertEqual(resolved.x, "ai_model")
        self.assertEqual(resolved.y, "total_tokens")

    def test_funnel_maps_to_funnel_helper(self):
        chart = {"title": "Activation", "type": "funnel", "metric": "activation"}
        df = pd.DataFrame([{"label": "Sign in", "users": 10}])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            dashboard_render.dashboard, "funnel_df", return_value=df
        ) as helper:
            resolved = dashboard_render.resolve_chart_data(
                chart, _config(tmp), DashboardFilters()
            )

        helper.assert_called_once()
        self.assertEqual(resolved.render_type, "bar")
        self.assertEqual(resolved.x, "label")
        self.assertEqual(resolved.y, "users")

    def test_heatmap_maps_to_retention_matrix(self):
        chart = {"title": "Retention", "type": "heatmap", "metric": "retention"}
        df = pd.DataFrame(
            [{"cohort_week": "2026-01-05", "week_offset": 1, "retention": 0.5}]
        )
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            dashboard_render.dashboard, "retention_matrix_df", return_value=df
        ) as helper:
            resolved = dashboard_render.resolve_chart_data(
                chart, _config(tmp), DashboardFilters()
            )

        helper.assert_called_once()
        self.assertEqual(resolved.render_type, "heatmap")

    def test_bad_metric_repair_can_map_to_placeholder_fallback(self):
        chart = {"title": "Broken", "type": "line", "metric": "not_real"}
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                dashboard_render.resolve_chart_data(chart, _config(tmp))


class DashboardSpecPersistenceTests(unittest.TestCase):
    def test_save_spec_and_list_specs_roundtrip(self):
        spec = {
            "dashboard_title": "AI Usage Monitor",
            "purpose": "Track AI usage",
            "theme": "AI Generation",
            "charts": [
                {
                    "title": "DAU",
                    "type": "line",
                    "metric": "daily_active_users",
                    "dimensions": ["event_date"],
                    "filters": {},
                }
            ],
            "refresh_frequency": "manual",
            "recommended_alerts": [],
        }
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _config(tmp)
            path = dashboard_render.save_spec(spec, cfg)
            saved = json.loads(Path(path).read_text())
            listed = dashboard_render.list_specs(cfg)

        self.assertEqual(saved["dashboard_title"], "AI Usage Monitor")
        self.assertEqual(len(listed), 1)
        self.assertEqual(listed[0]["dashboard_title"], "AI Usage Monitor")


if __name__ == "__main__":
    unittest.main()
