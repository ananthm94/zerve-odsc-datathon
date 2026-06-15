import unittest

from analytics_agent.core.contracts import ReportRecord
from analytics_agent.ui.helpers import (
    dashboard_to_item,
    drift_badge,
    example_prompts,
    filter_items,
    format_item_date,
    group_items,
    library_counts,
    library_item_label,
    report_to_item,
)


class UIHelpersTests(unittest.TestCase):
    def test_group_items_by_theme_month_and_type(self):
        report = ReportRecord(
            id="r1",
            question="Activation report",
            summary="Activation improved.",
            metrics_used=["daily_active_users"],
            approval_status="approved",
            theme="Onboarding/Activation",
            created_at="2026-06-14T12:00:00",
        )
        dashboard = {
            "id": "d1",
            "dashboard_title": "AI Usage",
            "purpose": "Track AI",
            "theme": "AI Generation",
            "created_at": "2026-06-01T12:00:00",
            "charts": [{"metric": "total_ai_tokens"}],
        }
        items = [report_to_item(report), dashboard_to_item(dashboard)]

        self.assertIn("Onboarding/Activation", group_items(items, "theme"))
        self.assertIn("2026-06", group_items(items, "month"))
        self.assertIn("Dashboard", group_items(items, "type"))

    def test_filter_items_searches_combined_fields(self):
        item = dashboard_to_item(
            {
                "id": "d1",
                "dashboard_title": "AI Usage",
                "purpose": "Track model token usage",
                "theme": "AI Generation",
                "charts": [{"metric": "total_ai_tokens"}],
            }
        )

        self.assertEqual(len(filter_items([item], "token")), 1)
        self.assertEqual(len(filter_items([item], "credits")), 0)

    def test_drift_badge(self):
        self.assertEqual(drift_badge(None), "No baseline")
        self.assertEqual(
            drift_badge({"baseline": 100, "pct_diff": 0.2, "flag": True}),
            "up 20.0% (alert)",
        )

    def test_library_counts_splits_reports_and_dashboards(self):
        report = report_to_item(
            ReportRecord(
                id="r1",
                question="Activation report",
                summary="Activation improved.",
                metrics_used=[],
                created_at="2026-06-14T12:00:00",
            )
        )
        dashboard = dashboard_to_item(
            {"id": "d1", "dashboard_title": "AI Usage", "charts": []}
        )

        self.assertEqual(library_counts([report, dashboard]), (1, 1, 2))

    def test_library_item_label_includes_type_and_title(self):
        item = dashboard_to_item({"id": "d1", "dashboard_title": "AI Usage"})

        self.assertEqual(library_item_label(item), "Dashboard - AI Usage")

    def test_format_item_date_returns_short_date_or_fallback(self):
        self.assertEqual(format_item_date("2026-06-14T12:00:00"), "Jun 14, 2026")
        self.assertEqual(format_item_date(""), "No saved date")

    def test_example_prompts_are_non_empty_strings(self):
        prompts = example_prompts()

        self.assertGreaterEqual(len(prompts), 3)
        self.assertTrue(all(isinstance(prompt, str) and prompt for prompt in prompts))


if __name__ == "__main__":
    unittest.main()
