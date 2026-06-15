"""Tests for the dashboard data layer (filters + query construction).

The query runner is faked so these run offline. They assert that the filter
compiler emits the right WHERE predicates (date range + user-cohort subquery)
and that ``daily_activity_df`` switches to the event-grain recompute path when a
user filter is active. Table identifiers come from the real dbt manifest.
"""

import unittest
from dataclasses import replace
from datetime import date
from unittest import mock

from analytics_agent import dashboard
from analytics_agent.core.config import AgentConfig


def _config(**overrides) -> AgentConfig:
    base = AgentConfig(
        openai_api_key="x",
        qdrant_url="x",
        qdrant_api_key="x",
        qdrant_collection_name="c",
        backend="duckdb",
    )
    return replace(base, **overrides)


class FakeRunner:
    """Records executed SQL and returns canned rows."""

    def __init__(self, rows=None):
        self.executed: list[str] = []
        self.rows = rows if rows is not None else [{}]

    def dry_run(self, sql):
        return 0

    def execute(self, sql, max_results=100):
        self.executed.append(sql)
        return self.rows


_KPI_ROW = [
    {
        "total_users": 10,
        "total_events": 100,
        "ai_generations": 20,
        "total_ai_tokens": 5000,
        "total_credits_used": 12.5,
        "total_exceptions": 7,
        "ai_users": 3,
    }
]


def _patch(runner):
    # The query functions live in the ``queries`` submodule and look up
    # ``build_runner`` in that module's globals, so patch it there.
    return mock.patch.object(dashboard.queries, "build_runner", return_value=runner)


class FilterCompilerTests(unittest.TestCase):
    def test_no_filters_emit_no_where(self):
        runner = FakeRunner(_KPI_ROW)
        with _patch(runner):
            dashboard.kpis(_config(), dashboard.DashboardFilters())
        sql = runner.executed[-1]
        # No filter predicates (the FILTER (WHERE ...) aggregates are expected).
        self.assertNotIn("user_id IN", sql)
        self.assertNotIn("event_date >=", sql)

    def test_date_filter_emits_range_predicates(self):
        runner = FakeRunner(_KPI_ROW)
        filters = dashboard.DashboardFilters(
            date_start=date(2026, 1, 1), date_end=date(2026, 1, 31)
        )
        with _patch(runner):
            dashboard.kpis(_config(), filters)
        sql = runner.executed[-1]
        self.assertIn("event_date >= '2026-01-01'", sql)
        self.assertIn("event_date <= '2026-01-31'", sql)

    def test_user_filter_emits_cohort_subquery(self):
        runner = FakeRunner(_KPI_ROW)
        filters = dashboard.DashboardFilters(
            user_roles=("Data Scientist",), activity_segments=("high_activity",)
        )
        with _patch(runner):
            dashboard.kpis(_config(), filters)
        sql = runner.executed[-1]
        self.assertIn("user_id IN (SELECT u.user_id FROM", sql)
        self.assertIn("u.user_role IN ('Data Scientist')", sql)
        self.assertIn("s.activity_segment IN ('high_activity')", sql)

    def test_literal_escaping_blocks_quote_injection(self):
        runner = FakeRunner(_KPI_ROW)
        filters = dashboard.DashboardFilters(user_roles=("O'Brien",))
        with _patch(runner):
            dashboard.kpis(_config(), filters)
        # The single quote is doubled, not left to break out of the literal.
        self.assertIn("'O''Brien'", runner.executed[-1])


class DailyActivityGrainTests(unittest.TestCase):
    def test_no_user_filter_uses_preaggregate(self):
        runner = FakeRunner([])
        with _patch(runner):
            dashboard.daily_activity_df(_config(), dashboard.DashboardFilters())
        sql = runner.executed[-1]
        self.assertIn("agg_daily_activity", sql)
        self.assertNotIn("GROUP BY", sql)

    def test_user_filter_recomputes_from_events(self):
        runner = FakeRunner([])
        filters = dashboard.DashboardFilters(device_types=("Desktop",))
        with _patch(runner):
            dashboard.daily_activity_df(_config(), filters)
        sql = runner.executed[-1]
        self.assertIn("stg_events", sql)
        self.assertIn("GROUP BY", sql)
        self.assertIn("user_id IN (SELECT u.user_id FROM", sql)


class FeatureUsageTests(unittest.TestCase):
    def test_feature_usage_returns_rows_as_dataframe(self):
        rows = [
            {"feature_category": "Agent", "events": 9, "reach": 4, "adoption_rate": 0.4}
        ]
        runner = FakeRunner(rows)
        with _patch(runner):
            df = dashboard.feature_usage_df(_config(), dashboard.DashboardFilters())
        self.assertEqual(list(df["feature_category"]), ["Agent"])
        self.assertIn("stg_feature_events", runner.executed[-1])

    def test_feature_trend_filters_to_chosen_categories(self):
        runner = FakeRunner([])
        with _patch(runner):
            dashboard.feature_trend_df(
                _config(),
                dashboard.DashboardFilters(),
                categories=("AI Generation", "Agent"),
            )
        sql = runner.executed[-1]
        self.assertIn("feature_category IN ('AI Generation', 'Agent')", sql)


class RetentionTests(unittest.TestCase):
    def test_matrix_caps_offset_and_filters_cohort_by_date(self):
        runner = FakeRunner([])
        filters = dashboard.DashboardFilters(date_start=date(2026, 1, 1))
        with _patch(runner):
            dashboard.retention_matrix_df(_config(), filters, max_offset=6)
        sql = runner.executed[-1]
        self.assertIn("fct_user_activity_weekly", sql)
        self.assertIn("week_offset <= 6", sql)
        # Date range applies to cohort_week, not activity_week.
        self.assertIn("cohort_week >= '2026-01-01'", sql)

    def test_non_integer_offset_is_rejected(self):
        # int() coercion rejects an injection attempt via the offset argument.
        runner = FakeRunner([])
        with _patch(runner), self.assertRaises(ValueError):
            dashboard.retention_curve_df(
                _config(), dashboard.DashboardFilters(), max_offset="8; DROP TABLE x"
            )


class FunnelTests(unittest.TestCase):
    _STEPS = (
        ("A", ("sign_in",)),
        ("B", ("run_block", "run_all_blocks")),
        ("C", ("$ai_generation",)),
    )

    def test_empty_steps_returns_empty_frame(self):
        with _patch(FakeRunner([])):
            df = dashboard.funnel_df(_config(), dashboard.DashboardFilters(), ())
        self.assertTrue(df.empty)

    def test_ordered_funnel_emits_timestamp_ordering(self):
        runner = FakeRunner([{"step1": 100, "step2": 40, "step3": 10}])
        with _patch(runner):
            df = dashboard.funnel_df(
                _config(), dashboard.DashboardFilters(), self._STEPS, ordered=True
            )
        sql = runner.executed[-1]
        self.assertIn("t2 >= t1", sql)
        self.assertIn("t3 >= t2", sql)
        # Step B matches either of its two events.
        self.assertIn("event_name IN ('run_block', 'run_all_blocks')", sql)
        self.assertEqual(list(df["users"]), [100, 40, 10])
        self.assertEqual(list(df["label"]), ["A", "B", "C"])

    def test_unordered_funnel_drops_timestamp_ordering(self):
        runner = FakeRunner([{"step1": 100, "step2": 80, "step3": 50}])
        with _patch(runner):
            dashboard.funnel_df(
                _config(), dashboard.DashboardFilters(), self._STEPS, ordered=False
            )
        sql = runner.executed[-1]
        self.assertNotIn(">=", sql)
        self.assertIn("t3 IS NOT NULL", sql)


_AHA_KPI_ROW = [
    {
        "signups": 100,
        "converters": 5,
        "churned_w4": 60,
        "observable_activated": 80,
        "median_days_to_convert": 0,
    }
]


class AhaDataLayerTests(unittest.TestCase):
    def test_kpis_query_outcomes_with_no_filter(self):
        runner = FakeRunner(_AHA_KPI_ROW)
        with _patch(runner):
            out = dashboard.aha_kpis(_config(), dashboard.DashboardFilters())
        sql = runner.executed[-1]
        self.assertIn("dim_user_outcomes", sql)
        self.assertNotIn("user_id IN", sql)
        self.assertNotIn("signup_date >=", sql)
        self.assertAlmostEqual(out["conversion_rate"], 0.05)
        self.assertAlmostEqual(out["w4_churn_rate"], 60 / 80)

    def test_kpis_apply_signup_date_and_user_cohort(self):
        runner = FakeRunner(_AHA_KPI_ROW)
        filters = dashboard.DashboardFilters(
            date_start=date(2026, 1, 1),
            date_end=date(2026, 1, 31),
            user_roles=("Data Scientist",),
        )
        with _patch(runner):
            dashboard.aha_kpis(_config(), filters)
        sql = runner.executed[-1]
        self.assertIn("signup_date >= '2026-01-01'", sql)
        self.assertIn("signup_date <= '2026-01-31'", sql)
        self.assertIn("user_id IN (SELECT u.user_id FROM", sql)

    def test_funnel_joins_outcomes_and_milestones(self):
        row = {f"s{i}": v for i, v in enumerate([100, 40, 38, 20, 8, 5])}
        runner = FakeRunner([row])
        with _patch(runner):
            df = dashboard.aha_funnel_df(_config(), dashboard.DashboardFilters())
        sql = runner.executed[-1]
        self.assertIn("dim_user_outcomes o", sql)
        self.assertIn("fct_activation_milestones m", sql)
        self.assertEqual(list(df["users"]), [100, 40, 38, 20, 8, 5])
        self.assertAlmostEqual(df["pct_of_signups"].iloc[3], 0.20)

    def test_funnel_user_filter_targets_outcome_alias(self):
        row = {f"s{i}": 0 for i in range(6)}
        runner = FakeRunner([row])
        filters = dashboard.DashboardFilters(device_types=("Desktop",))
        with _patch(runner):
            dashboard.aha_funnel_df(_config(), filters)
        sql = runner.executed[-1]
        self.assertIn("o.user_id IN (SELECT u.user_id FROM", sql)

    def test_lift_filters_by_signal_type(self):
        runner = FakeRunner([])
        with _patch(runner):
            dashboard.aha_lift_df(_config(), "product")
        sql = runner.executed[-1]
        self.assertIn("agg_aha_lift", sql)
        self.assertIn("signal_type = 'product'", sql)
        self.assertIn("ORDER BY conversion_lift DESC", sql)

    def test_lift_no_signal_type_returns_all(self):
        runner = FakeRunner([])
        with _patch(runner):
            dashboard.aha_lift_df(_config(), None)
        self.assertNotIn("signal_type =", runner.executed[-1])

    def test_lift_rejects_unknown_signal_type(self):
        with self.assertRaises(ValueError):
            dashboard.aha_lift_df(_config(), "bogus")

    def test_depth_buckets_week1_active_days(self):
        runner = FakeRunner([])
        with _patch(runner):
            dashboard.aha_depth_df(_config(), dashboard.DashboardFilters())
        sql = runner.executed[-1]
        self.assertIn("w1_active_days", sql)
        self.assertIn("depth_bucket", sql)

    def test_churn_watch_scopes_to_activated_non_payers(self):
        runner = FakeRunner([])
        with _patch(runner):
            dashboard.churn_watch_df(_config(), dashboard.DashboardFilters())
        sql = runner.executed[-1]
        self.assertIn("dim_user_outcomes", sql)
        self.assertIn("activated", sql)
        self.assertIn("NOT is_converter", sql)
        self.assertIn("recency_days", sql)


if __name__ == "__main__":
    unittest.main()
