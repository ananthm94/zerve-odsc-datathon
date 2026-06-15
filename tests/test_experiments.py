"""Tests for experiment / cohort analysis (v0.6).

Stats functions are checked against hand-computed reference values; cohort SQL
construction is checked offline through a fake runner (so it still exercises the
real validate_sql guardrails); and a live DuckDB comparison runs when the built
warehouse is present.
"""

import math
import os
import unittest
from dataclasses import replace
from unittest import mock

from analytics_agent import experiments as exp
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


class ContinuousStatsTests(unittest.TestCase):
    def test_means_and_cohens_d(self):
        a = [2, 4, 6, 8]
        b = [1, 3, 5, 7]
        result = exp.compare_continuous("m", "d", "A", "B", a, b)
        self.assertEqual(result.mean_a, 5.0)
        self.assertEqual(result.mean_b, 4.0)
        self.assertEqual(result.mean_diff, 1.0)
        # pooled sd = sqrt(20/3) -> d = 1 / 2.5820 = 0.3873
        self.assertAlmostEqual(result.cohens_d, 0.3873, places=3)

    def test_ci_brackets_difference(self):
        a = [2, 4, 6, 8]
        b = [1, 3, 5, 7]
        result = exp.compare_continuous("m", "d", "A", "B", a, b)
        self.assertLessEqual(result.ci_low, result.mean_diff)
        self.assertGreaterEqual(result.ci_high, result.mean_diff)

    def test_clear_separation_is_significant(self):
        a = [10, 11, 12, 10, 11]
        b = [1, 2, 1, 2, 1]
        result = exp.compare_continuous("m", "d", "A", "B", a, b)
        self.assertLess(result.t_p_value, 0.001)
        self.assertLess(result.mw_p_value, 0.05)
        self.assertGreater(result.cohens_d, 3.0)

    def test_requires_two_observations(self):
        with self.assertRaises(ValueError):
            exp.compare_continuous("m", "d", "A", "B", [1.0], [1.0, 2.0])


class ProportionStatsTests(unittest.TestCase):
    def test_rates_and_ztest_reference(self):
        # 30/100 vs 20/100: pooled p=0.25, se=0.061237, z=1.6330, p=0.10247
        result = exp.compare_proportions("r", "d", "A", "B", 30, 100, 20, 100)
        self.assertAlmostEqual(result.rate_a, 0.30)
        self.assertAlmostEqual(result.rate_b, 0.20)
        self.assertAlmostEqual(result.rate_diff, 0.10)
        self.assertAlmostEqual(abs(result.z_statistic), 1.6330, places=3)
        self.assertAlmostEqual(result.z_p_value, 0.10247, places=4)

    def test_chi2_equals_z_squared_for_2x2(self):
        # Without continuity correction, chi-square on a 2x2 equals z^2.
        result = exp.compare_proportions("r", "d", "A", "B", 30, 100, 20, 100)
        self.assertAlmostEqual(
            result.chi2_statistic, result.z_statistic**2, places=4
        )

    def test_wald_ci_reference(self):
        # diff=0.1, se=sqrt(.3*.7/100+.2*.8/100)=0.060828, margin=1.95996*se=0.11922
        result = exp.compare_proportions("r", "d", "A", "B", 30, 100, 20, 100)
        self.assertAlmostEqual(result.ci_low, 0.1 - 0.11922, places=4)
        self.assertAlmostEqual(result.ci_high, 0.1 + 0.11922, places=4)

    def test_requires_nonempty_cohorts(self):
        with self.assertRaises(ValueError):
            exp.compare_proportions("r", "d", "A", "B", 0, 0, 1, 10)


class FakeRunner:
    def __init__(self, rows):
        self.rows = rows
        self.sql = None

    def execute(self, sql, max_results=100):
        self.sql = sql
        return self.rows


class CohortSqlConstructionTests(unittest.TestCase):
    def test_continuous_sql_passes_guardrails_with_join(self):
        spec = exp.CohortSpec(
            metric_table="user_summary",
            metric_column="total_ai_tokens",
            group_table="dim_users",
            group_column="user_role",
            cohort_a="Data Scientist",
            cohort_b="AI Engineer",
        )
        rows = [
            {"cohort": "Data Scientist", "value": 100.0},
            {"cohort": "Data Scientist", "value": 120.0},
            {"cohort": "AI Engineer", "value": 80.0},
            {"cohort": "AI Engineer", "value": 90.0},
        ]
        runner = FakeRunner(rows)
        with mock.patch.object(exp, "build_runner", return_value=runner):
            a, b = exp.fetch_continuous_samples(_config(), spec)
        self.assertEqual(a, [100.0, 120.0])
        self.assertEqual(b, [80.0, 90.0])
        # JOIN form + fully-qualified identifiers + safe literal escaping.
        self.assertIn("main.user_summary m JOIN main.dim_users g", runner.sql)
        self.assertIn("'Data Scientist'", runner.sql)

    def test_proportion_sql_passes_guardrails_single_table(self):
        spec = exp.CohortSpec(
            metric_table="user_summary",
            group_table="user_summary",
            group_column="activity_segment",
            success_expr="is_ai_user",
            cohort_a="high_activity",
            cohort_b="low_activity",
        )
        rows = [
            {"cohort": "high_activity", "n": 100, "successes": 60},
            {"cohort": "low_activity", "n": 200, "successes": 40},
        ]
        runner = FakeRunner(rows)
        with mock.patch.object(exp, "build_runner", return_value=runner):
            result = exp.run_proportion_experiment(_config(), spec)
        self.assertEqual(result.successes_a, 60)
        self.assertEqual(result.n_b, 200)
        self.assertAlmostEqual(result.rate_a, 0.60)
        # No JOIN for a single-table spec.
        self.assertNotIn("JOIN", runner.sql)

    def test_injection_guard_on_identifiers(self):
        spec = exp.CohortSpec(
            metric_table="user_summary",
            metric_column="total_ai_tokens; DROP TABLE x",
            group_table="user_summary",
            group_column="activity_segment",
            cohort_a="a",
            cohort_b="b",
        )
        with mock.patch.object(exp, "build_runner", return_value=FakeRunner([])):
            with self.assertRaises(ValueError):
                exp.fetch_continuous_samples(_config(), spec)


_DUCKDB = AgentConfig(
    openai_api_key="x", qdrant_url="x", qdrant_api_key="x",
    qdrant_collection_name="c", backend="duckdb",
).duckdb_path


@unittest.skipUnless(
    os.path.exists(_DUCKDB), "built DuckDB warehouse not present"
)
class LiveDuckDBExperimentTests(unittest.TestCase):
    def test_proportion_experiment_against_warehouse(self):
        spec = exp.CohortSpec(
            metric_table="user_summary",
            group_table="user_summary",
            group_column="activity_segment",
            success_expr="is_ai_user",
            cohort_a="high_activity",
            cohort_b="low_activity",
        )
        result = exp.run_proportion_experiment(_config(), spec)
        self.assertGreater(result.n_a, 0)
        self.assertGreater(result.n_b, 0)
        self.assertTrue(0.0 <= result.rate_a <= 1.0)
        self.assertIn("z-test", exp.format_comparison(result))


if __name__ == "__main__":
    unittest.main()
