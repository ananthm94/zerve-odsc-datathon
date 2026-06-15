"""Invariant checks for the AHA-moment marts, run against the built DuckDB.

These query the real warehouse (dim_user_outcomes, fct_activation_milestones,
agg_aha_lift) and assert structural invariants the dashboard relies on. They skip
cleanly when the warehouse has not been built, so the suite still runs offline.
"""

import os
import unittest

import duckdb

_DB = os.environ.get("DUCKDB_PATH", "zerve_transform/zerve.duckdb")


def _has_warehouse() -> bool:
    if not os.path.exists(_DB):
        return False
    try:
        con = duckdb.connect(_DB, read_only=True)
        names = {r[0] for r in con.execute(
            "select table_name from information_schema.tables"
        ).fetchall()}
        con.close()
    except Exception:
        return False
    return {
        "dim_user_outcomes", "fct_activation_milestones", "agg_aha_lift"
    }.issubset(names)


@unittest.skipUnless(_has_warehouse(), "DuckDB warehouse with AHA marts not built")
class AhaModelInvariantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.con = duckdb.connect(_DB, read_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.con.close()

    def q(self, sql):
        return self.con.execute(sql).fetchone()

    def test_outcomes_unique_per_user(self):
        total, distinct = self.q(
            "select count(*), count(distinct user_id) from dim_user_outcomes"
        )
        self.assertEqual(total, distinct)
        self.assertGreater(total, 0)

    def test_churn_implies_activated_in_week1(self):
        # is_churned_w4 must never be true for a user who was not active in week 1.
        violations, = self.q(
            "select count(*) from dim_user_outcomes "
            "where is_churned_w4 and not active_week1"
        )
        self.assertEqual(violations, 0)

    def test_churn_excludes_converters_and_unobservable(self):
        bad, = self.q(
            "select count(*) from dim_user_outcomes "
            "where is_churned_w4 and (is_converter or not outcome_observable "
            "or returned_after_w4)"
        )
        self.assertEqual(bad, 0)

    def test_lifecycle_stage_partitions_users(self):
        rows = self.con.execute(
            "select lifecycle_stage, count(*) from dim_user_outcomes group by 1"
        ).fetchall()
        stages = {r[0] for r in rows}
        self.assertTrue(stages.issubset({
            "converted", "retained_free", "churned_w4",
            "never_activated", "too_new",
        }))
        # converted stage <=> is_converter
        mismatch, = self.q(
            "select count(*) from dim_user_outcomes "
            "where (lifecycle_stage = 'converted') <> is_converter"
        )
        self.assertEqual(mismatch, 0)

    def test_days_to_convert_only_for_converters(self):
        bad, = self.q(
            "select count(*) from dim_user_outcomes "
            "where (days_to_convert is not null) <> is_converter"
        )
        self.assertEqual(bad, 0)

    def test_milestones_one_row_per_outcome_user(self):
        o, m, joined = (
            self.q("select count(*) from dim_user_outcomes")[0],
            self.q("select count(*) from fct_activation_milestones")[0],
            self.q(
                "select count(*) from dim_user_outcomes o "
                "join fct_activation_milestones m using(user_id)"
            )[0],
        )
        self.assertEqual(o, m)
        self.assertEqual(o, joined)

    def test_w1_flag_implies_timestamp_present(self):
        # did_deploy_w1 cannot be true without a first_deploy_ts.
        bad, = self.q(
            "select count(*) from fct_activation_milestones "
            "where did_deploy_w1 and first_deploy_ts is null"
        )
        self.assertEqual(bad, 0)

    def test_aha_lift_definition(self):
        # conversion_lift == conversion_rate / baseline_rate, and the split is clean.
        rows = self.con.execute(
            "select action, signal_type, conversion_rate, baseline_rate, "
            "conversion_lift, converters_among_them, users_did_w1 from agg_aha_lift"
        ).fetchall()
        self.assertGreater(len(rows), 0)
        for action, stype, cr, base, lift, conv, n in rows:
            self.assertIn(stype, ("product", "paywall"))
            self.assertLessEqual(conv, n)  # converters can't exceed cohort
            self.assertAlmostEqual(cr, conv / n, places=6)
            if base:
                self.assertAlmostEqual(lift, cr / base, places=4)

    def test_conversion_rate_is_plausible(self):
        # Sanity anchor: overall conversion sits in the low single digits.
        signups, converters = self.q(
            "select count(*), sum(is_converter::int) from dim_user_outcomes"
        )
        rate = converters / signups
        self.assertGreater(rate, 0.02)
        self.assertLess(rate, 0.15)


if __name__ == "__main__":
    unittest.main()
