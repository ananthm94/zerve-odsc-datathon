import unittest

from analytics_agent.core.sql import SQLValidationError, ensure_default_limit, validate_sql


ALLOWED_TABLES = {"stg_events", "user_events", "user_summary"}
FQ_TABLES = {
    "`my-zerve-project.zerveevents.stg_events`",
    "`my-zerve-project.zerveevents.user_events`",
    "`my-zerve-project.zerveevents.user_summary`",
}


class SQLValidationTests(unittest.TestCase):
    def test_accepts_safe_select(self):
        result = validate_sql(
            "select count(*) as total_users from `my-zerve-project.zerveevents.user_summary`",
            ALLOWED_TABLES,
            FQ_TABLES,
        )

        self.assertEqual(result.referenced_tables, {"user_summary"})

    def test_rejects_non_select(self):
        with self.assertRaises(SQLValidationError):
            validate_sql(
                "delete from `my-zerve-project.zerveevents.user_summary` where true",
                ALLOWED_TABLES,
                FQ_TABLES,
            )

    def test_rejects_raw_table_access(self):
        with self.assertRaises(SQLValidationError):
            validate_sql(
                "select * from `my-zerve-project.zerveevents.zerve_events_raw`",
                ALLOWED_TABLES,
                FQ_TABLES,
            )

    def test_rejects_unqualified_allowed_table(self):
        with self.assertRaises(SQLValidationError):
            validate_sql("select * from user_summary", ALLOWED_TABLES, FQ_TABLES)

    def test_adds_limit_to_detail_query(self):
        sql = ensure_default_limit(
            "select user_id from `my-zerve-project.zerveevents.user_summary`", limit=10
        )

        self.assertTrue(sql.endswith("LIMIT 10"))

    def test_does_not_add_limit_to_aggregate_query(self):
        sql = ensure_default_limit(
            "select count(*) from `my-zerve-project.zerveevents.user_summary`", limit=10
        )

        self.assertNotIn("LIMIT", sql)


if __name__ == "__main__":
    unittest.main()
