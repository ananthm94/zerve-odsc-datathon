"""Dashboard data layer (v0.7+).

Canned, read-only queries that power the Streamlit dashboard. Unlike the agent,
these are author-written (not LLM-generated) trusted queries over the dbt marts.
They still run through the shared ``QueryRunner`` so the same backend (DuckDB by
default, BigQuery optionally) and read-only connection guarantees apply.

Every query accepts an optional :class:`DashboardFilters`. Filtering is unified
around two ideas so all charts stay mutually consistent:

* a **date range** applied to any event-grain query (``event_date``), and
* a **user cohort** — the user-attribute filters (role, segment, geo, device, os)
  resolve to a single ``user_id IN (SELECT ... )`` subquery, so every chart shows
  the same population.

``agg_daily_activity`` is pre-aggregated and not sliced by user, so daily metrics
recompute from ``stg_events`` whenever user filters are active (the fast pre-agg
path is used only when no user filter is set).

Results are intended to be cached by the caller (Streamlit ``st.cache_data``)
since the warehouse is static between dbt builds.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from analytics_agent.config import AgentConfig, load_config
from analytics_agent.runners import QueryRunner, build_runner


# --- Filters -----------------------------------------------------------------


# Maps a DashboardFilters field to (table alias, column) in the user-cohort
# subquery. ``u`` = dim_users (modal attributes), ``s`` = user_summary.
_USER_ATTR_COLUMNS: dict[str, tuple[str, str]] = {
    "user_roles": ("u", "user_role"),
    "countries": ("u", "country"),
    "continents": ("u", "continent"),
    "device_types": ("u", "device_type"),
    "operating_systems": ("u", "os"),
    "activity_segments": ("s", "activity_segment"),
}


@dataclass(frozen=True)
class DashboardFilters:
    """A hashable bundle of dashboard filter selections.

    Empty tuples / ``None`` mean "no filter on this dimension". Frozen so it can
    key Streamlit's cache via :attr:`cache_key`.
    """

    date_start: date | None = None
    date_end: date | None = None
    user_roles: tuple[str, ...] = ()
    activity_segments: tuple[str, ...] = ()
    countries: tuple[str, ...] = ()
    continents: tuple[str, ...] = ()
    device_types: tuple[str, ...] = ()
    operating_systems: tuple[str, ...] = ()

    @property
    def has_user_filters(self) -> bool:
        return any(getattr(self, f) for f in _USER_ATTR_COLUMNS)

    @property
    def has_any(self) -> bool:
        return (
            self.has_user_filters
            or self.date_start is not None
            or self.date_end is not None
        )

    @property
    def cache_key(self) -> tuple:
        """A fully-hashable signature for Streamlit's cache_data."""
        return (
            self.date_start.isoformat() if self.date_start else None,
            self.date_end.isoformat() if self.date_end else None,
            self.user_roles,
            self.activity_segments,
            self.countries,
            self.continents,
            self.device_types,
            self.operating_systems,
        )


_NO_FILTERS = DashboardFilters()


def _literal(value) -> str:
    """Escape a value as a SQL string literal (single-quote doubling)."""
    return "'" + str(value).replace("'", "''") + "'"


def _in_list(col: str, values: tuple[str, ...]) -> str:
    return f"{col} IN ({', '.join(_literal(v) for v in values)})"


def _matching_users_subquery(
    config: AgentConfig, filters: DashboardFilters
) -> str | None:
    """A ``(SELECT user_id ...)`` subquery for the filtered user cohort, or None.

    Joins dim_users (modal role/geo/device/os) and user_summary (activity
    segment) on ``user_id`` and applies the active user-attribute predicates.
    """
    if not filters.has_user_filters:
        return None
    dim = _table(config, "dim_users")
    summ = _table(config, "user_summary")
    preds = [
        _in_list(f"{alias}.{col}", getattr(filters, fld))
        for fld, (alias, col) in _USER_ATTR_COLUMNS.items()
        if getattr(filters, fld)
    ]
    return (
        f"(SELECT u.user_id FROM {dim} u "
        f"JOIN {summ} s ON u.user_id = s.user_id "
        f"WHERE {' AND '.join(preds)})"
    )


def _filter_conditions(
    config: AgentConfig,
    filters: DashboardFilters,
    *,
    date_col: str | None = None,
    user_col: str | None = None,
) -> list[str]:
    """Build the WHERE predicates a query should apply for the active filters."""
    conds: list[str] = []
    if date_col:
        if filters.date_start:
            conds.append(f"{date_col} >= {_literal(filters.date_start.isoformat())}")
        if filters.date_end:
            conds.append(f"{date_col} <= {_literal(filters.date_end.isoformat())}")
    if user_col:
        sub = _matching_users_subquery(config, filters)
        if sub:
            conds.append(f"{user_col} IN {sub}")
    return conds


def _where(*condition_groups: list[str]) -> str:
    conds = [c for group in condition_groups for c in group]
    return ("WHERE " + " AND ".join(conds)) if conds else ""


# --- Plumbing ----------------------------------------------------------------


def _df(runner: QueryRunner, sql: str, max_results: int = 100_000) -> pd.DataFrame:
    return pd.DataFrame(runner.execute(sql, max_results=max_results))


def _table(config: AgentConfig, name: str) -> str:
    """Backend-correct identifier for a mart (DuckDB schema.table or BigQuery FQTN)."""
    return config.fully_qualified_tables[name]


def date_bounds(config: AgentConfig | None = None) -> tuple[date | None, date | None]:
    """Min and max ``event_date`` in the warehouse, for the date-range picker."""
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "agg_daily_activity")
    rows = runner.execute(
        f"SELECT MIN(event_date) AS lo, MAX(event_date) AS hi FROM {t}", max_results=1
    )
    if not rows:
        return None, None
    return rows[0]["lo"], rows[0]["hi"]


# --- Headline + existing charts ----------------------------------------------


def kpis(
    config: AgentConfig | None = None, filters: DashboardFilters = _NO_FILTERS
) -> dict[str, float]:
    """Headline scalars, computed from stg_events so date/user filters apply."""
    config = config or load_config()
    runner = build_runner(config)
    events = _table(config, "stg_events")
    where = _where(
        _filter_conditions(config, filters, date_col="event_date", user_col="user_id")
    )

    row = _df(
        runner,
        f"""
        SELECT
            COUNT(DISTINCT user_id) AS total_users,
            COUNT(*) AS total_events,
            COUNT(*) FILTER (WHERE event_name = '$ai_generation') AS ai_generations,
            SUM(COALESCE(ai_input_tokens, 0) + COALESCE(ai_output_tokens, 0))
                FILTER (WHERE event_name = '$ai_generation') AS total_ai_tokens,
            SUM(credits_used) AS total_credits_used,
            COUNT(*) FILTER (WHERE event_name = '$exception') AS total_exceptions,
            COUNT(DISTINCT user_id) FILTER (WHERE event_name = '$ai_generation')
                AS ai_users
        FROM {events}
        {where}
        """,
        max_results=1,
    ).iloc[0]

    total_users = float(row["total_users"] or 0)
    ai_users = float(row["ai_users"] or 0)
    return {
        "total_users": total_users,
        "total_events": float(row["total_events"] or 0),
        "ai_generations": float(row["ai_generations"] or 0),
        "total_ai_tokens": float(row["total_ai_tokens"] or 0),
        "total_credits_used": float(row["total_credits_used"] or 0),
        "total_exceptions": float(row["total_exceptions"] or 0),
        "ai_users": ai_users,
        "ai_adoption_rate": (ai_users / total_users) if total_users else 0.0,
    }


def daily_activity_df(
    config: AgentConfig | None = None, filters: DashboardFilters = _NO_FILTERS
) -> pd.DataFrame:
    """One row per day: active users and the event-type breakdown.

    Uses the pre-aggregated mart when no user filter is active; otherwise
    recomputes from stg_events so the user cohort is honoured.
    """
    config = config or load_config()
    runner = build_runner(config)

    if not filters.has_user_filters:
        t = _table(config, "agg_daily_activity")
        where = _where(_filter_conditions(config, filters, date_col="event_date"))
        sql = f"""
            SELECT event_date, active_users, total_events,
                   ai_generations, exceptions, pageviews, credit_events
            FROM {t}
            {where}
            ORDER BY event_date
        """
    else:
        events = _table(config, "stg_events")
        where = _where(
            _filter_conditions(
                config, filters, date_col="event_date", user_col="user_id"
            )
        )
        sql = f"""
            SELECT event_date,
                   COUNT(DISTINCT user_id) AS active_users,
                   COUNT(*) AS total_events,
                   COUNT(*) FILTER (WHERE event_name = '$ai_generation') AS ai_generations,
                   COUNT(*) FILTER (WHERE event_name = '$exception') AS exceptions,
                   COUNT(*) FILTER (WHERE event_name = '$pageview') AS pageviews,
                   COUNT(*) FILTER (
                       WHERE event_name IN ('credits_used', 'addon_credits_used')
                   ) AS credit_events
            FROM {events}
            {where}
            GROUP BY 1
            ORDER BY 1
        """

    df = _df(runner, sql)
    if not df.empty:
        df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def ai_tokens_by_model_df(
    config: AgentConfig | None = None, filters: DashboardFilters = _NO_FILTERS
) -> pd.DataFrame:
    """AI usage aggregated by model: generations, tokens, and average latency."""
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "fct_ai_generations")
    where = _where(
        ["ai_model IS NOT NULL"],
        _filter_conditions(config, filters, date_col="event_date", user_col="user_id"),
    )
    return _df(
        runner,
        f"""
        SELECT ai_model,
               COUNT(*) AS generations,
               SUM(total_tokens) AS total_tokens,
               AVG(ai_latency_seconds) AS avg_latency_seconds
        FROM {t}
        {where}
        GROUP BY ai_model
        ORDER BY total_tokens DESC NULLS LAST
        """,
    )


def ai_tokens_by_provider_df(
    config: AgentConfig | None = None, filters: DashboardFilters = _NO_FILTERS
) -> pd.DataFrame:
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "fct_ai_generations")
    where = _where(
        ["ai_provider IS NOT NULL"],
        _filter_conditions(config, filters, date_col="event_date", user_col="user_id"),
    )
    return _df(
        runner,
        f"""
        SELECT ai_provider,
               COUNT(*) AS generations,
               SUM(total_tokens) AS total_tokens
        FROM {t}
        {where}
        GROUP BY ai_provider
        ORDER BY total_tokens DESC NULLS LAST
        """,
    )


def activity_segment_df(
    config: AgentConfig | None = None, filters: DashboardFilters = _NO_FILTERS
) -> pd.DataFrame:
    """User distribution and AI adoption across activity segments (user cohort)."""
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "user_summary")
    where = _where(_filter_conditions(config, filters, user_col="user_id"))
    return _df(
        runner,
        f"""
        SELECT activity_segment,
               COUNT(*) AS users,
               SUM(CASE WHEN is_ai_user THEN 1 ELSE 0 END) AS ai_users,
               AVG(total_event_count) AS avg_events
        FROM {t}
        {where}
        GROUP BY activity_segment
        ORDER BY users DESC
        """,
    )


def users_by_role_df(
    config: AgentConfig | None = None, filters: DashboardFilters = _NO_FILTERS
) -> pd.DataFrame:
    """User counts by self-reported role (nulls labelled 'Unknown')."""
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "dim_users")
    where = _where(_filter_conditions(config, filters, user_col="user_id"))
    return _df(
        runner,
        f"""
        SELECT COALESCE(user_role, 'Unknown') AS user_role,
               COUNT(*) AS users
        FROM {t}
        {where}
        GROUP BY 1
        ORDER BY users DESC
        """,
    )


def exceptions_by_os_df(
    config: AgentConfig | None = None, filters: DashboardFilters = _NO_FILTERS
) -> pd.DataFrame:
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "fct_exceptions")
    where = _where(
        _filter_conditions(config, filters, date_col="event_date", user_col="user_id")
    )
    return _df(
        runner,
        f"""
        SELECT COALESCE(os, 'Unknown') AS os,
               COUNT(*) AS exceptions
        FROM {t}
        {where}
        GROUP BY 1
        ORDER BY exceptions DESC
        LIMIT 10
        """,
    )


def feature_usage_df(
    config: AgentConfig | None = None, filters: DashboardFilters = _NO_FILTERS
) -> pd.DataFrame:
    """Per feature area: event volume, distinct-user reach, and adoption rate.

    Adoption = users who used the feature / users active in the filtered scope,
    so it always reflects the current cohort and date window.
    """
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "stg_feature_events")
    where = _where(
        _filter_conditions(config, filters, date_col="event_date", user_col="user_id")
    )
    return _df(
        runner,
        f"""
        WITH scoped AS (
            SELECT user_id, feature_category FROM {t} {where}
        )
        SELECT feature_category,
               COUNT(*) AS events,
               COUNT(DISTINCT user_id) AS reach,
               COUNT(DISTINCT user_id) * 1.0
                   / NULLIF((SELECT COUNT(DISTINCT user_id) FROM scoped), 0)
                   AS adoption_rate
        FROM scoped
        GROUP BY feature_category
        ORDER BY events DESC
        """,
    )


def feature_trend_df(
    config: AgentConfig | None = None,
    filters: DashboardFilters = _NO_FILTERS,
    categories: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Daily event volume per feature area (long-form), for a trend line.

    ``categories`` restricts to the chosen feature areas; empty means all.
    """
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "stg_feature_events")
    cat_cond = [_in_list("feature_category", categories)] if categories else []
    where = _where(
        cat_cond,
        _filter_conditions(config, filters, date_col="event_date", user_col="user_id"),
    )
    df = _df(
        runner,
        f"""
        SELECT event_date, feature_category, COUNT(*) AS events
        FROM {t}
        {where}
        GROUP BY 1, 2
        ORDER BY 1
        """,
    )
    if not df.empty:
        df["event_date"] = pd.to_datetime(df["event_date"])
    return df


def retention_matrix_df(
    config: AgentConfig | None = None,
    filters: DashboardFilters = _NO_FILTERS,
    max_offset: int = 12,
) -> pd.DataFrame:
    """Cohort retention triangle: cohort_week x week_offset -> retention rate.

    The date filter (if any) restricts which cohorts are shown (by cohort_week);
    user filters restrict the population. Capped at ``max_offset`` weeks.
    """
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "fct_user_activity_weekly")
    where = _where(
        _filter_conditions(config, filters, date_col="cohort_week", user_col="user_id")
    )
    df = _df(
        runner,
        f"""
        WITH base AS (
            SELECT user_id, cohort_week, week_offset FROM {t} {where}
        ),
        sizes AS (
            SELECT cohort_week, COUNT(DISTINCT user_id) AS cohort_size
            FROM base WHERE week_offset = 0 GROUP BY 1
        )
        SELECT b.cohort_week,
               b.week_offset,
               COUNT(DISTINCT b.user_id) AS active_users,
               s.cohort_size,
               COUNT(DISTINCT b.user_id) * 1.0 / NULLIF(s.cohort_size, 0) AS retention
        FROM base b
        JOIN sizes s ON b.cohort_week = s.cohort_week
        WHERE b.week_offset <= {int(max_offset)}
        GROUP BY 1, 2, s.cohort_size
        ORDER BY 1, 2
        """,
    )
    if not df.empty:
        df["cohort_week"] = pd.to_datetime(df["cohort_week"])
    return df


def retention_curve_df(
    config: AgentConfig | None = None,
    filters: DashboardFilters = _NO_FILTERS,
    max_offset: int = 12,
) -> pd.DataFrame:
    """Pooled retention curve: % of all filtered users active at each week offset."""
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "fct_user_activity_weekly")
    where = _where(
        _filter_conditions(config, filters, date_col="cohort_week", user_col="user_id")
    )
    return _df(
        runner,
        f"""
        WITH base AS (
            SELECT user_id, week_offset FROM {t} {where}
        ),
        sizes AS (SELECT COUNT(DISTINCT user_id) AS total FROM base WHERE week_offset = 0)
        SELECT week_offset,
               COUNT(DISTINCT user_id) AS active_users,
               (SELECT total FROM sizes) AS cohort_size,
               COUNT(DISTINCT user_id) * 1.0 / NULLIF((SELECT total FROM sizes), 0)
                   AS retention
        FROM base
        WHERE week_offset <= {int(max_offset)}
        GROUP BY 1
        ORDER BY 1
        """,
    )


def funnel_df(
    config: AgentConfig | None = None,
    filters: DashboardFilters = _NO_FILTERS,
    steps: tuple[tuple[str, tuple[str, ...]], ...] = (),
    ordered: bool = True,
) -> pd.DataFrame:
    """Conversion funnel over an ordered list of milestone steps.

    ``steps`` is an ordered tuple of ``(label, event_names)``. A user passes step
    *k* if they have an event for every step up to *k*; when ``ordered`` is True
    each step's first occurrence must also be no earlier than the previous step's
    (sequential first-touch). When False, the steps need only all be present in
    any order (cumulative reach). Returns one row per step with the surviving user
    count. Honours the active date/user filters.
    """
    if not steps:
        return pd.DataFrame(columns=["step", "label", "users"])
    config = config or load_config()
    runner = build_runner(config)
    t = _table(config, "stg_events")

    all_events = tuple(dict.fromkeys(e for _, evs in steps for e in evs))
    base = _filter_conditions(
        config, filters, date_col="event_date", user_col="user_id"
    )
    where = _where([_in_list("event_name", all_events)], base)

    mins = ",\n            ".join(
        f"MIN(CASE WHEN {_in_list('event_name', evs)} THEN event_timestamp END) AS t{i}"
        for i, (_, evs) in enumerate(steps, 1)
    )
    selects = []
    for i in range(1, len(steps) + 1):
        conds = [f"t{j} IS NOT NULL" for j in range(1, i + 1)]
        if ordered:
            conds += [f"t{j} >= t{j - 1}" for j in range(2, i + 1)]
        selects.append(f"COUNT(*) FILTER (WHERE {' AND '.join(conds)}) AS step{i}")

    sql = f"""
        WITH ev AS (
            SELECT user_id, event_name, event_timestamp FROM {t} {where}
        ),
        firsts AS (
            SELECT user_id,
            {mins}
            FROM ev GROUP BY user_id
        )
        SELECT {', '.join(selects)} FROM firsts
    """
    rows = runner.execute(sql, max_results=1)
    counts = rows[0] if rows else {}
    return pd.DataFrame(
        [
            {"step": i, "label": label, "users": int(counts.get(f"step{i}", 0) or 0)}
            for i, (label, _) in enumerate(steps, 1)
        ]
    )


def distinct_values(
    table: str, column: str, config: AgentConfig | None = None, limit: int = 50
) -> list[str]:
    """Distinct non-null values of a column, for populating selectors.

    ``table``/``column`` are validated against the dbt allow-list and a strict
    identifier pattern, since they originate from UI selections.
    """
    config = config or load_config()
    if table not in config.fully_qualified_tables:
        raise ValueError(f"Table {table!r} is not allow-listed.")
    if not column.replace("_", "").isalnum():
        raise ValueError(f"Unsafe column identifier: {column!r}")
    runner = build_runner(config)
    t = _table(config, table)
    rows = runner.execute(
        f"SELECT DISTINCT {column} AS v FROM {t} "
        f"WHERE {column} IS NOT NULL ORDER BY 1 LIMIT {int(limit)}",
        max_results=limit,
    )
    return [str(r["v"]) for r in rows]


__all__ = [
    "DashboardFilters",
    "date_bounds",
    "kpis",
    "distinct_values",
    "daily_activity_df",
    "ai_tokens_by_model_df",
    "ai_tokens_by_provider_df",
    "activity_segment_df",
    "users_by_role_df",
    "exceptions_by_os_df",
    "feature_usage_df",
    "feature_trend_df",
    "retention_matrix_df",
    "retention_curve_df",
    "funnel_df",
]
