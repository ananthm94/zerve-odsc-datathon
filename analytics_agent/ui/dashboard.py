"""The "Dashboard" tab — prebuilt activation/conversion metrics and product
analytics over the dbt marts, with its own inline filters.

The cached ``@st.cache_data`` wrappers below take a hashable ``key`` (the
filters' ``cache_key``) plus the filters object passed with a leading underscore
so Streamlit keys the cache on ``key`` and skips hashing the dataclass itself.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from analytics_agent import dashboard
from analytics_agent.core.config import ConfigError
from analytics_agent.ui import common


# --- Cached dashboard data -------------------------------------------------


@st.cache_data(show_spinner=False)
def _kpis(key, _filters):
    return dashboard.kpis(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _daily_activity(key, _filters):
    return dashboard.daily_activity_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _tokens_by_model(key, _filters):
    return dashboard.ai_tokens_by_model_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _tokens_by_provider(key, _filters):
    return dashboard.ai_tokens_by_provider_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _activity_segments(key, _filters):
    return dashboard.activity_segment_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _users_by_role(key, _filters):
    return dashboard.users_by_role_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _exceptions_by_os(key, _filters):
    return dashboard.exceptions_by_os_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _feature_usage(key, _filters):
    return dashboard.feature_usage_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _feature_trend(key, _filters, categories):
    return dashboard.feature_trend_df(common.config(), _filters, categories)


@st.cache_data(show_spinner=False)
def _retention_matrix(key, _filters, max_offset):
    return dashboard.retention_matrix_df(common.config(), _filters, max_offset)


@st.cache_data(show_spinner=False)
def _retention_curve(key, _filters, max_offset):
    return dashboard.retention_curve_df(common.config(), _filters, max_offset)


@st.cache_data(show_spinner=False)
def _funnel(key, _filters, steps, ordered):
    return dashboard.funnel_df(common.config(), _filters, steps, ordered)


@st.cache_data(show_spinner=False)
def _aha_kpis(key, _filters):
    return dashboard.aha_kpis(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _aha_funnel(key, _filters):
    return dashboard.aha_funnel_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _aha_lift(signal_type):
    return dashboard.aha_lift_df(common.config(), signal_type)


@st.cache_data(show_spinner=False)
def _aha_depth(key, _filters):
    return dashboard.aha_depth_df(common.config(), _filters)


@st.cache_data(show_spinner=False)
def _churn_watch(key, _filters):
    return dashboard.churn_watch_df(common.config(), _filters)


# Funnel presets: ordered (label, event_names) steps grounded in the event feed.
_FUNNEL_PRESETS = {
    "Activation": (
        ("Sign in", ("sign_in",)),
        ("Onboarding", ("skip_onboarding_form",)),
        ("Run block", ("run_block",)),
        ("AI generation", ("$ai_generation",)),
        ("Use credits", ("credits_used",)),
    ),
    "Agent workflow": (
        ("Agent created", ("agent_worker_created",)),
        (
            "Tool call",
            (
                "agent_tool_call_create_block_tool",
                "agent_tool_call_run_block_tool",
                "agent_tool_call_get_block_tool",
            ),
        ),
        ("Finish ticket", ("agent_tool_call_finish_ticket_tool",)),
    ),
}


def _build_filters() -> dashboard.DashboardFilters:
    """Render the Dashboard-tab filter controls and return the selected filters.

    Rendered inline inside the Dashboard tab (not the global sidebar) so they only
    appear where they apply. Selectors are populated from the warehouse.
    """
    with st.expander("Filters", expanded=False):
        lo, hi = common.date_bounds()
        date_start = date_end = None
        if lo and hi:
            picked = st.date_input(
                "Date range", value=(lo, hi), min_value=lo, max_value=hi
            )
            if isinstance(picked, (list, tuple)) and len(picked) == 2:
                date_start, date_end = picked

        col_a, col_b = st.columns(2)

        def _ms(container, label, table, column):
            return tuple(container.multiselect(label, common.distinct(table, column)))

        filters = dashboard.DashboardFilters(
            date_start=date_start,
            date_end=date_end,
            user_roles=_ms(col_a, "User role", "dim_users", "user_role"),
            activity_segments=_ms(
                col_a, "Activity segment", "user_summary", "activity_segment"
            ),
            continents=_ms(col_a, "Continent", "dim_users", "continent"),
            countries=_ms(col_b, "Country", "dim_users", "country"),
            device_types=_ms(col_b, "Device type", "dim_users", "device_type"),
            operating_systems=_ms(col_b, "Operating system", "dim_users", "os"),
        )
        if filters.has_any:
            st.success("Filters active")
    return filters


def render() -> None:
    st.subheader("Activation & Conversion")
    st.caption(
        "Which early actions — the **AHA moments** — separate users who pay from "
        "users who churn. Conversion is fast here (most upgrades happen in week 1), "
        "so the lens is the first 7 days after signup."
    )

    try:
        filters = _build_filters()
        fkey = filters.cache_key
        ak = _aha_kpis(fkey, filters)
        afn = _aha_funnel(fkey, filters)
        lift_product = _aha_lift("product")
        lift_paywall = _aha_lift("paywall")
        depth = _aha_depth(fkey, filters)
        churn = _churn_watch(fkey, filters)
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
    except Exception as exc:
        st.error(f"Could not load dashboard data: {exc}")
        st.stop()

    if filters.has_any:
        st.caption("Showing a filtered view — see Filters above.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Signups", common.fmt(ak["signups"]))
    c2.metric("Conversion rate", f"{ak['conversion_rate'] * 100:.1f}%",
              help="Signup users who upgraded their subscription.")
    c3.metric("W4 churn rate", f"{ak['w4_churn_rate'] * 100:.1f}%",
              help="Activated, non-paying users who never returned after week 4 "
                   "(observable cohort only).")
    c4.metric("Median days to pay", f"{ak['median_days_to_convert']:.0f}",
              help="Median days from signup to first upgrade, among converters.")

    st.divider()
    st.markdown("### The AHA leaderboard")
    st.caption(
        "How much more likely a user is to convert if they take an action in week 1, "
        "versus the signup baseline. **Product actions** are levers Zerve can drive in "
        "onboarding; **paywall signals** are shown separately because they *are* "
        "purchase intent, not causes of it."
    )

    def _lift_chart(df, title):
        if df.empty:
            st.caption("No data in scope.")
            return
        d = df.assign(
            conv_pct=(df["conversion_rate"] * 100).round(1),
            ret_pct=(df["retention_rate"] * 100).round(1),
            base_pct=(df["baseline_rate"] * 100).round(1),
            lift=df["conversion_lift"].round(1),
        )
        bars = (
            alt.Chart(d)
            .mark_bar(color="#7C3AED")
            .encode(
                y=alt.Y("action:N", sort=d["action"].tolist(), title=None),
                x=alt.X("conversion_lift:Q", title="Conversion lift (× baseline)"),
                tooltip=[
                    alt.Tooltip("action:N", title="Action"),
                    alt.Tooltip("users_did_w1:Q", title="Users (week 1)", format=","),
                    alt.Tooltip("conv_pct:Q", title="Conversion %"),
                    alt.Tooltip("base_pct:Q", title="Baseline %"),
                    alt.Tooltip("lift:Q", title="Lift ×"),
                    alt.Tooltip("ret_pct:Q", title="W4 retention %"),
                ],
            )
        )
        rule = alt.Chart(pd.DataFrame({"x": [1.0]})).mark_rule(
            color="#94A3B8", strokeDash=[4, 4]
        ).encode(x="x:Q")
        st.markdown(f"#### {title}")
        st.altair_chart(
            (bars + rule).properties(height=42 * len(d) + 20),
            use_container_width=True,
        )

    _lift_chart(lift_product, "Product actions (the levers)")
    _lift_chart(lift_paywall, "Paywall / intent signals (symptoms, not causes)")

    st.divider()
    st.markdown("### Week-1 depth is the strongest signal")
    st.caption(
        "Breadth of the first week — how many distinct days a user is active — "
        "predicts both conversion and retention better than any single feature."
    )
    if not depth.empty:
        dd = depth.assign(
            Conversion=(depth["conversion_rate"] * 100).round(1),
            Retention=(depth["retention_rate"] * 100).round(1),
        )
        long = dd.melt(
            id_vars=["depth_bucket", "sort_key"],
            value_vars=["Conversion", "Retention"],
            var_name="metric", value_name="pct",
        )
        depth_chart = (
            alt.Chart(long)
            .mark_bar()
            .encode(
                x=alt.X("depth_bucket:N", sort=dd["depth_bucket"].tolist(),
                        title="Distinct active days in week 1"),
                xOffset="metric:N",
                y=alt.Y("pct:Q", title="%"),
                color=alt.Color("metric:N", scale=alt.Scale(
                    domain=["Conversion", "Retention"],
                    range=["#7C3AED", "#22C55E"]), title=None),
                tooltip=["depth_bucket:N", "metric:N",
                         alt.Tooltip("pct:Q", title="%")],
            )
            .properties(height=300)
        )
        st.altair_chart(depth_chart, use_container_width=True)
        st.dataframe(
            dd[["depth_bucket", "users", "Conversion", "Retention"]],
            use_container_width=True, hide_index=True,
            column_config={
                "depth_bucket": st.column_config.TextColumn("Week-1 active days"),
                "users": st.column_config.NumberColumn("Users", format="%d"),
                "Conversion": st.column_config.NumberColumn("Conversion %", format="%.1f%%"),
                "Retention": st.column_config.NumberColumn("W4 retention %", format="%.1f%%"),
            },
        )

    st.divider()
    fcol, ccol = st.columns([3, 2])
    with fcol:
        st.markdown("### Activation reach")
        st.caption(
            "Distinct signup users who ever reach each milestone. Not a strict "
            "funnel — some users pay without deploying."
        )
        if not afn.empty and afn["users"].iloc[0]:
            af = afn.assign(pct=(afn["pct_of_signups"] * 100).round(1))
            reach_chart = (
                alt.Chart(af)
                .mark_bar(color="#7C3AED")
                .encode(
                    y=alt.Y("label:N", sort=af["label"].tolist(), title=None),
                    x=alt.X("users:Q", title="Users"),
                    tooltip=[
                        alt.Tooltip("label:N", title="Milestone"),
                        alt.Tooltip("users:Q", title="Users", format=","),
                        alt.Tooltip("pct:Q", title="% of signups"),
                    ],
                )
                .properties(height=48 * len(af) + 20)
            )
            st.altair_chart(reach_chart, use_container_width=True)
    with ccol:
        st.markdown("### Churn watch")
        st.caption("Activated, non-paying users by recency — the at-risk watchlist.")
        if not churn.empty:
            st.bar_chart(churn, x="recency_band", y="users", height=300)
            st.dataframe(churn, use_container_width=True, hide_index=True)

    # --- Product analytics (context) -----------------------------------------
    st.divider()
    with st.expander("Product analytics (context: volume, features, retention)"):
        try:
            k = _kpis(fkey, filters)
            daily = _daily_activity(fkey, filters)
            seg = _activity_segments(fkey, filters)
            by_model = _tokens_by_model(fkey, filters)
            by_provider = _tokens_by_provider(fkey, filters)
            by_role = _users_by_role(fkey, filters)
            exc_os = _exceptions_by_os(fkey, filters)
        except Exception as exc:
            st.error(f"Could not load context data: {exc}")
            st.stop()

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Users", common.fmt(k["total_users"]))
        c2.metric("Events", common.fmt(k["total_events"]))
        c3.metric("AI generations", common.fmt(k["ai_generations"]))
        c4.metric("AI adoption", f"{k['ai_adoption_rate'] * 100:.1f}%")
        c5, c6, c7, c8 = st.columns(4)
        c5.metric("AI tokens", common.fmt(k["total_ai_tokens"]))
        c6.metric("Credits used", common.fmt(k["total_credits_used"]))
        c7.metric("Exceptions", common.fmt(k["total_exceptions"]))
        c8.metric("AI users", common.fmt(k["ai_users"]))

        st.markdown("#### Daily active users")
        if not daily.empty:
            st.line_chart(daily, x="event_date", y="active_users", height=260)
            st.markdown("#### Daily event mix")
            st.area_chart(
                daily,
                x="event_date",
                y=["ai_generations", "exceptions", "pageviews", "credit_events"],
                height=260,
            )

        left, right = st.columns(2)
        with left:
            st.markdown("#### AI tokens by model")
            if not by_model.empty:
                st.bar_chart(by_model, x="ai_model", y="total_tokens", height=300)
            st.markdown("#### Users by role")
            if not by_role.empty:
                st.bar_chart(by_role, x="user_role", y="users", height=300)
        with right:
            st.markdown("#### AI tokens by provider")
            if not by_provider.empty:
                st.bar_chart(by_provider, x="ai_provider", y="total_tokens", height=300)
            st.markdown("#### Exceptions by OS")
            if not exc_os.empty:
                st.bar_chart(exc_os, x="os", y="exceptions", height=300)

        st.markdown("#### Activity segments")
        if not seg.empty:
            st.dataframe(seg, use_container_width=True, hide_index=True)

        st.markdown("### Feature usage")
        feat = _feature_usage(fkey, filters)
        if feat.empty:
            st.caption("No feature events in scope.")
        else:
            fl, fr = st.columns([3, 2])
            with fl:
                st.markdown("#### Events by feature")
                st.bar_chart(feat, x="feature_category", y="events", height=320)
            with fr:
                st.markdown("#### Reach & adoption")
                display = feat.assign(
                    adoption=(feat["adoption_rate"] * 100).round(1)
                )[["feature_category", "reach", "adoption"]]
                st.dataframe(
                    display,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "reach": st.column_config.NumberColumn("Users reached"),
                        "adoption": st.column_config.NumberColumn(
                            "Adoption %", format="%.1f%%"
                        ),
                    },
                )

            st.markdown("#### Usage trend")
            all_cats = feat["feature_category"].tolist()
            default_cats = all_cats[:4]
            chosen = st.multiselect(
                "Feature areas", all_cats, default=default_cats,
                key="feature_trend_cats",
            )
            if chosen:
                trend = _feature_trend(fkey, filters, tuple(chosen))
                if not trend.empty:
                    st.line_chart(
                        trend, x="event_date", y="events",
                        color="feature_category", height=300,
                    )

    st.divider()
    st.markdown("### Weekly cohort retention")
    st.caption(
        "Cohort = the user's first active week. Retention at week N is the share "
        "of the cohort active N weeks later. Week 0 is 100% by definition."
    )
    max_offset = st.slider("Weeks tracked", 4, 24, 12, key="retention_weeks")
    curve = _retention_curve(fkey, filters, max_offset)
    matrix = _retention_matrix(fkey, filters, max_offset)

    if curve.empty:
        st.caption("No retention data in scope.")
    else:
        ret = curve.set_index("week_offset")["retention"]
        m1, m2, m3 = st.columns(3)
        m1.metric("Week 1", f"{ret.get(1, 0) * 100:.1f}%")
        m2.metric("Week 4", f"{ret.get(4, 0) * 100:.1f}%")
        m3.metric(
            f"Week {max_offset}", f"{ret.get(max_offset, 0) * 100:.1f}%"
        )

        st.markdown("#### Retention curve")
        curve_plot = curve.assign(retention_pct=(curve["retention"] * 100).round(2))
        st.line_chart(curve_plot, x="week_offset", y="retention_pct", height=240)

        if not matrix.empty:
            st.markdown("#### Retention triangle")
            heat = matrix.assign(
                cohort=matrix["cohort_week"].dt.strftime("%Y-%m-%d"),
                retention_pct=(matrix["retention"] * 100).round(1),
            )
            chart = (
                alt.Chart(heat)
                .mark_rect()
                .encode(
                    x=alt.X("week_offset:O", title="Weeks since first active"),
                    y=alt.Y("cohort:O", title="Cohort (first week)", sort="descending"),
                    color=alt.Color(
                        "retention_pct:Q",
                        title="Retention %",
                        scale=alt.Scale(scheme="purples"),
                    ),
                    tooltip=[
                        alt.Tooltip("cohort:N", title="Cohort"),
                        alt.Tooltip("week_offset:O", title="Week"),
                        alt.Tooltip("active_users:Q", title="Active users"),
                        alt.Tooltip("cohort_size:Q", title="Cohort size"),
                        alt.Tooltip("retention_pct:Q", title="Retention %"),
                    ],
                )
                .properties(height=22 * heat["cohort"].nunique() + 40)
            )
            st.altair_chart(chart, use_container_width=True)

    st.divider()
    st.markdown("### Conversion funnel")
    fc1, fc2 = st.columns([2, 1])
    preset_name = fc1.selectbox("Funnel", list(_FUNNEL_PRESETS))
    ordered = fc2.toggle(
        "Ordered", value=True, help="Require steps to occur in sequence (first-touch)."
    )
    steps = _FUNNEL_PRESETS[preset_name]
    fn = _funnel(fkey, filters, steps, ordered)

    if fn.empty or fn["users"].iloc[0] == 0:
        st.caption("No users enter this funnel in the current scope.")
    else:
        top = fn["users"].iloc[0]
        fn = fn.assign(
            of_start=(fn["users"] / top * 100).round(1),
            step_conv=(fn["users"] / fn["users"].shift(1) * 100).round(1),
        )
        funnel_chart = (
            alt.Chart(fn)
            .mark_bar(color="#7C3AED")
            .encode(
                y=alt.Y("label:N", sort=fn["label"].tolist(), title=None),
                x=alt.X("users:Q", title="Users"),
                tooltip=[
                    alt.Tooltip("label:N", title="Step"),
                    alt.Tooltip("users:Q", title="Users", format=","),
                    alt.Tooltip("of_start:Q", title="% of start"),
                    alt.Tooltip("step_conv:Q", title="% of previous"),
                ],
            )
            .properties(height=55 * len(fn) + 20)
        )
        st.altair_chart(funnel_chart, use_container_width=True)
        st.dataframe(
            fn[["step", "label", "users", "of_start", "step_conv"]],
            use_container_width=True,
            hide_index=True,
            column_config={
                "of_start": st.column_config.NumberColumn("% of start", format="%.1f%%"),
                "step_conv": st.column_config.NumberColumn(
                    "% of previous", format="%.1f%%"
                ),
            },
        )
