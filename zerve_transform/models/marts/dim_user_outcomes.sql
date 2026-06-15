-- One row per signup user, labeled with the monetization & retention OUTCOME.
-- This is the join target for the AHA-moment analysis: every behavioral feature
-- table (fct_activation_milestones, agg_aha_lift) attaches to these labels.
--
-- Definitions (locked with stakeholders):
--   * signup        = first sign_up / new_user_created event.
--   * converted     = ever fired upgrade_subscription / subscription_upgraded
--                     (committed-to-pay signal; addon credit top-ups excluded).
--   * activation    = did at least one CORE product action (canvas / block /
--                     agent / file upload) -- i.e. engaged beyond auth, identity,
--                     pageviews and the onboarding tour.
--   * churn (W4)    = activated in week 1 but did NOT return on/after signup+28d.
--                     Only meaningful for users observable >= 35 days, so the
--                     non-observable tail is held out as `too_new`.
--
-- Conversion is fast in this product (median 0 days, ~85% within 7 days), so the
-- AHA window is week 1; week-1 behavior is what fct_activation_milestones captures.

with events as (
    select user_id, event_name, event_timestamp, event_date
    from {{ ref('stg_events') }}
),

bounds as (
    select max(event_date) as max_date from events
),

-- A CORE product action = real engagement with the canvas/agent/blocks/files,
-- as opposed to auth, identity ($-prefixed), pageviews, or onboarding noise.
tagged as (
    select
        *,
        (
            event_name in (
                'canvas_create', 'canvas_open', 'canvas_clone', 'block_create',
                'run_block', 'run_all_blocks', 'run_from_block', 'run_upto_block',
                'agent_message', 'agent_new_chat', 'agent_start_from_prompt',
                'files_upload', 'quickstart_upload_file', 'agent_upload_files',
                'notebook_import', 'app_create', 'app_publish'
            )
            or event_name like 'agent_tool_call_%'
            or event_name like '%deployment%'
        ) as is_core_action
    from events
),

signup as (
    select
        user_id,
        min(event_timestamp) as signup_ts,
        cast(min(event_timestamp) as date) as signup_date
    from events
    where event_name in ('sign_up', 'new_user_created')
    group by 1
),

conv as (
    select
        user_id,
        min(event_timestamp) as first_upgrade_ts
    from events
    where event_name in ('upgrade_subscription', 'subscription_upgraded')
    group by 1
),

per_user as (
    select
        s.user_id,
        s.signup_ts,
        s.signup_date,
        c.first_upgrade_ts,
        max(t.event_date) as last_active_date,
        -- activation within the first 7 days (the AHA window)
        max(case
                when t.is_core_action
                     and t.event_timestamp < s.signup_ts + interval 7 day
                then 1 else 0 end) as active_week1_flag,
        max(case when t.is_core_action then 1 else 0 end) as activated_flag,
        -- returned for any activity on/after signup + 28 days
        max(case
                when t.event_timestamp >= s.signup_ts + interval 28 day
                then 1 else 0 end) as returned_after_w4_flag
    from signup s
    join tagged t on s.user_id = t.user_id
    left join conv c on s.user_id = c.user_id
    group by 1, 2, 3, 4
)

select
    p.user_id,
    p.signup_ts,
    p.signup_date,
    (p.first_upgrade_ts is not null)                              as is_converter,
    p.first_upgrade_ts,
    case when p.first_upgrade_ts is not null
         then {{ dbt.datediff("p.signup_ts", "p.first_upgrade_ts", "day") }}
    end                                                          as days_to_convert,
    (
        p.first_upgrade_ts is not null
        and p.first_upgrade_ts < p.signup_ts + interval 7 day
    )                                                            as converted_within_7d,
    (p.active_week1_flag = 1)                                    as active_week1,
    (p.activated_flag = 1)                                       as activated,
    (p.returned_after_w4_flag = 1)                               as returned_after_w4,
    (p.signup_date <= b.max_date - 35)                           as outcome_observable,
    p.last_active_date,
    (b.max_date - p.last_active_date)                            as recency_days,
    -- W4 churn: observable, never paid, activated in week 1, but never returned.
    (
        p.signup_date <= b.max_date - 35
        and p.first_upgrade_ts is null
        and p.active_week1_flag = 1
        and p.returned_after_w4_flag = 0
    )                                                            as is_churned_w4,
    case
        when p.first_upgrade_ts is not null then 'converted'
        when p.signup_date > b.max_date - 35 then 'too_new'
        when p.returned_after_w4_flag = 1 then 'retained_free'
        when p.active_week1_flag = 1 then 'churned_w4'
        else 'never_activated'
    end                                                          as lifecycle_stage
from per_user p
cross join bounds b
