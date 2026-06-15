-- The AHA feature table: one row per signup user capturing WHEN they first hit
-- each candidate activation milestone, whether they hit it within the week-1 AHA
-- window, and how broad/deep their first week was. Joins to dim_user_outcomes for
-- the pay/churn labels, so a single table answers "which early actions precede
-- conversion vs churn".
--
-- Milestones are GENUINE product actions (build/ship), kept distinct from the
-- paywall/intent symptoms (clicked_upgrade, credits_exceeded, banners) which are
-- analysed separately in agg_aha_lift. The one exception is first_credit_exhaust,
-- retained here because running out of credits is a real usage-depth milestone.

with outcomes as (
    select user_id, signup_ts from {{ ref('dim_user_outcomes') }}
),

events as (
    select e.user_id, e.event_name, e.event_timestamp, e.event_date
    from {{ ref('stg_events') }} e
    join outcomes o on e.user_id = o.user_id
),

-- First-occurrence timestamp of each milestone, per user.
firsts as (
    select
        user_id,
        min(event_timestamp) filter (where
            event_name in ('files_upload', 'quickstart_upload_file', 'agent_upload_files')
        ) as first_upload_ts,
        min(event_timestamp) filter (where event_name = 'agent_message') as first_agent_message_ts,
        min(event_timestamp) filter (where
            event_name in (
                'agent_block_created', 'agent_block_run', 'agent_refactor_block',
                'agent_tool_call_create_script_tool', 'agent_tool_call_create_block_tool',
                'agent_tool_call_apply_script_template_tool',
                'agent_tool_call_update_script_content_tool',
                'agent_tool_call_refactor_block_tool', 'agent_tool_call_run_block_tool'
            )
        ) as first_agent_build_ts,
        min(event_timestamp) filter (where
            event_name in ('run_block', 'run_all_blocks', 'run_from_block', 'run_upto_block')
        ) as first_block_run_ts,
        min(event_timestamp) filter (where event_name = 'canvas_create') as first_canvas_create_ts,
        min(event_timestamp) filter (where
            event_name in (
                'notebook_deployment_created', 'notebook_deployment_deployed',
                'hosted_apps_deploy', 'app_publish', 'api_deploy'
            )
        ) as first_deploy_ts,
        min(event_timestamp) filter (where
            event_name in ('credits_below_1', 'credits_exceeded')
        ) as first_credit_exhaust_ts
    from events
    group by 1
),

-- Week-1 breadth/depth signals (first 7 days after signup).
week1 as (
    select
        e.user_id,
        count(distinct e.event_date)  as w1_active_days,
        count(*)                      as w1_event_count,
        count(distinct e.event_name)  as w1_distinct_features
    from events e
    join outcomes o on e.user_id = o.user_id
    where e.event_timestamp < o.signup_ts + interval 7 day
    group by 1
)

select
    o.user_id,
    f.first_upload_ts,
    f.first_agent_message_ts,
    f.first_agent_build_ts,
    f.first_block_run_ts,
    f.first_canvas_create_ts,
    f.first_deploy_ts,
    f.first_credit_exhaust_ts,

    {% set milestones = [
        'upload', 'agent_message', 'agent_build', 'block_run',
        'canvas_create', 'deploy', 'credit_exhaust'
    ] %}
    {% for m in milestones %}
    (f.first_{{ m }}_ts is not null
        and f.first_{{ m }}_ts < o.signup_ts + interval 7 day) as did_{{ m }}_w1,
    {% endfor %}

    coalesce(w.w1_active_days, 0)        as w1_active_days,
    coalesce(w.w1_event_count, 0)        as w1_event_count,
    coalesce(w.w1_distinct_features, 0)  as w1_distinct_features
from outcomes o
left join firsts f on o.user_id = f.user_id
left join week1  w on o.user_id = w.user_id
