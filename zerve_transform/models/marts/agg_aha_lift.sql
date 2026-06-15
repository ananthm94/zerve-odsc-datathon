-- The AHA leaderboard: for each candidate early action, how much more likely is a
-- user to CONVERT if they did it in week 1, versus the signup baseline -- and how
-- well do they retain. This is the analysis-ready table the dashboard reads and
-- sorts; all the cohort logic lives here so the UI stays a thin projection.
--
-- signal_type splits the story:
--   * 'product' = genuine build/ship actions Zerve can drive in onboarding.
--   * 'paywall' = monetization symptoms (hit a limit, saw a banner, clicked
--                 upgrade). These "predict" conversion only because they ARE
--                 intent, so they are reported separately, never as AHA drivers.

with mapped as (
    select
        user_id,
        event_timestamp,
        case
            when event_name in ('files_upload', 'quickstart_upload_file', 'agent_upload_files')
                then 'Upload own data'
            when event_name = 'agent_message' then 'Message the AI agent'
            when event_name in (
                'agent_block_created', 'agent_block_run', 'agent_refactor_block',
                'agent_tool_call_create_script_tool', 'agent_tool_call_create_block_tool',
                'agent_tool_call_apply_script_template_tool',
                'agent_tool_call_update_script_content_tool',
                'agent_tool_call_refactor_block_tool', 'agent_tool_call_run_block_tool'
            ) then 'Agent builds code'
            when event_name in ('run_block', 'run_all_blocks', 'run_from_block', 'run_upto_block')
                then 'Run a block'
            when event_name = 'canvas_create' then 'Create a canvas'
            when event_name in (
                'notebook_deployment_created', 'notebook_deployment_deployed',
                'hosted_apps_deploy', 'app_publish', 'api_deploy'
            ) then 'Deploy / publish'
            when event_name = 'agent_tool_call_analyze_attachment_tool'
                then 'Agent analyzes a file'
            when event_name = 'notebook_onboarding_tour_finished'
                then 'Finish onboarding tour'
            -- paywall / intent symptoms
            when event_name = 'clicked_upgrade' then 'Clicked upgrade'
            when event_name = 'claim_free_offer_clicked' then 'Claimed free offer'
            when event_name = 'ai_credit_banner_shown' then 'Saw credit banner'
            when event_name = 'credits_exceeded' then 'Credits exceeded'
            when event_name in ('credits_below_1', 'credits_below_2') then 'Credits running low'
            when event_name = 'addon_credits_used' then 'Used add-on credits'
        end as action
    from {{ ref('stg_events') }}
),

action_type as (
    select *,
        case
            when action in (
                'Clicked upgrade', 'Claimed free offer', 'Saw credit banner',
                'Credits exceeded', 'Credits running low', 'Used add-on credits'
            ) then 'paywall' else 'product'
        end as signal_type
    from mapped
    where action is not null
),

-- One row per (user, action) they performed within their week-1 AHA window,
-- carrying the user's outcome labels.
did_w1 as (
    select distinct
        m.action,
        m.signal_type,
        m.user_id,
        o.is_converter,
        o.outcome_observable,
        o.returned_after_w4
    from action_type m
    join {{ ref('dim_user_outcomes') }} o on m.user_id = o.user_id
    where m.event_timestamp < o.signup_ts + interval 7 day
),

baseline as (
    select
        avg(case when is_converter then 1.0 else 0.0 end) as baseline_rate
    from {{ ref('dim_user_outcomes') }}
)

select
    d.action,
    d.signal_type,
    count(*)                                                          as users_did_w1,
    count(*) filter (where d.is_converter)                           as converters_among_them,
    count(*) filter (where d.is_converter) * 1.0 / count(*)          as conversion_rate,
    b.baseline_rate,
    (count(*) filter (where d.is_converter) * 1.0 / count(*))
        / nullif(b.baseline_rate, 0)                                 as conversion_lift,
    count(*) filter (where d.outcome_observable and d.returned_after_w4)
                                                                     as retained_among_them,
    count(*) filter (where d.outcome_observable and d.returned_after_w4) * 1.0
        / nullif(count(*) filter (where d.outcome_observable), 0)    as retention_rate
from did_w1 d
cross join baseline b
group by d.action, d.signal_type, b.baseline_rate
order by conversion_lift desc
