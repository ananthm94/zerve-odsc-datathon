-- Event-grain feature tagging.
--
-- The raw stream has 227 distinct event_names. This view maps each event to a
-- product feature area so usage can be analysed by feature without enumerating
-- every event. It is the single source of truth for that categorization: the
-- dashboard reads it at event grain (so user/date filters and distinct-user
-- reach are exact) and `agg_feature_usage` rolls it up to a daily mart.
--
-- Materialized as a view: it is a thin projection of stg_events, so there is no
-- need to bake another 3.5M-row copy into the database file.

{{ config(materialized='view') }}

with events as (
    select user_id, event_date, event_name
    from {{ ref('stg_events') }}
)

select
    user_id,
    event_date,
    event_name,
    case
        when event_name = '$ai_generation' then 'AI Generation'
        when event_name = '$exception' then 'Errors'
        when event_name in ('$pageview', '$pageleave', '$web_vitals')
            then 'Web / Pageviews'
        when event_name like 'agent_%' then 'Agent'
        when event_name like '%block%' then 'Block Execution'
        when event_name like '%deployment%' then 'Deployments'
        when event_name like 'credits_%' or event_name = 'addon_credits_used'
            then 'Credits'
        when event_name like '%onboarding%' then 'Onboarding'
        when event_name like 'sign_%'
            or event_name in ('login', 'logout', 'new_user_created')
            then 'Auth'
        when event_name like '$%' then 'System / Identity'
        else 'Other'
    end as feature_category
from events
