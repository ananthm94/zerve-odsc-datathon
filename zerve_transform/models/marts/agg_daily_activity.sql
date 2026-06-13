-- Daily activity rollup: one row per calendar day with active users and an
-- event-type breakdown. Powers DAU and daily trend metrics.

with events as (
    select * from {{ ref('stg_events') }}
)

select
    event_date,
    count(distinct user_id)                                                  as active_users,
    count(*)                                                                 as total_events,
    count(*) filter (where event_name = '$ai_generation')                    as ai_generations,
    count(*) filter (where event_name = '$exception')                        as exceptions,
    count(*) filter (where event_name = '$pageview')                         as pageviews,
    count(*) filter (where event_name in ('credits_used', 'addon_credits_used')) as credit_events
from events
group by 1
