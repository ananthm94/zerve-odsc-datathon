-- Daily feature-usage rollup: one row per (day, feature area) with event volume
-- and distinct-user reach. Reusable aggregate over stg_feature_events; powers
-- the dashboard's Feature Usage trend and is directly queryable by the agent.

with tagged as (
    select * from {{ ref('stg_feature_events') }}
)

select
    event_date,
    feature_category,
    count(*)                  as events,
    count(distinct user_id)   as distinct_users
from tagged
group by 1, 2
