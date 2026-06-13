-- Per-user event rollup: volume and lifespan, derived from the staged event grain.

with events as (
    select
        user_id,
        event_timestamp
    from {{ ref('stg_events') }}
    where user_id is not null
)

select
    user_id,
    count(*)              as total_event_count,
    min(event_timestamp)  as first_event_timestamp,
    max(event_timestamp)  as last_event_timestamp
from events
group by 1
