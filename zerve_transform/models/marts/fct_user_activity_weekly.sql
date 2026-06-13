-- One row per (user, active ISO week). The building block for cohort retention:
-- a user's cohort is their first active week, and week_offset is how many weeks
-- after that cohort the activity occurred. Retention at offset N is then the
-- share of a cohort with a row at that offset.

with weekly as (
    select distinct
        user_id,
        date_trunc('week', event_date) as activity_week
    from {{ ref('stg_events') }}
),

first_week as (
    select
        user_id,
        min(activity_week) as cohort_week
    from weekly
    group by 1
)

select
    w.user_id,
    f.cohort_week,
    w.activity_week,
    cast(date_diff('week', f.cohort_week, w.activity_week) as integer) as week_offset
from weekly w
join first_week f on w.user_id = f.user_id
