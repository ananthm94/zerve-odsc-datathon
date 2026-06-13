-- One row per user: stable attributes (person properties) plus the modal
-- device/geography context across all of the user's events. Modal values use
-- DuckDB's mode() which ignores nulls, so sparse fields resolve to the user's
-- most common observed value.

with events as (
    select * from {{ ref('stg_events') }}
)

select
    user_id,

    -- person attributes
    mode(user_role)            as user_role,
    mode(user_purpose)         as user_purpose,
    mode(acquisition_source)   as acquisition_source,
    mode(work_type)            as work_type,
    mode(cloud_provider)       as cloud_provider,

    -- modal geography
    mode(country)              as country,
    mode(continent)            as continent,
    mode(timezone)             as timezone,

    -- modal device context
    mode(device_type)          as device_type,
    mode(os)                   as os,
    mode(browser)              as browser,

    -- lifespan
    min(event_timestamp)              as first_seen_at,
    max(event_timestamp)              as last_seen_at,
    cast(min(event_timestamp) as date) as first_seen_date,
    count(*)                          as total_events
from events
group by 1
