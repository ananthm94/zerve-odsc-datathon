-- One row per $exception event, with the device/geo context for reliability analysis.

with exceptions as (
    select *
    from {{ ref('stg_events') }}
    where event_name = '$exception'
)

select
    user_id,
    event_timestamp,
    event_date,
    browser,
    device_type,
    os,
    os_version,
    country
from exceptions
