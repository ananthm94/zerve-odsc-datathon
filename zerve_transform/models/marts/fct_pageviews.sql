-- One row per $pageview event: device/geo context, prior-pageview engagement,
-- and Core Web Vitals where captured.

with pageviews as (
    select *
    from {{ ref('stg_events') }}
    where event_name = '$pageview'
)

select
    user_id,
    event_timestamp,
    event_date,
    browser,
    device_type,
    os,
    country,
    continent,
    prev_pageview_duration_ms,
    web_vitals_cls,
    web_vitals_fcp,
    web_vitals_inp,
    web_vitals_lcp
from pageviews
