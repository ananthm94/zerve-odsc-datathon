-- Staged, typed, curated event grain.
--
-- The raw CSV (PostHog-style) has 83 columns with `$`-prefixed / dotted names and
-- sparse numeric fields stored as text. This model selects the analytically useful
-- columns, gives them clean snake_case names, and casts numerics with TRY_CAST
-- (sparse fields contain blanks/garbage that would fail a hard CAST).

with raw_events as (
    select * from {{ source('zerveevents', 'zerve_events_raw') }}
)

select
    -- identity & time
    person_id                                              as user_id,
    cast("timestamp" as timestamp)                         as event_timestamp,
    cast("timestamp" as date)                              as event_date,
    event                                                  as event_name,

    -- person properties (attributes of the user, sparsely populated)
    "person_properties.cloudProvider"                      as cloud_provider,
    "person_properties.purpose"                            as user_purpose,
    "person_properties.role"                               as user_role,
    "person_properties.source"                             as acquisition_source,
    "person_properties.work_type"                          as work_type,

    -- device & browser
    "properties.$browser"                                  as browser,
    "properties.$browser_language"                         as browser_language,
    "properties.$device_type"                              as device_type,
    "properties.$os"                                       as os,
    "properties.$os_version"                               as os_version,

    -- geography (GeoIP)
    "properties.$geoip_continent_name"                     as continent,
    "properties.$geoip_country_name"                       as country,
    "properties.$geoip_time_zone"                          as timezone,

    -- AI / ML (populated on $ai_generation events)
    "properties.$ai_model"                                 as ai_model,
    "properties.$ai_provider"                              as ai_provider,
    try_cast("properties.$ai_input_tokens" as bigint)      as ai_input_tokens,
    try_cast("properties.$ai_output_tokens" as bigint)     as ai_output_tokens,
    try_cast("properties.$ai_latency" as double)           as ai_latency_seconds,
    try_cast("properties.$ai_tool_call_count" as integer)  as ai_tool_call_count,
    "properties.$ai_tools_called"                          as ai_tools_called,

    -- credits / billing
    try_cast("properties.credit_amount" as double)         as credit_amount,
    try_cast("properties.credits_used" as double)          as credits_used,
    try_cast("properties.credits_awarded" as double)       as credits_awarded,
    try_cast("properties.credits_remaining" as double)     as credits_remaining,
    try_cast("properties.total_credits" as double)         as total_credits,
    try_cast("properties.total_addon_credits" as double)   as total_addon_credits,

    -- product context
    "properties.canvas_id"                                 as canvas_id,
    "properties.tool_name"                                 as tool_name,
    "properties.$event_type"                               as interaction_type,

    -- engagement / performance
    try_cast("properties.$prev_pageview_duration" as double) as prev_pageview_duration_ms,
    try_cast("properties.$web_vitals_CLS_value" as double)   as web_vitals_cls,
    try_cast("properties.$web_vitals_FCP_value" as double)   as web_vitals_fcp,
    try_cast("properties.$web_vitals_INP_value" as double)   as web_vitals_inp,
    try_cast("properties.$web_vitals_LCP_value" as double)   as web_vitals_lcp,

    -- acquisition
    "properties.utm_source"                                as utm_source,
    "properties.utm_medium"                                as utm_medium,
    "properties.utm_campaign"                              as utm_campaign,

    current_timestamp                                      as updated_at
from raw_events
where person_id is not null
