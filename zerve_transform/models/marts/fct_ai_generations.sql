-- One row per $ai_generation event: token usage, latency, model and provider.
-- AI fields are only populated on these events.

with ai_events as (
    select *
    from {{ ref('stg_events') }}
    where event_name = '$ai_generation'
)

select
    user_id,
    event_timestamp,
    event_date,
    ai_model,
    ai_provider,
    ai_input_tokens,
    ai_output_tokens,
    coalesce(ai_input_tokens, 0) + coalesce(ai_output_tokens, 0) as total_tokens,
    ai_latency_seconds,
    ai_tool_call_count,
    ai_tools_called
from ai_events
