-- Credit-consumption events. Captures per-event credit usage and the
-- zero-balance trigger. Note: credits_remaining is ONLY populated when the
-- balance hits zero (all non-null values are 0.0) -- it is a depletion flag,
-- not a running balance.

with credit_events as (
    select *
    from {{ ref('stg_events') }}
    where event_name in ('credits_used', 'addon_credits_used')
       or credits_used is not null
       or credit_amount is not null
)

select
    user_id,
    event_timestamp,
    event_date,
    event_name,
    credit_amount,
    credits_used,
    credits_awarded,
    total_credits,
    total_addon_credits,
    (credits_remaining is not null) as hit_zero_balance
from credit_events
