-- Enriched user-level mart: lifetime + activity segment, plus AI usage, credit
-- consumption, and reliability signals joined from the domain facts. The
-- primary table for cohort slicing and per-user analysis.

with base as (
    select
        user_id,
        total_event_count,
        first_event_timestamp,
        last_event_timestamp
    from {{ ref('user_events') }}
),

ai as (
    select
        user_id,
        count(*)                  as ai_generation_count,
        sum(total_tokens)         as total_ai_tokens,
        avg(ai_latency_seconds)   as avg_ai_latency_seconds
    from {{ ref('fct_ai_generations') }}
    group by 1
),

credits as (
    select
        user_id,
        sum(credits_used)  as total_credits_used,
        count(*)           as credit_event_count
    from {{ ref('fct_credits') }}
    group by 1
),

exceptions as (
    select
        user_id,
        count(*) as exception_count
    from {{ ref('fct_exceptions') }}
    group by 1
)

select
    base.user_id,
    base.total_event_count,
    base.first_event_timestamp,
    cast(base.first_event_timestamp as date) as first_event_date,
    base.last_event_timestamp,
    cast(base.last_event_timestamp as date) as last_event_date,
    {{ dbt.datediff("base.first_event_timestamp", "base.last_event_timestamp", "day") }} as user_lifetime_days,

    coalesce(ai.ai_generation_count, 0)        as ai_generation_count,
    coalesce(ai.total_ai_tokens, 0)            as total_ai_tokens,
    ai.avg_ai_latency_seconds,
    coalesce(credits.total_credits_used, 0)    as total_credits_used,
    coalesce(credits.credit_event_count, 0)    as credit_event_count,
    coalesce(exceptions.exception_count, 0)    as exception_count,

    case
        when base.total_event_count >= 100 then 'high_activity'
        when base.total_event_count >= 20 then 'medium_activity'
        else 'low_activity'
    end as activity_segment,
    (coalesce(ai.ai_generation_count, 0) > 0) as is_ai_user
from base
left join ai          on base.user_id = ai.user_id
left join credits     on base.user_id = credits.user_id
left join exceptions  on base.user_id = exceptions.user_id
