# Zerve Analytics Agent

An **AI-native data science platform** built on a dbt-modeled semantic layer over
the Zerve product-analytics dataset (PostHog-style events). Ask a business
question in plain English and the agent autonomously **plans**, **queries**,
**reflects**, and **synthesizes** a cited report — backed by read-only SQL,
a RAG semantic layer, and a statistics module for cohort experiments.

The long-term goal (inspired by a Perplexity data-team JD) is an agent that does
autonomous end-to-end analysis against an AI-readable semantic layer, runs
experiment analysis, and self-heals data-quality issues.

---

## Status

Sequential, versioned build (`v0.1` → `v0.8`):

| Version | Capability | Status |
|---|---|---|
| v0.1 | Repo hygiene (scrubbed secrets, untracked venv) | ✅ Done |
| v0.2 | Multi-provider LLM (`analytics_agent/llm.py`) | ✅ Done |
| v0.3 | DuckDB backend (default) + BigQuery kept as backup | ✅ Done |
| v0.4 | Full dbt semantic layer (marts, semantic models, metrics) | ✅ Done |
| v0.5 | Autonomous plan → query-loop → reflect → synthesize agent | ✅ Done |
| v0.6 | Experiment / cohort analysis (`analytics_agent/experiments.py`) | ✅ Done |
| v0.7 | Productized multi-tab UI: live agent-state streaming + dashboard + experiments | ✅ Done |

> **Self-healing data quality** was originally planned as a phase but is **not
> pursued** — the dbt build is green and the marts have no quality issues in this
> dataset. It's documented as a [future possibility](#future-possibilities)
> rather than built speculatively.

**Dataset:** ~3.5M events × 83 columns (PostHog-style), ~17.5k users, spanning
2025-09 → 2026-04.

---

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Data Layer                                                    │
│    raw_data/zerve_events.csv  (local, default)                 │
│    BigQuery zerve_events_raw  (optional backup backend)        │
└─────────────────────────┬──────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  dbt Transformations + Semantic Layer  (zerve_transform/)      │
│    staging/  stg_events (40 typed/curated cols), user_events   │
│    marts/    dim_users, fct_ai_generations, fct_credits,       │
│              fct_pageviews, fct_exceptions, agg_daily_activity, │
│              user_summary  +  metricflow_time_spine             │
│    semantic_models + metrics  (MetricFlow)                     │
│    → builds local DuckDB file: zerve_transform/zerve.duckdb    │
└─────────────────────────┬──────────────────────────────────────┘
                          │  manifest.json + schema.yml
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  Embedding Pipeline  (embedding_script.py)                     │
│    dbt models + metrics + semantic models → Documents          │
│    OpenAI text-embedding-3-small → Qdrant (29 docs)            │
│    collection: dbt_semantic_dictionary                         │
└─────────────────────────┬──────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  Autonomous Analytics Agent  (analytics_agent/)                │
│    LangGraph: plan → [run_query]* → reflect → synthesize       │
│    multi-provider LLM · RAG retrieval · read-only SQL guards   │
│    + experiments.py  (observational cohort statistics)         │
└─────────────────────────┬──────────────────────────────────────┘
                          ▼
┌──────────────────────────────────────────────────────────────┐
│  Streamlit UI  (streamlit_app.py)                              │
│    question → synthesized report + result preview + SQL        │
└──────────────────────────────────────────────────────────────┘
```

### Backends

DuckDB is the **default** backend (local, free, reproducible). dbt builds a local
`zerve.duckdb` file and the agent queries it read-only. BigQuery is retained as a
selectable backup via the `BACKEND` env var; when active, queries also get a
dry-run byte estimate as a cost guard. Both backends sit behind a single
`QueryRunner` protocol (`analytics_agent/runners.py`), so the agent is
backend-agnostic.

### Multi-provider LLM

The chat model is provider-configurable across **OpenAI, Claude (Anthropic),
Kimi (Moonshot), and OpenRouter** (`analytics_agent/llm.py`, selected by
`LLM_PROVIDER`). Embeddings deliberately stay on OpenAI so the existing Qdrant
collection remains valid.

---

## dbt Models

| Model | Layer | Grain | Description |
|---|---|---|---|
| `stg_events` | staging | event | 40 typed/curated columns from the raw event stream |
| `user_events` | staging | user | Per-user event counts and first/last timestamps |
| `dim_users` | marts | user | Stable person attributes + modal device/geo |
| `fct_ai_generations` | marts | $ai_generation | Token usage, latency, model, provider |
| `fct_credits` | marts | credit event | Credit consumption events |
| `fct_pageviews` | marts | $pageview | Pageviews with device/geo + Core Web Vitals |
| `fct_exceptions` | marts | $exception | Exceptions with device/geo for reliability |
| `agg_daily_activity` | marts | day | Daily active users + event-type breakdown |
| `user_summary` | marts | user | Enriched per-user mart (AI/credit/reliability + activity segment) |
| `stg_feature_events` | staging | event | Event tagged with a product `feature_category` (view; single source of truth for feature grouping) |
| `agg_feature_usage` | marts | day × feature | Daily feature volume + distinct-user reach |
| `fct_user_activity_weekly` | marts | user × week | User's active weeks + cohort week + offset (powers retention) |
| `metricflow_time_spine` | marts | day | Daily spine required by the semantic layer |

**Semantic layer (MetricFlow):** 5 semantic models (`events`, `ai_generations`,
`credits`, `exceptions`, `user_summary`) expose metrics including
`daily_active_users`, `total_ai_tokens`, `avg_ai_latency`, `total_credits_used`,
`total_exceptions`, `total_users`, and the ratio metric `ai_adoption_rate`.

The agent's queryable allow-list is **derived from the dbt manifest** at runtime
(`config.py`), so new marts are automatically queryable; the time-spine helper is
excluded.

---

## Autonomous Agent

`analytics_agent/graph.py` is a [LangGraph](https://github.com/langchain-ai/langgraph)
state machine that replaces the old linear "one question → one query → one answer"
path with an autonomous loop:

```
question
   │
   ▼
[plan_analysis]     — LLM → hypotheses + ordered, self-contained sub-questions (JSON)
   │
   ▼
[run_query] ◄────┐  — per sub-question: retrieve → generate_sql → validate
   │             │    → (dry-run if BigQuery) → execute → record a finding
   │             │    Bounded SQL self-repair: a validation/DB error is fed back
   │             │    to the LLM for one corrective retry (_MAX_SQL_ATTEMPTS=2).
   ▼             │
[reflect]  ──────┘  — LLM decides "sufficient" or enqueues bounded follow-ups
   │
   ▼
[synthesize_report] — cited [Q1]… report with explicit uncertainty;
   │                  never invents numbers not present in results
   ▼
report (+ rows / sql / context for the UI)
```

- **Bounded:** total sub-queries (planned + follow-ups) capped by
  `max_sub_queries` (default 8, env `MAX_SUB_QUERIES`).
- **Resilient:** a failed sub-question is captured as a finding and never aborts
  the whole run.
- **Self-repairing SQL:** bounded retry lifted live success from 2/8 → 6/8 on
  "What's driving AI credit consumption?".
- **Backward-compatible:** still surfaces the last successful query's
  `answer/sql/rows/bytes_processed/context_documents` for the single-box UI.

### Read-only SQL guardrails (`analytics_agent/sql.py`)

Every query — agent-generated or internal — passes `validate_sql`:
`SELECT`/`WITH` only, no mutation keywords, must reference only allow-listed
fully-qualified tables. A default `LIMIT` is injected for non-aggregate queries.
DuckDB connections are opened read-only per query as a second line of defense.

### Key files

| File | Purpose |
|---|---|
| [`analytics_agent/graph.py`](analytics_agent/graph.py) | Autonomous LangGraph agent and node functions |
| [`analytics_agent/config.py`](analytics_agent/config.py) | `AgentConfig`, env loading, manifest-derived allow-list |
| [`analytics_agent/llm.py`](analytics_agent/llm.py) | Provider-agnostic chat model factory |
| [`analytics_agent/runners.py`](analytics_agent/runners.py) | DuckDB / BigQuery `QueryRunner` backends |
| [`analytics_agent/retriever.py`](analytics_agent/retriever.py) | Qdrant RAG retriever (OpenAI embeddings) |
| [`analytics_agent/bigquery_client.py`](analytics_agent/bigquery_client.py) | BigQuery dry-run + execute |
| [`analytics_agent/sql.py`](analytics_agent/sql.py) | SQL validation, fence stripping, default LIMIT |
| [`analytics_agent/experiments.py`](analytics_agent/experiments.py) | Observational cohort statistics + narrative |

---

## Experiment / Cohort Analysis

The dataset has **no experiment-assignment column**, so there are no true
randomized A/B tests. `analytics_agent/experiments.py` instead compares
**observational cohorts** (slices by role, cloud provider, activity segment, or a
time-based pre/post split) and is explicit that results are **quasi-experimental**
(differences may be confounded by self-selection).

- **Continuous metrics** (tokens, latency, events/user): Welch's t-test +
  Mann-Whitney U, with Cohen's d and a Welch CI for the mean difference.
- **Proportions / rates** (AI-adoption, conversion): two-proportion z-test +
  chi-square, with the risk difference and its Wald CI.

Cohort samplers build **trusted, internally-constructed SQL** (not LLM-generated)
but still run through `validate_sql` and allow-listed FQTNs, with `_ident`/
`_literal` guards against injection. `narrate_experiment` then drafts an LLM
ship / no-ship recommendation that flags observational confounding, CIs that
cross zero, sample imbalance, and multiple-comparison risk — judging on effect
size and CI, not just the p-value.

---

## Productized UI

`streamlit_app.py` is a four-tab app over the agent and the warehouse:

- **💬 Ask** — the autonomous agent with its reasoning **streamed live**. Instead
  of a spinner, the user watches each stage land as it completes: the plan
  (hypotheses + sub-questions), each sub-question's SQL and result rows (failed
  sub-questions shown inline, not hidden), the reflection, then the synthesized
  cited report. Powered by `graph.stream_analytics_events`, which yields one
  event per LangGraph node (`STAGE_PLAN` / `STAGE_QUERY` / `STAGE_REFLECT` /
  `STAGE_SYNTHESIZE`); `run_analytics_question` remains for non-streaming callers.
- **📊 Dashboard** — headline KPIs (users, events, AI generations, tokens, credits,
  exceptions, AI-adoption rate) plus charts: daily active users, daily event mix,
  AI tokens by model and by provider, users by role, exceptions by OS, and the
  activity-segment table. Backed by `analytics_agent/dashboard.py` — author-written,
  read-only canned queries over the marts, cached with `st.cache_data`. Adds:
  - **Global filters** (sidebar): date range + user-cohort multiselects (role,
    activity segment, geo, device, OS). User filters resolve to one
    `user_id IN (matching users)` subquery so every chart shows the same cohort;
    daily metrics recompute from `stg_events` when a user filter is active.
  - **Feature usage** — events, distinct-user reach, and adoption % per product
    feature area, plus a per-feature usage trend (over `stg_feature_events` /
    `agg_feature_usage`).
  - **Retention** — weekly cohort retention curve, W1/W4/Wn headline numbers, and
    an Altair cohort triangle (over `fct_user_activity_weekly`).
  - **Funnel** — ordered first-touch (or cumulative) conversion funnel with
    presets (Activation, Agent workflow); honours the global filters.
- **🧪 Experiments** — pick a cohort dimension (role, segment, provider, …), two
  cohorts, and a continuous metric or a success definition; runs the v0.6
  statistics and renders the ship/no-ship narrative with the full statistical
  detail. Explicitly labels results as quasi-experimental.
- **ℹ️ About** — architecture and status summary.

```bash
streamlit run streamlit_app.py
```

---

## Setup

### Prerequisites

- **Python 3.9** (`.venv` is the working env). Run dbt as `.venv/bin/dbt`.
- An OpenAI API key (required for embeddings; also the default chat provider).
- A Qdrant Cloud cluster (or local Qdrant) for the semantic-layer vector store.
- *(Optional)* A Google Cloud project with BigQuery if using the BigQuery backend.

### Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Environment variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `BACKEND` | `duckdb` (default) or `bigquery` |
| `OPENAI_API_KEY` | OpenAI key (embeddings + default chat) |
| `OPENAI_EMBEDDING_MODEL` | Embedding model (default `text-embedding-3-small`) |
| `LLM_PROVIDER` | `openai` (default), `claude`, `kimi`, or `openrouter` |
| `LLM_MODEL` / `OPENAI_CHAT_MODEL` | Chat model id (default `gpt-4.1-mini`) |
| `LLM_API_KEY` / `LLM_BASE_URL` | Optional overrides for the chat provider |
| `MAX_SUB_QUERIES` | Agent sub-query budget (default 8) |
| `QDRANT_URL`, `QDRANT_API_KEY`, `QDRANT_COLLECTION_NAME` | Qdrant vector store |
| `DUCKDB_PATH`, `DUCKDB_SCHEMA` | DuckDB file + schema (defaults provided) |
| `BIGQUERY_PROJECT`, `BIGQUERY_DATASET`, `BIGQUERY_LOCATION` | Only when `BACKEND=bigquery` |

### Build the warehouse (DuckDB)

```bash
cd zerve_transform
PYTHONWARNINGS=ignore ../.venv/bin/dbt build --target duckdb --profiles-dir .
```

### Build the vector store

```bash
python embedding_script.py --force-recreate     # rebuild from current dbt manifest
python embedding_script.py --local-smoke-test    # offline smoke test (no OpenAI/Qdrant)
python embedding_script.py --query "AI token usage by model"   # retrieval check
```

### Launch the UI

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501) and ask, e.g.:

- *What's driving AI credit consumption?*
- *How does AI adoption differ across user roles?*
- *Which segments have the highest exception rates?*

### Run tests

```bash
python -m unittest discover -s tests
```

Suites: `test_agent.py` (plan → query×N → reflect → synthesize, citations, budget
cap, self-repair), `test_experiments.py` (stats vs hand-computed refs, SQL
construction, injection guard, live DuckDB), `test_sql_validation.py`,
`test_backend.py`, `test_llm.py`.

---

## Project structure

```
.
├── analytics_agent/          # Autonomous LangGraph analytics agent
│   ├── graph.py              # plan → query-loop → reflect → synthesize
│   ├── config.py             # Config, env loading, manifest-derived allow-list
│   ├── llm.py                # Multi-provider chat model factory
│   ├── runners.py            # DuckDB / BigQuery query backends
│   ├── retriever.py          # Qdrant RAG retriever
│   ├── bigquery_client.py    # BigQuery dry-run + execute
│   ├── sql.py                # Read-only SQL validation
│   ├── experiments.py        # Observational cohort statistics + narrative
│   └── dashboard.py          # Canned read-only queries for the dashboard tab
├── zerve_transform/          # dbt project (semantic layer + DuckDB build)
│   └── models/
│       ├── staging/          # stg_events, user_events
│       └── marts/            # dims, facts, aggregates, user_summary, semantic models
├── embedding_script.py       # Embed dbt metadata into Qdrant
├── streamlit_app.py          # Q&A UI
├── tests/                    # unittest suites
├── requirements.txt
└── .env.example
```

---

## Future possibilities

These are **deliberately not built** for this dataset, but are the natural next
steps and are sketched here so the direction is clear.

### Self-healing data quality

Not pursued because the current dbt build is green and the marts show no quality
issues (no failing tests, no schema drift, sensible distributions). If the
pipeline grew or ingested live data, it would be added as a monitoring +
remediation loop:

1. **Detect** — expand dbt tests (freshness, `accepted_range`, `not_null_proportion`,
   row-count and distribution anomaly checks via `dbt_utils` / a `dbt source freshness`
   run) and parse `run_results.json` for failures.
2. **Triage** — an LLM node reads the failing test, the model SQL, and a sample of
   offending rows, then classifies the cause (upstream nulls, a type change, a
   broken join, an outlier source row).
3. **Propose / heal** — for low-risk fixes (coalesce a new null, widen a type,
   quarantine bad rows into a `_rejected` table) it drafts a dbt model patch and
   opens it for review; for ambiguous cases it raises an alert with its diagnosis.
   Fixes go through review/PR, never silent auto-apply to the warehouse.

This reuses the existing pieces — the multi-provider LLM, the read-only guardrails,
and the manifest-driven model registry — so it slots in as additional LangGraph
nodes rather than a rewrite.

### Dashboard expansion

Global filters, feature usage, weekly retention, and a conversion funnel are now
**built** (see the Dashboard section above). Remaining stretch ideas: an **AI
cost** view once a token→price mapping is available, explicit **geographic**
breakdowns (`country`/`continent` are already in the marts and filterable),
custom user-defined funnels in the UI, and pinning an agent answer's result table
as a saved dashboard card.
