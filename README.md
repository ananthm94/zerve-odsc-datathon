# Zerve Analytics Agent

An AI-native analytics application for the Zerve product-analytics dataset. It
combines a dbt semantic layer, a local DuckDB warehouse, retrieval over dbt
metadata, a LangGraph analysis agent, statistical cohort comparisons, and a
Streamlit UI.

The core workflow is simple from the user's point of view: ask a business
question in plain English, and the agent plans an analysis, retrieves relevant
semantic-layer context, writes guarded read-only SQL, executes bounded queries,
reflects on the evidence, and synthesizes a cited answer.

Dataset summary: approximately 3.5M PostHog-style product events, 83 raw
columns, about 17.5k users, spanning 2025-09 through 2026-04.

## What Is In This Repo

- A dbt project in `zerve_transform/` that stages raw product events and builds
  analysis-ready marts into DuckDB.
- A Python package in `analytics_agent/`, organized by feature so it maps onto
  the UI tabs: `core/` (shared config, SQL validation, query runners, retrieval,
  contracts, LLM access), `ask/` (the LangGraph agent and trust loop), `library/`
  (saved-report memory), `dashboard/` (dashboard queries and rendering),
  `experiments/` (cohort statistics), and `ui/` (one Streamlit module per tab).
- `embedding_script.py`, which turns dbt models, semantic models, and metrics
  into Qdrant documents for RAG.
- `streamlit_app.py`, a thin entrypoint that wires together a five-tab UI — Ask,
  Library, Dashboard, Experiments, and About — each backed by a module in
  `analytics_agent/ui/`.
- `tests/`, a `unittest` suite covering the agent loop, SQL guardrails, backend
  selection, dashboard query construction, LLM configuration, and experiments.

## Current Status

| Version | Capability | Status |
|---|---|---|
| v0.1 | Repo hygiene and secret scrubbing | Done |
| v0.2 | Multi-provider LLM factory | Done |
| v0.3 | DuckDB default backend plus optional BigQuery backend | Done |
| v0.4 | dbt semantic layer with marts, semantic models, and metrics | Done |
| v0.5 | Autonomous plan -> query -> reflect -> synthesize agent | Done |
| v0.6 | Observational cohort experiment analysis | Done |
| v0.7 | Streamlit app with live agent streaming, dashboard, and experiments | Done |
| v0.8 | Expanded dashboard: filters, feature usage, retention, funnel | Done |
| v0.9 | Agentic trust loop: intent routing, semantic grounding, report memory/Library, snapshot verification, dashboard generation, model-cost tracking | Done |

The **trust loop** (v0.9) wraps the autonomous agent in an orchestrator
(`analytics_agent/ask/graph.py`, `stream_agentic_events`): it routes the question to
an intent (definition, simple metric, dashboard request, or analysis), grounds it
in official dbt metric contracts via the semantic layer
(`analytics_agent/ask/semantic_layer.py`), recalls relevant approved reports
(`analytics_agent/library/memory.py`), runs the plan→query→reflect→synthesize loop for
open questions, and verifies headline numbers against trusted snapshots
(`analytics_agent/ask/snapshots.py`). Answers can be saved to the **Library** and
generated dashboard specs (`analytics_agent/dashboard/render.py`) approved and
persisted. Per-run model usage and estimated cost are tracked in
`analytics_agent/ask/model_router.py`, and `evals/run_evals.py` scores intent and
metric grounding against `evals/cases.yaml`. The UI ships a light violet theme
(`.streamlit/config.toml`).

Self-healing data quality is intentionally not implemented for this static
datathon dataset. The current dbt build is expected to be green, and the
self-healing idea is listed under future possibilities rather than built
speculatively.

## Architecture

```text
                       local CSV or optional BigQuery source
                                      |
                                      v
+------------------------------------------------------------------+
| dbt project: zerve_transform/                                    |
|                                                                  |
| staging models: stg_events, stg_feature_events, user_events      |
| mart models: dim_users, fct_* facts, agg_* rollups, user_summary |
| semantic layer: MetricFlow semantic_models + metrics             |
|                                                                  |
| output: zerve_transform/zerve.duckdb and target/manifest.json    |
+------------------------------------------------------------------+
                         |                         |
                         |                         v
                         |       embedding_script.py builds semantic docs
                         |       from dbt manifest + schema YAML
                         |                         |
                         v                         v
+----------------------------------+     +-------------------------+
| analytics_agent/core/runners.py  |     | Qdrant vector store     |
| DuckDBRunner or BigQueryRunner   |     | dbt_semantic_dictionary |
+----------------------------------+     +-------------------------+
                         |                         |
                         v                         v
+------------------------------------------------------------------+
| analytics_agent/ask/graph.py                                     |
| LangGraph state machine:                                         |
| plan_analysis -> run_query* -> reflect -> run_query* -> synthesize|
|                                                                  |
| Each query: retrieve context -> generate SQL -> validate SQL     |
| -> dry-run if BigQuery -> execute -> record cited finding        |
+------------------------------------------------------------------+
                         |
                         v
+------------------------------------------------------------------+
| streamlit_app.py + analytics_agent/ui/ (one module per tab)      |
| Ask · Library · Dashboard · Experiments · About                  |
+------------------------------------------------------------------+
```

### Data Layer

The raw event feed is modeled as a dbt source named `zerveevents.zerve_events_raw`.
For local DuckDB development, `zerve_transform/profiles.yml` and the source
metadata expect a CSV at `raw_data/zerve_events.csv` relative to the repo root.
For optional BigQuery execution, the same source name can point at
`BIGQUERY_PROJECT.BIGQUERY_DATASET.zerve_events_raw`.

The dbt build materializes models as tables so the app can query a static local
DuckDB file without repeatedly scanning the source CSV.

### dbt Transform And Semantic Layer

The dbt project lives in `zerve_transform/`.

| Model | Layer | Grain | Purpose |
|---|---|---|---|
| `stg_events` | staging | event | Typed, curated raw event feed with useful product analytics columns |
| `stg_feature_events` | staging | event | Adds `feature_category` for product-area usage analysis |
| `user_events` | staging | user | Per-user event counts and first/last event timestamps |
| `dim_users` | mart | user | Stable user attributes plus modal device, browser, OS, and geo |
| `fct_ai_generations` | mart | AI generation event | Tokens, latency, model, provider, and tool-call usage |
| `fct_credits` | mart | credit event | Credit consumption and zero-balance signals |
| `fct_pageviews` | mart | pageview event | Pageview context plus sparse Core Web Vitals |
| `fct_exceptions` | mart | exception event | Reliability events with device and geo context |
| `agg_daily_activity` | mart | day | Daily active users and event-type counts |
| `agg_feature_usage` | mart | day x feature | Daily feature volume and distinct-user reach |
| `fct_user_activity_weekly` | mart | user x week | Weekly active-user cohorts and retention offsets |
| `user_summary` | mart | user | Enriched user-level facts for cohorting and per-user analysis |
| `metricflow_time_spine` | mart | day | Daily time spine required by the semantic layer |

The semantic layer is declared in `zerve_transform/models/marts/schema.yml`.
It defines five MetricFlow semantic models:

- `events`
- `ai_generations`
- `credits`
- `exceptions`
- `user_summary`

Important metrics include:

- `daily_active_users`
- `total_users`
- `total_events`
- `avg_events_per_user`
- `total_ai_generations`
- `total_ai_tokens`
- `avg_ai_latency`
- `total_credits_used`
- `total_exceptions`
- `ai_users`
- `ai_adoption_rate`

`analytics_agent/core/config.py` derives the query allow-list from
`zerve_transform/target/manifest.json` when it exists. The MetricFlow time spine
is excluded from agent querying because it is an internal helper.

### Semantic Retrieval

`embedding_script.py` reads:

- dbt model metadata from `zerve_transform/target/manifest.json`
- semantic models and metrics from `zerve_transform/models/**/*.yml`

It converts them to LangChain `Document` objects and loads them into Qdrant under
`QDRANT_COLLECTION_NAME`, defaulting to `dbt_semantic_dictionary`. The default
embedding model is OpenAI `text-embedding-3-small`.

At runtime, `analytics_agent/core/retriever.py` connects to that existing collection
and retrieves relevant model, column, semantic-model, and metric context for
each sub-question before SQL generation.

### Query Backends

`analytics_agent/core/runners.py` exposes one `QueryRunner` protocol:

- `DuckDBRunner` opens `DUCKDB_PATH` read-only for each query. This is the
  default, local, reproducible path.
- `BigQueryRunner` in `analytics_agent/core/bigquery_client.py` supports optional
  BigQuery execution and dry-run byte estimates.

The rest of the app builds against the protocol rather than a specific
warehouse. Select the backend with `BACKEND=duckdb` or `BACKEND=bigquery`.

### SQL Safety

`analytics_agent/core/sql.py` is the shared guardrail layer.

It:

- strips markdown SQL fences,
- requires queries to start with `SELECT` or `WITH`,
- blocks mutation keywords such as `insert`, `update`, `delete`, `drop`, and
  `merge`,
- requires at least one referenced table,
- rejects tables outside the manifest-derived allow-list,
- requires fully qualified backend-specific table identifiers,
- injects a default `LIMIT` for non-aggregate detail queries.

DuckDB connections are also opened read-only, so local execution has a second
defense even if validation is bypassed.

### Autonomous Agent

`analytics_agent/ask/graph.py` defines the LangGraph state machine.

```text
question
  |
  v
plan_analysis
  - LLM returns hypotheses and ordered sub-questions as JSON
  |
  v
run_query, repeated
  - retrieve dbt context from Qdrant
  - generate one SQL query
  - validate and limit SQL
  - dry-run when using BigQuery
  - execute and store a finding
  - retry once with repair prompt on validation/database error
  |
  v
reflect
  - LLM decides whether evidence is sufficient
  - may enqueue bounded follow-up sub-questions
  |
  v
synthesize_report
  - final cited report using only returned query results
```

Runtime is bounded by `MAX_SUB_QUERIES` through `AgentConfig.max_sub_queries`.
The default is 8 planned plus follow-up sub-queries total. A failed sub-question
is recorded as a finding with an error and does not abort the full run.

The streaming interface, `stream_analytics_events`, yields one event per graph
stage so the Streamlit UI can show the plan, each query, reflection, and final
answer as they complete. `run_analytics_question` remains available for
non-streaming callers.

### LLM Providers

`analytics_agent/core/llm.py` builds the chat model. Supported providers are:

- `openai`
- `claude`
- `kimi`
- `openrouter`

`LLM_PROVIDER` chooses the provider. `LLM_MODEL` chooses the model. `LLM_API_KEY`
and `LLM_BASE_URL` can override provider defaults. Embeddings intentionally stay
on OpenAI so one Qdrant collection remains compatible across chat providers.

### Dashboard Data Layer

`analytics_agent/dashboard/queries.py` contains author-written read-only SQL helpers for
the Streamlit dashboard. These are not LLM-generated queries, but they still use
the shared query runner and backend identifiers.

The dashboard supports:

- headline KPIs: users, events, AI generations, tokens, credits, exceptions,
  and AI adoption,
- daily activity and event mix,
- AI token usage by model and provider,
- users by role,
- exceptions by OS,
- activity-segment distribution,
- feature usage and feature trends,
- weekly retention curves and cohort matrix,
- ordered first-touch or unordered cumulative funnels.

Global filters live in the Streamlit sidebar:

- date range,
- user role,
- activity segment,
- continent,
- country,
- device type,
- operating system.

User-attribute filters compile to one consistent matching-user subquery. Daily
activity uses `agg_daily_activity` when no user filter is active, and recomputes
from `stg_events` when user filters are active so every chart reflects the same
cohort.

### Experiment And Cohort Analysis

The dataset has no randomized experiment-assignment column. The Experiments tab
and `analytics_agent/experiments/` therefore run observational cohort
comparisons and label them as quasi-experimental.

Supported comparison types:

- continuous metrics: Welch's t-test, Mann-Whitney U, Cohen's d, and a Welch
  confidence interval for the mean difference,
- proportions/rates: two-proportion z-test, chi-square test, risk difference,
  and Wald confidence interval.

The DB-backed samplers build internal SQL with identifier and literal guards,
then still pass the query through `validate_sql`. The optional LLM narrative
summarizes ship/no-ship implications while calling out confounding, sample
imbalance, confidence intervals crossing zero, and multiple-comparison risk.

### Streamlit App

`streamlit_app.py` is the product UI.

Tabs:

- `Ask`: streams the autonomous agent state live.
- `Dashboard`: renders cached KPIs, charts, feature usage, retention, and funnel
  views over the dbt marts.
- `Experiments`: lets users choose cohorts and metrics, then runs the
  observational statistics workflow.
- `About`: summarizes the architecture and project status.

## Setup

### Prerequisites

- Python 3.9+
- OpenAI API key for embeddings and the default chat provider
- Qdrant Cloud or local Qdrant for semantic retrieval
- Optional Google Cloud credentials for BigQuery backend use

### Install Dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Configure Environment

Copy the example environment file:

```bash
cp .env.example .env
```

Fill in the values you need.

| Variable | Purpose |
|---|---|
| `BACKEND` | `duckdb` by default; set to `bigquery` for BigQuery |
| `OPENAI_API_KEY` | Required for embeddings and default OpenAI chat |
| `OPENAI_EMBEDDING_MODEL` | Defaults to `text-embedding-3-small` |
| `LLM_PROVIDER` | `openai`, `claude`, `kimi`, or `openrouter` |
| `LLM_MODEL` | Chat model id for the chosen provider |
| `LLM_API_KEY` | Optional explicit chat-provider key |
| `LLM_BASE_URL` | Optional custom OpenAI-compatible base URL |
| `QDRANT_URL` | Qdrant endpoint |
| `QDRANT_API_KEY` | Qdrant API key |
| `QDRANT_COLLECTION_NAME` | Defaults to `dbt_semantic_dictionary` |
| `DUCKDB_PATH` | Defaults to `zerve_transform/zerve.duckdb` |
| `DUCKDB_SCHEMA` | Defaults to `main` |
| `BIGQUERY_PROJECT` | Required only for `BACKEND=bigquery` |
| `BIGQUERY_DATASET` | Required only for `BACKEND=bigquery` |
| `BIGQUERY_LOCATION` | Optional BigQuery location |
| `MAX_SUB_QUERIES` | Query budget per agent run, default 8 |

Never commit `.env`, service-account files, API keys, or generated databases.

## Common Workflows

### Build The Local DuckDB Warehouse

Run dbt from the dbt project directory:

```bash
cd zerve_transform
PYTHONWARNINGS=ignore ../.venv/bin/dbt build --target duckdb --profiles-dir .
```

This creates or refreshes `zerve_transform/zerve.duckdb` and
`zerve_transform/target/manifest.json`.

### Run dbt Tests

```bash
cd zerve_transform
PYTHONWARNINGS=ignore ../.venv/bin/dbt test --target duckdb --profiles-dir .
```

### Build Or Check The Vector Store

Rebuild Qdrant from the current dbt manifest and schema files:

```bash
python embedding_script.py --force-recreate
```

Run a local smoke test without OpenAI or Qdrant Cloud:

```bash
python embedding_script.py --local-smoke-test
```

Run a retrieval check:

```bash
python embedding_script.py --query "AI token usage by model"
```

### Launch The App

```bash
streamlit run streamlit_app.py
```

Then open `http://localhost:8501`.

Good example questions:

- What is driving AI credit consumption?
- How does AI adoption differ across user roles?
- Which segments have the highest exception rates?
- Which product features have the highest reach?
- How does week-one retention vary by cohort?

### Run Python Tests

```bash
python -m unittest discover -s tests
```

Test coverage includes:

- `tests/test_agent.py`: LangGraph control flow, streaming events, citations,
  query budget, and SQL repair,
- `tests/test_sql_validation.py`: read-only SQL validation and default limits,
- `tests/test_backend.py`: runner/backend behavior,
- `tests/test_dashboard.py`: filter compilation and dashboard query paths,
- `tests/test_experiments.py`: statistics, SQL construction, and injection
  guards,
- `tests/test_llm.py`: provider selection and model configuration.

## Project Structure

```text
.
├── README.md
├── .env.example
├── requirements.txt
├── model_profiles.yml
├── streamlit_app.py                # thin entrypoint: wires the UI tabs together
├── embedding_script.py
├── Data Dictionary -- ODSC_Zerve Datathon FINAL.xlsx
├── analytics_agent/                # organized by feature (one folder per UI tab)
│   ├── __init__.py
│   ├── core/                       # shared infra (no feature imports)
│   │   ├── config.py
│   │   ├── contracts.py
│   │   ├── llm.py
│   │   ├── sql.py
│   │   ├── runners.py
│   │   ├── bigquery_client.py
│   │   └── retriever.py
│   ├── ask/                        # "Ask" tab: the agentic trust loop
│   │   ├── graph.py
│   │   ├── semantic_layer.py
│   │   ├── model_router.py
│   │   └── snapshots.py
│   ├── library/                    # "Library" tab: saved-report memory
│   │   └── memory.py
│   ├── dashboard/                  # "Dashboard" tab
│   │   ├── queries.py              # read-only metric SQL helpers
│   │   └── render.py               # generated-dashboard rendering/persistence
│   ├── experiments/                # "Experiments" tab (cohort statistics)
│   │   └── __init__.py
│   └── ui/                         # Streamlit UI, one module per tab
│       ├── theme.py
│       ├── helpers.py
│       ├── common.py
│       ├── ask.py
│       ├── library.py
│       ├── dashboard.py
│       ├── experiments.py
│       └── about.py
├── tests/
│   ├── test_agent.py
│   ├── test_backend.py
│   ├── test_dashboard.py
│   ├── test_experiments.py
│   ├── test_llm.py
│   └── test_sql_validation.py
└── zerve_transform/
    ├── dbt_project.yml
    ├── profiles.yml
    ├── packages.yml
    ├── package-lock.yml
    ├── README.md
    └── models/
        ├── staging/
        │   ├── schema.yml
        │   ├── stg_events.sql
        │   ├── stg_feature_events.sql
        │   └── user_events.sql
        └── marts/
            ├── schema.yml
            ├── agg_daily_activity.sql
            ├── agg_feature_usage.sql
            ├── dim_users.sql
            ├── fct_ai_generations.sql
            ├── fct_credits.sql
            ├── fct_exceptions.sql
            ├── fct_pageviews.sql
            ├── fct_user_activity_weekly.sql
            ├── metricflow_time_spine.sql
            └── user_summary.sql
```

Generated or local-only artifacts you may see:

- `.venv/`: local Python environment,
- `__pycache__/`: Python bytecode cache,
- `zerve_transform/target/`: dbt compiled output and `manifest.json`,
- `zerve_transform/dbt_packages/`: installed dbt packages,
- `zerve_transform/zerve.duckdb`: local warehouse,
- `logs/`: dbt and query logs,
- `.env`: local secrets and runtime configuration.

## Maintenance Notes

- When adding a dbt model, document it in the appropriate `schema.yml` and run
  `dbt build`; the manifest-derived allow-list will pick it up automatically.
- When adding dashboard queries, keep them read-only, backend-aware through
  `config.fully_qualified_tables`, and covered by focused tests.
- When changing agent prompts or state flow, update tests that fake the LLM and
  runner so the control flow remains bounded and inspectable.
- When adding experiment metrics or cohort dimensions, preserve the
  quasi-experimental framing and validate generated SQL through the shared
  guardrails.
- Rebuild the Qdrant collection after changing dbt model descriptions, columns,
  semantic models, or metrics so retrieval reflects the new semantic layer.

## Troubleshooting

- `Missing required environment variable`: check `.env` and confirm
  `load_dotenv()` can find it from the repo root.
- `No dbt model documents found`: build dbt first so
  `zerve_transform/target/manifest.json` exists.
- DuckDB file not found: run the dbt build command and confirm `DUCKDB_PATH`
  matches the path in `zerve_transform/profiles.yml`.
- SQL validation rejects a query: use fully qualified identifiers from
  `AgentConfig.fully_qualified_tables`; unqualified table names are rejected.
- Empty or stale retrieval results: rebuild the Qdrant collection with
  `python embedding_script.py --force-recreate`.
- BigQuery credential errors: set `GOOGLE_APPLICATION_CREDENTIALS` and the
  `BIGQUERY_*` environment variables, or switch back to `BACKEND=duckdb`.

## Future Possibilities

### Self-Healing Data Quality

This is not implemented because the current static datathon pipeline does not
need it. If the project ingested live data, a natural extension would be:

1. Detect failures using stronger dbt tests, source freshness, row-count checks,
   distribution checks, and parsed `run_results.json`.
2. Triage failures by giving an LLM the failing test, model SQL, and sampled
   offending rows.
3. Propose reviewed dbt patches for low-risk fixes, such as null handling, type
   widening, or quarantining malformed rows.

Fixes should go through review rather than silently mutating the warehouse.

### Product Extensions

- AI cost view once a token-to-price mapping is available.
- More geographic breakdowns using existing `country` and `continent` fields.
- Custom user-defined funnels in the UI.
- Saved dashboard cards from agent result tables.
- Scheduled refresh or deployment workflow if the project moves beyond local
  datathon use.
