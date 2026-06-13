from __future__ import annotations

import json
import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_MANIFEST_PATH = "zerve_transform/target/manifest.json"
# Fallback allowlist if the dbt manifest cannot be read.
FALLBACK_TABLE_NAMES = {"stg_events", "user_events", "user_summary"}
DEFAULT_COLLECTION_NAME = "dbt_semantic_dictionary"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_LLM_PROVIDER = "openai"
DEFAULT_RESULT_LIMIT = 100
# Ceiling on how many sub-queries the autonomous agent may run for one question
# (planned + reflection follow-ups), bounding runtime and LLM/warehouse cost.
DEFAULT_MAX_SUB_QUERIES = 8

DEFAULT_BACKEND = "duckdb"
# dbt (run from the zerve_transform/ dir) writes the database here; the agent,
# run from the repo root, reads the same file at this path.
DEFAULT_DUCKDB_PATH = "zerve_transform/zerve.duckdb"
DEFAULT_DUCKDB_SCHEMA = "main"
KNOWN_BACKENDS = {"duckdb", "bigquery"}


class ConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class AgentConfig:
    openai_api_key: str
    qdrant_url: str
    qdrant_api_key: str
    qdrant_collection_name: str
    bigquery_project: str = ""
    bigquery_dataset: str = ""
    bigquery_location: str | None = None
    backend: str = DEFAULT_BACKEND
    duckdb_path: str = DEFAULT_DUCKDB_PATH
    duckdb_schema: str = DEFAULT_DUCKDB_SCHEMA
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    chat_model: str = DEFAULT_CHAT_MODEL
    llm_provider: str = DEFAULT_LLM_PROVIDER
    llm_api_key: str | None = None
    llm_base_url: str | None = None
    manifest_path: str = DEFAULT_MANIFEST_PATH
    default_result_limit: int = DEFAULT_RESULT_LIMIT
    max_sub_queries: int = DEFAULT_MAX_SUB_QUERIES

    @property
    def allowed_table_names(self) -> set[str]:
        """Queryable models, derived from the dbt manifest so new marts are
        automatically allowed. Falls back to the core models if the manifest
        is unavailable. The time-spine helper model is excluded."""
        names = _model_names_from_manifest(self.manifest_path)
        return names or set(FALLBACK_TABLE_NAMES)

    @property
    def fully_qualified_tables(self) -> dict[str, str]:
        """Map each allowed model to its identifier in the active backend's dialect.

        - BigQuery: backtick-qualified ``project.dataset.table``.
        - DuckDB:   ``schema.table`` (no backticks).
        """
        if self.backend == "bigquery":
            return {
                table: f"`{self.bigquery_project}.{self.bigquery_dataset}.{table}`"
                for table in self.allowed_table_names
            }
        return {
            table: f"{self.duckdb_schema}.{table}"
            for table in self.allowed_table_names
        }


def _model_names_from_manifest(manifest_path: str) -> set[str]:
    """Read dbt model names from the manifest. Excludes the MetricFlow time
    spine (not a queryable analytics table). Returns an empty set on failure."""
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (OSError, ValueError):
        return set()
    names = {
        node["name"]
        for node in manifest.get("nodes", {}).values()
        if node.get("resource_type") == "model"
    }
    names.discard("metricflow_time_spine")
    return names


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ConfigError(f"Missing required environment variable: {name}")
    return value


def load_config() -> AgentConfig:
    load_dotenv()

    backend = os.getenv("BACKEND", DEFAULT_BACKEND).lower()
    if backend not in KNOWN_BACKENDS:
        raise ConfigError(
            f"Unknown BACKEND: {backend!r}. Expected one of {sorted(KNOWN_BACKENDS)}."
        )

    # BigQuery project/dataset are only required when BigQuery is the active backend.
    if backend == "bigquery":
        bigquery_project = _required_env("BIGQUERY_PROJECT")
        bigquery_dataset = _required_env("BIGQUERY_DATASET")
    else:
        bigquery_project = os.getenv("BIGQUERY_PROJECT", "")
        bigquery_dataset = os.getenv("BIGQUERY_DATASET", "")

    return AgentConfig(
        openai_api_key=_required_env("OPENAI_API_KEY"),
        qdrant_url=_required_env("QDRANT_URL"),
        qdrant_api_key=_required_env("QDRANT_API_KEY"),
        qdrant_collection_name=os.getenv(
            "QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME
        ),
        bigquery_project=bigquery_project,
        bigquery_dataset=bigquery_dataset,
        bigquery_location=os.getenv("BIGQUERY_LOCATION"),
        backend=backend,
        duckdb_path=os.getenv("DUCKDB_PATH", DEFAULT_DUCKDB_PATH),
        duckdb_schema=os.getenv("DUCKDB_SCHEMA", DEFAULT_DUCKDB_SCHEMA),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
        chat_model=(
            os.getenv("LLM_MODEL")
            or os.getenv("OPENAI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        ),
        llm_provider=os.getenv("LLM_PROVIDER", DEFAULT_LLM_PROVIDER),
        llm_api_key=os.getenv("LLM_API_KEY") or None,
        llm_base_url=os.getenv("LLM_BASE_URL") or None,
        manifest_path=os.getenv("DBT_MANIFEST_PATH", DEFAULT_MANIFEST_PATH),
        max_sub_queries=int(
            os.getenv("MAX_SUB_QUERIES", str(DEFAULT_MAX_SUB_QUERIES))
        ),
    )
