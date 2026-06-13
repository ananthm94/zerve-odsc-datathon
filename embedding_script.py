import argparse
import glob
import json
import os
from typing import Iterable

import yaml
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore


DEFAULT_MANIFEST_PATH = "zerve_transform/target/manifest.json"
DEFAULT_SCHEMA_GLOB = "zerve_transform/models/**/*.yml"
DEFAULT_COLLECTION_NAME = "dbt_semantic_dictionary"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


class SmokeTestEmbeddings(Embeddings):
    """Tiny deterministic embeddings for local Qdrant plumbing checks."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    @staticmethod
    def _embed(text: str) -> list[float]:
        lowered = text.lower()
        return [
            float("user" in lowered),
            float("event" in lowered),
            float("metric" in lowered),
            float("timestamp" in lowered),
            float("activity" in lowered),
        ]


def load_dbt_nodes(manifest_path: str) -> dict:
    with open(manifest_path) as f:
        manifest = json.load(f)
    return manifest.get("nodes", {})


def build_model_documents(nodes: dict) -> list[Document]:
    documents = []

    for node_id, node_data in nodes.items():
        if node_data.get("resource_type") != "model":
            continue

        model_name = node_data.get("name")
        description = node_data.get("description") or "No description provided."
        columns = node_data.get("columns", {})
        column_lines = [
            f"- {column_name} ({column_data.get('data_type', 'unknown')}): "
            f"{column_data.get('description', '')}"
            for column_name, column_data in columns.items()
        ]

        page_content = (
            f"Table Name: {model_name}\n"
            f"Description: {description}\n"
            f"Columns:\n{chr(10).join(column_lines)}"
        )
        metadata = {
            "doc_type": "model",
            "node_id": node_id,
            "model_name": model_name,
            "materialized": node_data.get("config", {}).get("materialized"),
            "database": node_data.get("database"),
            "schema": node_data.get("schema"),
            "resource_type": node_data.get("resource_type"),
        }
        documents.append(Document(page_content=page_content, metadata=metadata))

    return documents


def _load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _format_named_items(items: list[dict], fields: tuple[str, ...]) -> str:
    lines = []
    for item in items or []:
        values = [f"{field}: {item.get(field)}" for field in fields if item.get(field)]
        lines.append("- " + ", ".join(values))
    return "\n".join(lines)


def build_metric_documents(schema_glob: str = DEFAULT_SCHEMA_GLOB) -> list[Document]:
    documents = []

    for path in glob.glob(schema_glob, recursive=True):
        schema = _load_yaml(path)

        for metric in schema.get("metrics", []) or []:
            name = metric.get("name")
            label = metric.get("label") or name
            description = metric.get("description") or "No description provided."
            metric_type = metric.get("type")
            type_params = metric.get("type_params", {})
            measure = type_params.get("measure")
            page_content = (
                f"Metric Name: {name}\n"
                f"Label: {label}\n"
                f"Description: {description}\n"
                f"Type: {metric_type}\n"
                f"Measure: {measure}"
            )
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "doc_type": "metric",
                        "name": name,
                        "label": label,
                        "source_path": path,
                    },
                )
            )

        for semantic_model in schema.get("semantic_models", []) or []:
            name = semantic_model.get("name")
            model = semantic_model.get("model")
            dimensions = _format_named_items(
                semantic_model.get("dimensions", []), ("name", "type", "expr")
            )
            measures = _format_named_items(
                semantic_model.get("measures", []),
                ("name", "description", "agg", "expr"),
            )
            entities = _format_named_items(
                semantic_model.get("entities", []), ("name", "type", "expr")
            )
            page_content = (
                f"Semantic Model: {name}\n"
                f"dbt Model: {model}\n"
                f"Entities:\n{entities}\n"
                f"Dimensions:\n{dimensions}\n"
                f"Measures:\n{measures}"
            )
            documents.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "doc_type": "semantic_model",
                        "name": name,
                        "model": model,
                        "source_path": path,
                    },
                )
            )

    return documents


def build_documents(nodes: dict, schema_glob: str = DEFAULT_SCHEMA_GLOB) -> list[Document]:
    return build_model_documents(nodes) + build_metric_documents(schema_glob)


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def create_vector_store(
    documents: list[Document],
    embeddings: Embeddings,
    collection_name: str,
    force_recreate: bool,
    use_memory: bool,
) -> QdrantVectorStore:
    if use_memory:
        return QdrantVectorStore.from_documents(
            documents,
            embeddings,
            location=":memory:",
            collection_name=collection_name,
            force_recreate=force_recreate,
        )

    return QdrantVectorStore.from_documents(
        documents,
        embeddings,
        url=require_env("QDRANT_URL"),
        api_key=require_env("QDRANT_API_KEY"),
        collection_name=collection_name,
        force_recreate=force_recreate,
        check_compatibility=False,
    )


def print_results(results: Iterable[Document]) -> None:
    for result in results:
        name = result.metadata.get("model_name") or result.metadata.get("name")
        doc_type = result.metadata.get("doc_type", "document")
        print(f"Found: {name} ({doc_type})")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Embed dbt manifest metadata into Qdrant and run a retrieval check."
    )
    parser.add_argument("--manifest-path", default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--schema-glob", default=DEFAULT_SCHEMA_GLOB)
    parser.add_argument(
        "--collection-name",
        default=os.getenv("QDRANT_COLLECTION_NAME", DEFAULT_COLLECTION_NAME),
    )
    parser.add_argument(
        "--query",
        default="Which table has data about user?",
        help="Retriever query to run after loading documents.",
    )
    parser.add_argument(
        "--force-recreate",
        action="store_true",
        help="Drop and recreate the Qdrant collection before loading documents.",
    )
    parser.add_argument(
        "--local-smoke-test",
        action="store_true",
        help="Use in-memory Qdrant and deterministic embeddings. No OpenAI or Qdrant Cloud calls.",
    )
    args = parser.parse_args()

    documents = build_documents(load_dbt_nodes(args.manifest_path), args.schema_glob)
    if not documents:
        raise RuntimeError(f"No dbt model documents found in {args.manifest_path}")

    if args.local_smoke_test:
        embeddings: Embeddings = SmokeTestEmbeddings()
    else:
        embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
        )

    vector_store = create_vector_store(
        documents=documents,
        embeddings=embeddings,
        collection_name=args.collection_name,
        force_recreate=args.force_recreate,
        use_memory=args.local_smoke_test,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    print(f"Loaded {len(documents)} dbt semantic documents into {args.collection_name}.")
    print_results(retriever.invoke(args.query))


if __name__ == "__main__":
    main()
