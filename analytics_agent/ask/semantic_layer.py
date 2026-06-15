"""Structured access to the dbt MetricFlow semantic layer.

This module turns ``target/semantic_manifest.json`` into the shared
``MetricDefinition`` contract used by the agentic orchestrator. The goal is to
ground metric selection in dbt's official definitions before any LLM writes SQL.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from analytics_agent.core.config import AgentConfig
from analytics_agent.core.contracts import MetricDefinition


def load_semantic_layer(config: AgentConfig) -> dict[str, MetricDefinition]:
    """Load MetricFlow metric definitions keyed by metric name."""
    path = Path(config.semantic_manifest_path)
    with path.open() as f:
        manifest = json.load(f)

    semantic_models = manifest.get("semantic_models") or []
    metrics = manifest.get("metrics") or []
    measure_index = _index_measures(semantic_models)

    layer: dict[str, MetricDefinition] = {}
    for metric in metrics:
        name = str(metric.get("name") or "").strip()
        if not name:
            continue
        input_measures = _input_measure_names(metric)
        models = [
            measure_index[measure]
            for measure in input_measures
            if measure in measure_index
        ]
        layer[name] = _definition_from_metric(metric, models)
    return layer


def lookup_metrics(
    question: str, layer: dict[str, MetricDefinition], llm
) -> list[MetricDefinition]:
    """Return likely metrics for a question.

    Deterministic term matching is the primary path because it is predictable,
    cheap, and easy to test. If nothing matches, a small LLM fallback can return
    metric names as JSON/text; failures simply return an empty list.
    """
    scored = sorted(
        (
            (_match_score(question, definition), definition.metric, definition)
            for definition in layer.values()
        ),
        reverse=True,
    )
    matches = [definition for score, _, definition in scored if score > 0]
    if matches:
        return matches[:5]

    for name in _llm_metric_names(question, layer, llm):
        if name in layer:
            return [layer[name]]
    return []


def format_for_prompt(defs: list[MetricDefinition]) -> str:
    """Format metric definitions for an LLM prompt."""
    if not defs:
        return "No official metric definitions matched."

    blocks = []
    for definition in defs:
        blocks.append(
            "\n".join(
                [
                    f"Metric: {definition.metric}",
                    f"Definition: {definition.definition}",
                    f"Type: {definition.metric_type}",
                    f"Measure: {definition.measure}",
                    f"Grain: {definition.grain}",
                    f"Time column: {definition.time_column}",
                    f"Source table: {definition.source_table}",
                    "Allowed dimensions: "
                    + (
                        ", ".join(definition.allowed_dimensions)
                        if definition.allowed_dimensions
                        else "(none)"
                    ),
                    "Join keys: "
                    + (
                        ", ".join(definition.join_keys)
                        if definition.join_keys
                        else "(none)"
                    ),
                ]
            )
        )
    return "\n\n---\n\n".join(blocks)


def _index_measures(
    semantic_models: list[dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for model in semantic_models:
        for measure in model.get("measures") or []:
            name = measure.get("name")
            if name:
                index[str(name)] = {"model": model, "measure": measure}
    return index


def _input_measure_names(metric: dict[str, Any]) -> list[str]:
    params = metric.get("type_params") or {}
    input_measures = params.get("input_measures") or []
    names = [item.get("name") for item in input_measures if item.get("name")]
    if names:
        return [str(name) for name in names]

    measure = params.get("measure") or {}
    if measure.get("name"):
        return [str(measure["name"])]
    return []


def _definition_from_metric(
    metric: dict[str, Any], model_measure_pairs: list[dict[str, Any]]
) -> MetricDefinition:
    models = [pair["model"] for pair in model_measure_pairs]
    measures = [pair["measure"] for pair in model_measure_pairs]
    time_columns = _unique(
        (model.get("defaults") or {}).get("agg_time_dimension")
        for model in models
        if (model.get("defaults") or {}).get("agg_time_dimension")
    )
    dimensions = _unique(
        dim.get("name")
        for model in models
        for dim in model.get("dimensions") or []
        if dim.get("name")
    )
    source_tables = _unique(
        (model.get("node_relation") or {}).get("alias")
        for model in models
        if (model.get("node_relation") or {}).get("alias")
    )
    join_keys = _unique(
        entity.get("expr") or entity.get("name")
        for model in models
        for entity in model.get("entities") or []
        if entity.get("expr") or entity.get("name")
    )
    measure_names = _unique(measure.get("name") for measure in measures)

    return MetricDefinition(
        metric=str(metric.get("name") or ""),
        definition=str(metric.get("description") or metric.get("label") or ""),
        grain=_grain_from_time_columns(time_columns),
        time_column=", ".join(time_columns),
        allowed_dimensions=dimensions,
        source_table=", ".join(source_tables),
        join_keys=join_keys,
        measure=", ".join(measure_names),
        metric_type=str(metric.get("type") or ""),
    )


def _grain_from_time_columns(time_columns: list[str]) -> str:
    if not time_columns:
        return ""
    if any("date" in column or "day" in column for column in time_columns):
        return "day"
    return ", ".join(time_columns)


def _match_score(question: str, definition: MetricDefinition) -> int:
    question_terms = set(_terms(question))
    if not question_terms:
        return 0

    metric_terms = set(_terms(definition.metric))
    definition_terms = set(_terms(definition.definition))
    measure_terms = set(_terms(definition.measure))
    dimension_terms = set(_terms(" ".join(definition.allowed_dimensions)))

    score = 0
    score += 5 * len(question_terms & metric_terms)
    score += 3 * len(question_terms & measure_terms)
    score += 2 * len(question_terms & definition_terms)
    score += len(question_terms & dimension_terms)

    compact_question = " ".join(sorted(question_terms))
    if definition.metric.replace("_", " ") in question.lower():
        score += 10
    if "dau" in question_terms and definition.metric == "daily_active_users":
        score += 10
    if "adoption" in question_terms and definition.metric == "ai_adoption_rate":
        score += 4
    if "tokens" in question_terms and "token" in metric_terms:
        score += 3
    if compact_question and definition.metric in compact_question:
        score += 2
    return score


def _terms(text: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", text.lower().replace("_", " "))
    expanded = []
    for token in tokens:
        expanded.append(token)
        if token.endswith("s") and len(token) > 3:
            expanded.append(token[:-1])
    return expanded


def _llm_metric_names(
    question: str, layer: dict[str, MetricDefinition], llm
) -> list[str]:
    if llm is None:
        return []
    names = ", ".join(sorted(layer))
    try:
        response = llm.invoke(
            [
                (
                    "system",
                    "Choose the best official metric names for the question. "
                    "Return only a JSON array of metric names.",
                ),
                ("human", f"Question: {question}\nAvailable metrics: {names}"),
            ]
        )
    except Exception:
        return []

    content = getattr(response, "content", response)
    try:
        parsed = json.loads(str(content))
    except ValueError:
        parsed = re.findall(r"[a-z][a-z0-9_]+", str(content).lower())
    if not isinstance(parsed, list):
        return []
    return [str(name) for name in parsed]


def _unique(values) -> list[str]:
    seen = set()
    result = []
    for value in values:
        if value is None:
            continue
        text = str(value)
        if text and text not in seen:
            seen.add(text)
            result.append(text)
    return result


__all__ = ["format_for_prompt", "load_semantic_layer", "lookup_metrics"]
