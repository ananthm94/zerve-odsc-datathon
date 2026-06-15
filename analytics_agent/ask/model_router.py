"""Task-level model routing and lightweight cost estimation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from analytics_agent.core.config import AgentConfig
from analytics_agent.core.llm import build_chat_model


TASK_INTENT = "intent"
TASK_CONDENSE = "condense"
TASK_PLAN = "plan"
TASK_SQL = "sql"
TASK_REPAIR = "repair"
TASK_REFLECT = "reflect"
TASK_SYNTHESIZE = "synthesize"
TASK_DASHBOARD_REPAIR = "dashboard_repair"
TASK_EXPERIMENT_NARRATIVE = "experiment_narrative"


@dataclass(frozen=True)
class ModelSelection:
    task: str
    provider: str
    model: str
    difficulty: str = "standard"


@dataclass(frozen=True)
class ModelUsageEstimate:
    task: str
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    used_actual_usage: bool = False


def select_model(config: AgentConfig, task: str) -> ModelSelection:
    """Pick the model for a task from profile config or fall back globally."""
    profile = load_model_profile(config)
    if not config.model_routing_enabled:
        return ModelSelection(
            task=task,
            provider=config.llm_provider,
            model=config.chat_model,
            difficulty="global",
        )

    default_provider = profile.get("default_provider") or config.llm_provider
    default_model = profile.get("default_model") or config.chat_model
    task_cfg = (profile.get("tasks") or {}).get(task, {})
    return ModelSelection(
        task=task,
        provider=task_cfg.get("provider") or default_provider,
        model=task_cfg.get("model") or default_model,
        difficulty=task_cfg.get("difficulty") or "standard",
    )


def build_model_for_task(
    config: AgentConfig, task: str, *, temperature: float = 0
):
    selection = select_model(config, task)
    return build_chat_model(
        config,
        temperature=temperature,
        provider=selection.provider,
        model=selection.model,
    )


def estimate_call_cost(
    config: AgentConfig,
    task: str,
    prompt_text: str,
    response: Any,
    selection: ModelSelection | None = None,
) -> ModelUsageEstimate:
    selection = selection or select_model(config, task)
    actual = _usage_from_response(response)
    if actual:
        input_tokens, output_tokens = actual
        used_actual = True
    else:
        input_tokens = estimate_tokens(prompt_text)
        content = getattr(response, "content", response)
        output_tokens = estimate_tokens(str(content or ""))
        used_actual = False

    rates = _rates_for_model(config, selection.model)
    cost = (
        input_tokens / 1_000_000 * rates.get("input_per_million", 0.0)
        + output_tokens / 1_000_000 * rates.get("output_per_million", 0.0)
    )
    return ModelUsageEstimate(
        task=task,
        provider=selection.provider,
        model=selection.model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost,
        used_actual_usage=used_actual,
    )


def estimate_tokens(text: str) -> int:
    """Cheap provider-agnostic estimate: about four chars per token."""
    if not text:
        return 0
    return max(1, int(len(text) / 4))


def usage_summary(estimates: list[ModelUsageEstimate]) -> dict[str, Any]:
    return {
        "input_tokens": sum(item.input_tokens for item in estimates),
        "output_tokens": sum(item.output_tokens for item in estimates),
        "estimated_cost_usd": sum(item.estimated_cost_usd for item in estimates),
        "calls": [
            {
                "task": item.task,
                "provider": item.provider,
                "model": item.model,
                "input_tokens": item.input_tokens,
                "output_tokens": item.output_tokens,
                "estimated_cost_usd": item.estimated_cost_usd,
                "used_actual_usage": item.used_actual_usage,
            }
            for item in estimates
        ],
    }


def load_model_profile(config: AgentConfig) -> dict:
    path = Path(config.model_profile_path)
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _rates_for_model(config: AgentConfig, model: str) -> dict[str, float]:
    prices = (load_model_profile(config).get("prices") or {})
    rate = prices.get(model) or {}
    return {
        "input_per_million": float(rate.get("input_per_million", 0.0) or 0.0),
        "output_per_million": float(rate.get("output_per_million", 0.0) or 0.0),
    }


def _usage_from_response(response: Any) -> tuple[int, int] | None:
    metadata = getattr(response, "usage_metadata", None)
    if metadata:
        input_tokens = metadata.get("input_tokens") or metadata.get("prompt_tokens")
        output_tokens = metadata.get("output_tokens") or metadata.get("completion_tokens")
        if input_tokens is not None and output_tokens is not None:
            return int(input_tokens), int(output_tokens)

    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage")
    if token_usage:
        input_tokens = token_usage.get("prompt_tokens") or token_usage.get("input_tokens")
        output_tokens = token_usage.get("completion_tokens") or token_usage.get(
            "output_tokens"
        )
        if input_tokens is not None and output_tokens is not None:
            return int(input_tokens), int(output_tokens)
    return None


def messages_to_text(messages: list[Any]) -> str:
    parts = []
    for message in messages:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, tuple) and len(message) >= 2:
            content = message[1]
        parts.append(str(content or ""))
    return "\n\n".join(parts)


def dump_usage_json(summary: dict[str, Any]) -> str:
    return json.dumps(summary, indent=2, sort_keys=True)


__all__ = [
    "ModelSelection",
    "ModelUsageEstimate",
    "TASK_CONDENSE",
    "TASK_DASHBOARD_REPAIR",
    "TASK_EXPERIMENT_NARRATIVE",
    "TASK_INTENT",
    "TASK_PLAN",
    "TASK_REFLECT",
    "TASK_REPAIR",
    "TASK_SQL",
    "TASK_SYNTHESIZE",
    "build_model_for_task",
    "dump_usage_json",
    "estimate_call_cost",
    "estimate_tokens",
    "load_model_profile",
    "messages_to_text",
    "select_model",
    "usage_summary",
]
