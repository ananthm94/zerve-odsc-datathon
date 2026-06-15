import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from analytics_agent.core.config import AgentConfig
from analytics_agent.ask.model_router import (
    TASK_CONDENSE,
    TASK_SQL,
    estimate_call_cost,
    estimate_tokens,
    select_model,
    usage_summary,
)


class _Resp:
    content = "hello world"
    usage_metadata = {"input_tokens": 10, "output_tokens": 5}


def _config(**overrides) -> AgentConfig:
    base = AgentConfig(
        openai_api_key="openai",
        qdrant_url="http://qdrant",
        qdrant_api_key="qdrant",
        qdrant_collection_name="collection",
        llm_provider="openai",
        chat_model="gpt-4.1-mini",
    )
    return replace(base, **overrides)


class ModelRouterTests(unittest.TestCase):
    def test_global_model_when_routing_disabled(self):
        cfg = _config(model_routing_enabled=False)
        selection = select_model(cfg, TASK_SQL)

        self.assertEqual(selection.provider, "openai")
        self.assertEqual(selection.model, "gpt-4.1-mini")
        self.assertEqual(selection.difficulty, "global")

    def test_task_model_from_profile_when_enabled(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.yml"
            path.write_text(
                """
default_provider: openrouter
default_model: openai/gpt-4.1-mini
tasks:
  sql:
    difficulty: hard
    provider: openrouter
    model: anthropic/claude-sonnet-4
"""
            )
            cfg = _config(model_routing_enabled=True, model_profile_path=str(path))

            selection = select_model(cfg, TASK_SQL)

        self.assertEqual(selection.provider, "openrouter")
        self.assertEqual(selection.model, "anthropic/claude-sonnet-4")
        self.assertEqual(selection.difficulty, "hard")

    def test_estimate_call_cost_uses_actual_usage_when_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "profiles.yml"
            path.write_text(
                """
prices:
  gpt-4.1-mini:
    input_per_million: 1.0
    output_per_million: 2.0
"""
            )
            cfg = _config(model_profile_path=str(path))

            estimate = estimate_call_cost(cfg, TASK_SQL, "prompt", _Resp())

        self.assertTrue(estimate.used_actual_usage)
        self.assertEqual(estimate.input_tokens, 10)
        self.assertEqual(estimate.output_tokens, 5)
        self.assertAlmostEqual(estimate.estimated_cost_usd, 0.00002)

    def test_condense_task_falls_back_to_global_model(self):
        # No profile entry for "condense": it should use the global default like
        # any other unconfigured task rather than erroring.
        cfg = _config(model_routing_enabled=True, model_profile_path="missing.yml")
        selection = select_model(cfg, TASK_CONDENSE)

        self.assertEqual(selection.task, TASK_CONDENSE)
        self.assertEqual(selection.provider, "openai")
        self.assertEqual(selection.model, "gpt-4.1-mini")

    def test_usage_summary_totals(self):
        cfg = _config()
        estimate = estimate_call_cost(cfg, TASK_SQL, "abcd" * 10, "abcd" * 5)
        summary = usage_summary([estimate])

        self.assertEqual(summary["input_tokens"], estimate.input_tokens)
        self.assertEqual(summary["output_tokens"], estimate.output_tokens)
        self.assertEqual(len(summary["calls"]), 1)
        self.assertGreater(estimate_tokens("hello"), 0)


if __name__ == "__main__":
    unittest.main()
