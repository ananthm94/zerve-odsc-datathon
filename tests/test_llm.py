import unittest
from dataclasses import replace

from analytics_agent.config import AgentConfig, ConfigError
from analytics_agent.llm import (
    MOONSHOT_BASE_URL,
    OPENROUTER_BASE_URL,
    build_chat_model,
)


def _config(**overrides) -> AgentConfig:
    base = AgentConfig(
        openai_api_key="test-openai-key",
        qdrant_url="http://localhost",
        qdrant_api_key="test-qdrant-key",
        qdrant_collection_name="test",
        bigquery_project="test-project",
        bigquery_dataset="test-dataset",
        bigquery_location=None,
        chat_model="test-model",
        llm_api_key="test-llm-key",
    )
    return replace(base, **overrides)


class BuildChatModelTests(unittest.TestCase):
    def test_openai_provider(self):
        from langchain_openai import ChatOpenAI

        model = build_chat_model(_config(llm_provider="openai"))
        self.assertIsInstance(model, ChatOpenAI)

    def test_claude_provider(self):
        from langchain_anthropic import ChatAnthropic

        model = build_chat_model(_config(llm_provider="claude"))
        self.assertIsInstance(model, ChatAnthropic)

    def test_kimi_uses_moonshot_base_url(self):
        from langchain_openai import ChatOpenAI

        model = build_chat_model(_config(llm_provider="kimi"))
        self.assertIsInstance(model, ChatOpenAI)
        self.assertEqual(str(model.openai_api_base), MOONSHOT_BASE_URL)

    def test_openrouter_uses_openrouter_base_url(self):
        from langchain_openai import ChatOpenAI

        model = build_chat_model(_config(llm_provider="openrouter"))
        self.assertIsInstance(model, ChatOpenAI)
        self.assertEqual(str(model.openai_api_base), OPENROUTER_BASE_URL)

    def test_explicit_base_url_overrides_default(self):
        model = build_chat_model(
            _config(llm_provider="kimi", llm_base_url="https://custom.example/v1")
        )
        self.assertEqual(str(model.openai_api_base), "https://custom.example/v1")

    def test_unknown_provider_raises(self):
        with self.assertRaises(ConfigError):
            build_chat_model(_config(llm_provider="bogus"))


if __name__ == "__main__":
    unittest.main()
