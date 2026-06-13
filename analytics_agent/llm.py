"""Provider-agnostic chat model factory.

The chat LLM is configurable across OpenAI, Anthropic (Claude), Kimi (Moonshot),
and OpenRouter. Embeddings deliberately stay on OpenAI (see retriever.py) so the
existing Qdrant collection remains valid.

Provider SDKs are imported lazily inside each branch so that an optional package
(e.g. langchain-anthropic) being absent never breaks the default OpenAI path.
"""

from __future__ import annotations

import os

from analytics_agent.config import AgentConfig, ConfigError


KNOWN_PROVIDERS = {"openai", "claude", "kimi", "openrouter"}

# OpenAI-compatible providers reached via a custom base URL.
MOONSHOT_BASE_URL = "https://api.moonshot.ai/v1"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Standard env var to fall back to when LLM_API_KEY is not set explicitly.
_STANDARD_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "kimi": "MOONSHOT_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}


def _resolve_api_key(provider: str, config: AgentConfig) -> str | None:
    if config.llm_api_key:
        return config.llm_api_key
    env_name = _STANDARD_KEY_ENV.get(provider)
    return os.getenv(env_name) if env_name else None


def _resolve_base_url(provider: str, config: AgentConfig) -> str | None:
    if config.llm_base_url:
        return config.llm_base_url
    if provider == "kimi":
        return MOONSHOT_BASE_URL
    if provider == "openrouter":
        return OPENROUTER_BASE_URL
    return None


def build_chat_model(config: AgentConfig, *, temperature: float = 0):
    """Return a LangChain chat model for the configured provider."""
    provider = (config.llm_provider or "openai").lower()
    if provider not in KNOWN_PROVIDERS:
        raise ConfigError(
            f"Unknown LLM provider: {provider!r}. "
            f"Expected one of {sorted(KNOWN_PROVIDERS)}."
        )

    api_key = _resolve_api_key(provider, config)

    if provider == "claude":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:  # pragma: no cover - import guard
            raise ConfigError(
                "LLM_PROVIDER=claude requires the 'langchain-anthropic' package. "
                "Install it with: pip install langchain-anthropic"
            ) from exc
        kwargs: dict = {"model": config.chat_model, "temperature": temperature}
        if api_key:
            kwargs["api_key"] = api_key
        return ChatAnthropic(**kwargs)

    # openai / kimi / openrouter are all OpenAI-compatible.
    from langchain_openai import ChatOpenAI

    kwargs = {"model": config.chat_model, "temperature": temperature}
    if api_key:
        kwargs["api_key"] = api_key
    base_url = _resolve_base_url(provider, config)
    if base_url:
        kwargs["base_url"] = base_url
    return ChatOpenAI(**kwargs)
