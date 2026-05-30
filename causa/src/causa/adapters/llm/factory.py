"""LLM client factory.

Implements the dispatch from ``CAUSA_LLM_PROVIDER`` to a concrete adapter.
This keeps the rest of the system honest: nothing imports the Anthropic SDK
directly, so the unit tests stay LLM-free.
"""

from __future__ import annotations

from causa.adapters.llm.anthropic_client import AnthropicLLMClient
from causa.adapters.llm.mock_client import MockLLMClient
from causa.config.settings import CausaSettings, LLMProvider
from causa.ports.llm import LLMClient


def make_llm_client(settings: CausaSettings) -> LLMClient:
    """Return the configured LLM adapter."""
    match settings.llm_provider:
        case LLMProvider.ANTHROPIC:
            return AnthropicLLMClient(model=settings.llm_model)
        case LLMProvider.MOCK:
            return MockLLMClient(model=f"mock-{settings.llm_model}")
        case _:  # pragma: no cover
            raise ValueError(f"unknown LLM provider: {settings.llm_provider}")
