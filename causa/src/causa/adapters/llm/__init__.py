"""LLM adapters — Anthropic, Mock — implementing :class:`causa.ports.LLMClient`.

Use :func:`make_llm_client` to construct a client from settings so the rest
of the system never imports adapters directly.
"""

from __future__ import annotations

from causa.adapters.llm.anthropic_client import AnthropicLLMClient
from causa.adapters.llm.factory import make_llm_client
from causa.adapters.llm.mock_client import MockLLMClient

__all__ = ["AnthropicLLMClient", "MockLLMClient", "make_llm_client"]
