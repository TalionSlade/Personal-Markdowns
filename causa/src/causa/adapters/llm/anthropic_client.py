"""Anthropic LLM client adapter.

Wraps the official ``anthropic`` SDK.  Includes:

- prompt caching support (``cache_control`` blocks) for the system prompt
  and the SCM description — both of which are reused across many calls and
  drive most of the per-task cost (§L2 in the dissertation);
- token accounting written through to :class:`LLMResponse` so the eval
  harness can produce the compute-matched comparison table required by
  reviewers (§H3);
- structured error wrapping so transient API errors surface as
  :class:`LLMClientError` and never as raw HTTP exceptions in the agent
  loop.

The adapter is **only** imported when ``CAUSA_LLM_PROVIDER=anthropic`` —
``import anthropic`` is deferred to construction time so unit tests never
need the SDK installed.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

from causa.ports.llm import LLMMessage, LLMResponse, LLMRole

if TYPE_CHECKING:  # pragma: no cover
    pass


class LLMClientError(RuntimeError):
    """Raised on any LLM call failure that can't be retried locally."""


class AnthropicLLMClient:
    """Anthropic-backed LLM client.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``claude-3-5-sonnet-latest``).
    api_key:
        Anthropic API key; falls back to the ``ANTHROPIC_API_KEY`` env var.
    enable_cache:
        Whether to wrap the system prompt in a ``cache_control`` block.
        Strongly recommended — dramatically reduces per-task token cost
        for repeated SCM-prompting (the system prompt is large).
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        enable_cache: bool = True,
    ) -> None:
        try:
            import anthropic  # noqa: PLC0415  (deferred import is intentional)
        except ImportError as e:  # pragma: no cover
            raise LLMClientError(
                "anthropic SDK not installed; run `pip install anthropic`"
            ) from e
        self._anthropic = anthropic
        self._model = model
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
        )
        self._enable_cache = enable_cache

    @property
    def model(self) -> str:
        return self._model

    # ── public API ─────────────────────────────────────────────────────────

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        anthropic_msgs = [
            {"role": _role_to_anthropic(m.role), "content": m.content}
            for m in messages
            if m.role is not LLMRole.SYSTEM
        ]
        system_param = self._encode_system(system, json_mode)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_param if system_param else None,
                messages=anthropic_msgs,
            )
        except self._anthropic.APIError as e:
            raise LLMClientError(f"Anthropic API error: {e}") from e

        text = "".join(
            block.text for block in response.content
            if getattr(block, "type", None) == "text"
        )
        usage = response.usage
        return LLMResponse(
            content=text,
            model=response.model,
            input_tokens=getattr(usage, "input_tokens", 0),
            output_tokens=getattr(usage, "output_tokens", 0),
            finish_reason=response.stop_reason or "stop",
            raw={"id": response.id},
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _encode_system(self, system: str | None, json_mode: bool) -> list[dict[str, Any]] | None:
        if not system and not json_mode:
            return None
        blocks: list[dict[str, Any]] = []
        if system:
            block: dict[str, Any] = {"type": "text", "text": system}
            if self._enable_cache:
                block["cache_control"] = {"type": "ephemeral"}
            blocks.append(block)
        if json_mode:
            blocks.append({
                "type": "text",
                "text": "Respond with a single JSON object.  No prose.",
            })
        return blocks


def _role_to_anthropic(role: LLMRole) -> str:
    match role:
        case LLMRole.USER:
            return "user"
        case LLMRole.ASSISTANT:
            return "assistant"
        case _:
            raise ValueError(f"unsupported role for Anthropic: {role}")
