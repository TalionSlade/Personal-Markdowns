"""OpenAI LLM client adapter.

Wraps the official ``openai`` SDK.  Mirrors the Anthropic adapter's interface
so the rest of Causa is provider-agnostic.

Only imported when ``CAUSA_LLM_PROVIDER=openai``.
"""

from __future__ import annotations

import os
from pathlib import Path

from causa.ports.llm import LLMMessage, LLMResponse, LLMRole


def _resolve_api_key(explicit_key: str | None) -> str | None:
    """Return key from arg → os.environ → .env file (in that order)."""
    if explicit_key:
        return explicit_key
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    # pydantic-settings reads .env for CAUSA_* only; load it ourselves
    # so the raw OPENAI_API_KEY line is picked up too.
    for candidate in (Path.cwd() / ".env", Path(__file__).parents[5] / ".env"):
        if candidate.exists():
            for line in candidate.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("OPENAI_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


class OpenAILLMClient:
    """OpenAI-backed LLM client.

    Parameters
    ----------
    model:
        Model identifier (e.g. ``gpt-4o``, ``gpt-4o-mini``).
    api_key:
        OpenAI API key; falls back to the ``OPENAI_API_KEY`` env var.
    temperature:
        Sampling temperature (default 0 = deterministic).
    max_tokens:
        Max tokens per completion.
    """

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ) -> None:
        try:
            from openai import OpenAI  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK not installed; run `pip install openai`"
            ) from exc
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._client = OpenAI(api_key=_resolve_api_key(api_key))

    @property
    def model(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        openai_messages = []
        if system:
            openai_messages.append({"role": "system", "content": system})
        for m in messages:
            if m.role is LLMRole.SYSTEM:
                openai_messages.append({"role": "system", "content": m.content})
            elif m.role is LLMRole.USER:
                openai_messages.append({"role": "user", "content": m.content})
            elif m.role is LLMRole.ASSISTANT:
                openai_messages.append({"role": "assistant", "content": m.content})

        kwargs: dict = dict(
            model=self._model,
            messages=openai_messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        usage = response.usage

        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            finish_reason=choice.finish_reason or "stop",
            raw={"id": response.id},
        )
