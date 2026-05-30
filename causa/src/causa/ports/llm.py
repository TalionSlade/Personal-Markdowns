"""LLM client protocol.

The :class:`LLMClient` protocol is the only thing the rest of Causa knows
about LLMs.  Concrete adapters (Anthropic, mock) implement it.  This makes:

- the sensitivity-analysis story in dissertation §L1 a single-adapter swap;
- the unit-test pipeline LLM-free by default (the mock adapter is used);
- the cost story explicit — token accounting lives behind this port.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class LLMRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class LLMMessage:
    role: LLMRole
    content: str


@dataclass(frozen=True)
class LLMResponse:
    """Structured LLM response, including audit info."""

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = "stop"
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@runtime_checkable
class LLMClient(Protocol):
    """The single port through which Causa interacts with any LLM."""

    @property
    def model(self) -> str: ...

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        """Run one chat completion, return the response.

        Implementations MUST set the audit fields on the response
        (``input_tokens``, ``output_tokens``, ``model``).  Streaming is
        deliberately *not* part of this port — Causa is a batch agent.
        """
        ...
