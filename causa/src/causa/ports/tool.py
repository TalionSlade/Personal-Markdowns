"""Debugging-tool protocol.

The agent picks tools (its do-variable values) one at a time.  Each tool is
modelled as a :class:`DebuggingTool` — a unit of *executable causal action*.
Concrete adapters in :mod:`causa.adapters.tools` implement the contract:
stubbed adapters return realistic synthetic outputs, real adapters call
``pytest``/``ripgrep``/the LSP.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True)
class ToolInput:
    """A typed invocation argument set for one tool.

    The schema is tool-specific; we keep it as a dict to remain JSON-trace-
    friendly and to avoid a combinatorial explosion of typed inputs.
    """

    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutput:
    """The structured outcome of a single tool invocation.

    Attributes
    ----------
    success:
        Whether the tool executed without error.
    payload:
        Tool-specific result blob (logs, AST, search hits, …).
    information_score:
        Real value in [0, 1] estimating how much *new* information this
        invocation produced.  Feeds the SCM's ``information_gained`` node.
    elapsed_seconds:
        Wall-clock cost, included so the eval suite can report compute.
    """

    success: bool
    payload: dict[str, Any] = field(default_factory=dict)
    information_score: float = 0.0
    elapsed_seconds: float = 0.0


@runtime_checkable
class DebuggingTool(Protocol):
    """One executable tool in the agent's action space."""

    @property
    def name(self) -> str: ...

    @property
    def description(self) -> str:
        """Natural-language description used in LLM prompts and traces."""
        ...

    def __call__(self, input: ToolInput, *, state: dict[str, Any]) -> ToolOutput:
        """Run the tool against the current task state."""
        ...
