"""Protocols (interfaces) for the hexagonal architecture.

Every cross-cutting capability — LLM access, observation persistence, action
scoring, causal estimation, tool execution — is declared here as a
:class:`typing.Protocol`.  Concrete adapters in :mod:`causa.adapters` implement
these protocols; the rest of the system depends only on the abstract types.

This is the lever the dissertation's ablation studies (§J3) pull on:
swapping a port's adapter is a config-level change, not a code-level one.
"""

from __future__ import annotations

from causa.ports.estimator import CausalEstimator, EffectEstimate
from causa.ports.history import ObservationHistory
from causa.ports.llm import LLMClient, LLMMessage, LLMResponse, LLMRole
from causa.ports.scorer import ActionCandidate, ActionScore, ActionScorer
from causa.ports.tool import DebuggingTool, ToolInput, ToolOutput

__all__ = [
    "ActionCandidate",
    "ActionScore",
    "ActionScorer",
    "CausalEstimator",
    "DebuggingTool",
    "EffectEstimate",
    "LLMClient",
    "LLMMessage",
    "LLMResponse",
    "LLMRole",
    "ObservationHistory",
    "ToolInput",
    "ToolOutput",
]
