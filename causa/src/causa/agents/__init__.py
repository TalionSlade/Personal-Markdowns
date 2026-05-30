"""Decision-loop agents — the dissertation's evaluation lineup.

The :class:`CausalPlanningAgent` is the proposed system (Components 1+2+3).
The three baselines (:class:`ReActAgent`, :class:`ChainOfThoughtAgent`,
:class:`NoMemoryAgent`) are the comparators in §J of the FAQ.

All four agents share a single contract — :class:`BaseAgent.run` — so the
eval runner can swap them as drop-in arms of the experiment.  This is the
shape that makes the ablation study a config-level switch rather than a
code-level fork.
"""

from __future__ import annotations

from causa.agents.base import (
    AgentResult,
    AgentStep,
    BaseAgent,
    DecisionContext,
    DecisionTrace,
)
from causa.agents.causal import CausalPlanningAgent
from causa.agents.chain_of_thought import ChainOfThoughtAgent
from causa.agents.no_memory import NoMemoryAgent
from causa.agents.react import ReActAgent

__all__ = [
    "AgentResult",
    "AgentStep",
    "BaseAgent",
    "CausalPlanningAgent",
    "ChainOfThoughtAgent",
    "DecisionContext",
    "DecisionTrace",
    "NoMemoryAgent",
    "ReActAgent",
]
