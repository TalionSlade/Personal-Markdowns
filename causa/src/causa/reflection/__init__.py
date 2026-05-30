"""Counterfactual Reflection Module — Component 3.

This is the **novel contribution** of the dissertation (Objective 3,
§D in the FAQ).  The module sits *after* the agent executes an action and
*before* the next decision, and updates the SCM's observation history when
the LLM's counterfactual estimate of an alternative action's outcome
materially exceeds the observed outcome of the chosen action.

The update is structured (it lands in the DataFrame DoWhy reads), durable
(it compounds across decisions and episodes), and high-resolution (it
fires at every step, not at episode boundaries) — these are the three
ways the module differs from Reflexion (§D2).
"""

from __future__ import annotations

from causa.reflection.counterfactual import (
    CounterfactualEstimate,
    CounterfactualReflectionModule,
    ReflectionUpdate,
)
from causa.reflection.threshold import StaticThreshold, ThresholdPolicy

__all__ = [
    "CounterfactualEstimate",
    "CounterfactualReflectionModule",
    "ReflectionUpdate",
    "StaticThreshold",
    "ThresholdPolicy",
]
