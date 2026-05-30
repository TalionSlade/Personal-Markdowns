"""Causal Planning Layer (Component 2 of the dissertation).

Three concerns:

- :mod:`causa.planning.dowhy_scorer` — interventional scoring via
  do-calculus on the SCM.  The primary scoring strategy.
- :mod:`causa.planning.llm_scorer` — correlational LLM scoring; cold-start
  fallback and the dissertation's central ablation arm (§J3#2).
- :mod:`causa.planning.warm_start` — synthetic prior observation generator
  (§B8).

The composite :class:`HybridActionScorer` switches between the two based on
history size — implementing the cold-start handling described in §C5.
"""

from __future__ import annotations

from causa.planning.dowhy_scorer import DoWhyActionScorer
from causa.planning.hybrid_scorer import HybridActionScorer
from causa.planning.llm_scorer import LLMActionScorer
from causa.planning.warm_start import SyntheticWarmStart

__all__ = [
    "DoWhyActionScorer",
    "HybridActionScorer",
    "LLMActionScorer",
    "SyntheticWarmStart",
]
