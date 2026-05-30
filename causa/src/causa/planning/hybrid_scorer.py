"""Hybrid scorer — LLM until cold-start cleared, then DoWhy.

Implements §C5 of the dissertation: below ``min_history`` real observations,
defer to the LLM; once enough history accumulates, switch to do-calculus.

The strategy boundary is a property of the *scorer*, not the agent — the
agent's decision loop is identical in either regime.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from causa.ports.scorer import ActionCandidate, ActionScore, ActionScorer


class HybridActionScorer(ActionScorer):
    """Compose a cold-start scorer with a steady-state scorer."""

    name: str = "hybrid_cold_to_dowhy"

    def __init__(
        self,
        *,
        cold_start: ActionScorer,
        steady_state: ActionScorer,
        min_history: int,
    ) -> None:
        if min_history < 1:
            raise ValueError(f"min_history must be ≥ 1, got {min_history}")
        self._cold = cold_start
        self._steady = steady_state
        self._threshold = min_history

    @property
    def active_scorer_name(self) -> str:
        return f"cold={self._cold.name}|steady={self._steady.name}|θ={self._threshold}"

    def score(
        self,
        candidates: list[ActionCandidate],
        *,
        state: dict[str, Any],
        history: pd.DataFrame,
    ) -> list[ActionScore]:
        active = self._cold if len(history) < self._threshold else self._steady
        scores = active.score(candidates, state=state, history=history)
        # Annotate rationale with which arm fired (useful in the trace).
        tag = "[cold-start LLM]" if active is self._cold else "[steady-state DoWhy]"
        return [
            ActionScore(
                action=s.action,
                score=s.score,
                confidence=s.confidence,
                rationale=f"{tag} {s.rationale}",
                adjustment_set=s.adjustment_set,
            )
            for s in scores
        ]
