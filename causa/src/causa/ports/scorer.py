"""Action scoring protocol.

This is the *seam* the dissertation's central ablation (§J3 #2) cuts along:

- :class:`causa.planning.dowhy_scorer.DoWhyActionScorer` implements scoring
  via Pearl's do-calculus on a fitted causal model.
- :class:`causa.planning.llm_scorer.LLMActionScorer` implements scoring via
  prompted LLM ranking — the correlational baseline this thesis argues
  against.

Both implement :class:`ActionScorer`; the agent's decision loop is the same
in either case.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class ActionCandidate:
    """One candidate action the agent might take.

    Attributes
    ----------
    name:
        The action identifier (must match a level of the SCM's ACTION
        variable).
    metadata:
        Free-form action description used by the LLM scorer and the
        trace's human-readable rationale.
    """

    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ActionScore:
    """A scored action with the provenance needed for an auditable trace.

    Attributes
    ----------
    action:
        The candidate scored.
    score:
        The interventional expected outcome, :math:`P(Y \\mid do(X=a))` for
        binary Y, or :math:`E[Y \\mid do(X=a)]` for continuous Y.
    confidence:
        Optional confidence in the score (e.g. 1/SE of a regression
        coefficient).  ``None`` means unavailable.
    rationale:
        Free-form human-readable justification — used in the causal trace
        and the human-evaluation explanation study.
    adjustment_set:
        Variables that were adjusted on to identify the effect (back-door
        set).  Empty when no back-door adjustment was required.
    """

    action: ActionCandidate
    score: float
    confidence: float | None = None
    rationale: str = ""
    adjustment_set: frozenset[str] = frozenset()


@runtime_checkable
class ActionScorer(Protocol):
    """The contract every scorer fulfils — including baselines."""

    @property
    def name(self) -> str: ...

    def score(
        self,
        candidates: list[ActionCandidate],
        *,
        state: dict[str, Any],
        history: pd.DataFrame,
    ) -> list[ActionScore]:
        """Rank ``candidates`` given the agent's current ``state`` and history.

        Parameters
        ----------
        candidates:
            The candidate actions to score.
        state:
            The observed state at decision time (values for OBSERVATIONAL
            SCM variables).
        history:
            Append-only observation history; the data feeding DoWhy or the
            LLM prompt.

        Returns
        -------
        One :class:`ActionScore` per candidate, in the *same order* as
        ``candidates`` (the caller takes argmax).
        """
        ...
