"""Do-calculus action scorer — the heart of Causa.

For each candidate action `a`, this scorer:

1. Identifies the back-door adjustment set for ``P(outcome | do(action=a))``
   using :func:`causa.core.identifiability.identify_effect`.  This is
   *graph-only* — does not touch data — so the result is cached per graph.
2. Runs the configured :class:`CausalEstimator` against the current
   observation history with that adjustment set, returning
   :math:`\\hat{E}[outcome \\mid do(action=a)]`.
3. Wraps the numerical estimate in an :class:`ActionScore` carrying the
   provenance the auditable trace needs (the adjustment set used and the
   numerical method).

This is what makes the agent *interventional* — every score in the agent's
argmax is :math:`P(Y \\mid do(X))`, not :math:`P(Y \\mid X)`.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from causa.core.identifiability import IdentificationResult, identify_effect
from causa.core.scm import SCM
from causa.ports.estimator import CausalEstimator
from causa.ports.scorer import ActionCandidate, ActionScore, ActionScorer


class DoWhyActionScorer(ActionScorer):
    """Interventional scoring via back-door adjustment.

    Parameters
    ----------
    scm:
        The SCM whose graph defines the back-door set and whose ACTION
        variable defines the do-target.
    estimator:
        The numerical estimator (linear regression / propensity score / …).
    """

    name: str = "dowhy_backdoor"

    def __init__(self, *, scm: SCM, estimator: CausalEstimator) -> None:
        scm.validate()
        self._scm = scm
        self._estimator = estimator
        self._action_var = scm.action_variable.name
        self._outcome_var = scm.outcome_variable.name
        self._identification: IdentificationResult | None = None  # cached

    # ── public API ─────────────────────────────────────────────────────────

    def score(
        self,
        candidates: list[ActionCandidate],
        *,
        state: dict[str, Any],
        history: pd.DataFrame,
    ) -> list[ActionScore]:
        ident = self._identify_once()
        if not ident.identifiable:
            # The agent should never have reached here, but fall back to
            # uniform scoring so the loop still produces a usable action.
            return [ActionScore(c, 0.5, rationale="effect not identifiable") for c in candidates]

        # Filter the history to columns the estimator actually needs.
        needed = [self._action_var, self._outcome_var, *sorted(ident.adjustment_set)]
        present_cols = [c for c in needed if c in history.columns]
        sub = history[present_cols] if present_cols else history

        results: list[ActionScore] = []
        for cand in candidates:
            est = self._estimator.estimate(
                data=sub,
                treatment=self._action_var,
                outcome=self._outcome_var,
                treatment_value=cand.name,
                adjustment_set=ident.adjustment_set,
            )
            rationale = (
                f"E[{self._outcome_var} | do({self._action_var}={cand.name})] "
                f"≈ {est.value:.3f} via back-door on Z={sorted(ident.adjustment_set)} "
                f"({est.method}, n={est.n_observations})"
            )
            results.append(ActionScore(
                action=cand,
                score=est.value if est.value == est.value else 0.0,  # NaN-safe
                confidence=(1.0 / est.standard_error) if est.standard_error else None,
                rationale=rationale,
                adjustment_set=ident.adjustment_set,
            ))
        return results

    # ── helpers ────────────────────────────────────────────────────────────

    def _identify_once(self) -> IdentificationResult:
        if self._identification is None:
            self._identification = identify_effect(
                self._scm.graph, self._action_var, self._outcome_var,
            )
        return self._identification

    @property
    def adjustment_set(self) -> frozenset[str]:
        return self._identify_once().adjustment_set
