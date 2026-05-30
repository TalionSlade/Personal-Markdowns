"""Causal effect estimator protocol.

This port abstracts the *numerical* estimation step — given a graph, an
identified estimand, and observational data, return a point estimate of
:math:`E[Y \\mid do(X=x)]` plus its standard error.  Adapters wrap DoWhy's
linear-regression, propensity-score, and doubly-robust estimators; future
work could plug in EconML or causal-forest estimators (§J3 ablation #6).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd


@dataclass(frozen=True)
class EffectEstimate:
    """A causal-effect estimate with audit info."""

    value: float
    standard_error: float | None = None
    method: str = ""
    n_observations: int = 0
    raw: dict[str, Any] | None = None


@runtime_checkable
class CausalEstimator(Protocol):
    """One numerical estimator behind the planning port."""

    @property
    def name(self) -> str: ...

    def estimate(
        self,
        *,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        treatment_value: Any,
        adjustment_set: frozenset[str],
    ) -> EffectEstimate:
        """Return :math:`E[Y \\mid do(X=\\texttt{treatment\\_value})]`.

        Implementations MUST handle small ``data`` gracefully (return a
        :class:`EffectEstimate` with a large ``standard_error`` rather than
        raising), and MUST be deterministic given a fixed seed.
        """
        ...
