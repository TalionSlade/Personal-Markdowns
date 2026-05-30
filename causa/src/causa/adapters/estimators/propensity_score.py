"""Propensity-score weighted estimator for binary/categorical treatments.

Used in the §J3 ablation #4 to test estimator-sensitivity of headline
results.  Implementation is the IPTW (inverse-probability-of-treatment
weighting) estimator:

.. math::
    \\hat{E}[Y \\mid do(X=t)] = \\frac{
      \\sum_i \\mathbf{1}[X_i = t] \\cdot Y_i / \\hat{p}(t \\mid z_i)
    }{
      \\sum_i \\mathbf{1}[X_i = t] / \\hat{p}(t \\mid z_i)
    }

with :math:`\\hat{p}(t \\mid z)` from a multinomial logistic regression on
the adjustment set.  Reference: Rosenbaum & Rubin 1983; Hernán & Robins
ch. 14.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder

from causa.ports.estimator import EffectEstimate


class PropensityScoreEstimator:
    """IPTW estimator for categorical treatments."""

    name = "propensity_score"

    def __init__(self, *, clip: float = 0.02) -> None:
        """``clip`` truncates propensities to ``[clip, 1-clip]`` to bound IPTW variance."""
        if not 0.0 < clip < 0.5:
            raise ValueError(f"clip must be in (0, 0.5), got {clip}")
        self._clip = clip

    def estimate(
        self,
        *,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        treatment_value: Any,
        adjustment_set: frozenset[str],
    ) -> EffectEstimate:
        if data.empty:
            return EffectEstimate(value=float("nan"), method=self.name, n_observations=0)

        y = pd.to_numeric(data[outcome], errors="coerce").to_numpy()
        t = data[treatment].to_numpy()
        z_cols = sorted(adjustment_set)
        z = data[z_cols] if z_cols else pd.DataFrame(index=data.index)

        cat_cols = [c for c in z_cols if z[c].dtype == object]
        if cat_cols:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            z_mat = enc.fit_transform(z[cat_cols])
            other_cols = [c for c in z_cols if c not in cat_cols]
            if other_cols:
                z_mat = np.hstack([z[other_cols].to_numpy(dtype=float), z_mat])
        elif z_cols:
            z_mat = z.to_numpy(dtype=float)
        else:
            z_mat = np.ones((len(data), 1))

        # Mask NaNs out.
        mask = ~np.isnan(y) & ~np.isnan(z_mat).any(axis=1)
        z_fit, t_fit, y_fit = z_mat[mask], t[mask], y[mask]
        if len(np.unique(t_fit)) < 2 or len(y_fit) < 4:
            return EffectEstimate(
                value=float("nan"), method=self.name, n_observations=int(len(y_fit)),
            )

        propensity_model = LogisticRegression(max_iter=1000).fit(z_fit, t_fit)
        proba = propensity_model.predict_proba(z_fit)
        classes = propensity_model.classes_
        if treatment_value not in classes:
            return EffectEstimate(
                value=float("nan"), method=self.name, n_observations=int(len(y_fit)),
            )
        idx = list(classes).index(treatment_value)
        p_t = np.clip(proba[:, idx], self._clip, 1.0 - self._clip)
        selector = (t_fit == treatment_value).astype(float)
        weights = selector / p_t
        if weights.sum() == 0:
            return EffectEstimate(
                value=float("nan"), method=self.name, n_observations=int(len(y_fit)),
            )
        point = float(np.sum(weights * y_fit) / np.sum(weights))
        # Standard error via the influence-function approximation.
        psi = (weights * y_fit) - point * weights
        se = float(np.sqrt(np.var(psi, ddof=1) / len(psi)))

        return EffectEstimate(
            value=point,
            standard_error=se,
            method=self.name,
            n_observations=int(len(y_fit)),
        )
