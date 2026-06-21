"""Linear-regression back-door estimator.

The dissertation's default estimator (§B2).  Fast, well-understood, and
works with the small per-task history sizes typical at decision time.

We avoid DoWhy's heavyweight ``CausalModel`` here — the back-door
adjustment formula reduces to an OLS fit with the treatment plus the
adjustment set as covariates, then prediction at the target treatment
value.  This keeps the inner loop on the order of milliseconds per call.

Reference: Pearl 2009 §3.3.1; Hernán & Robins, *Causal Inference: What If*,
ch. 13.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

from causa.ports.estimator import EffectEstimate


class LinearRegressionEstimator:
    """Back-door adjustment via OLS.

    For binary or continuous outcomes, the *adjusted* expected outcome at
    ``do(treatment = t)`` is the average prediction across the empirical
    joint of the adjustment set evaluated with ``treatment = t``:

    .. math::
        \\hat{E}[Y \\mid do(X=t)] \\;=\\;
        \\frac{1}{n} \\sum_{i=1}^{n} \\hat{f}(t, z_i)

    where :math:`\\hat{f}` is the fitted regression and :math:`z_i` ranges
    over the observed rows.  This is the standardised-prediction (a.k.a.
    g-computation) formulation of back-door adjustment.
    """

    name = "linear_regression"

    def estimate(
        self,
        *,
        data: pd.DataFrame,
        treatment: str,
        outcome: str,
        treatment_value: Any,
        adjustment_set: frozenset[str],
    ) -> EffectEstimate:
        if treatment not in data.columns or outcome not in data.columns:
            raise KeyError(
                f"data missing required columns: treatment={treatment!r}, "
                f"outcome={outcome!r}"
            )
        if data.empty:
            return EffectEstimate(value=float("nan"), method=self.name, n_observations=0)

        features = [treatment, *sorted(adjustment_set)]
        x_raw = data[features].copy()
        # Map canonical binary labels to numeric before coercion.
        outcome_col = data[outcome].replace({"pass": 1, "fail": 0, "yes": 1, "no": 0})
        y = pd.to_numeric(outcome_col, errors="coerce").to_numpy()

        # One-hot encode any non-numeric feature for OLS.
        # pd 2.x uses pd.StringDtype (not object) for string columns.
        cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(x_raw[c])]
        num_cols = [c for c in features if c not in cat_cols]
        if cat_cols:
            enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            cat_block = enc.fit_transform(x_raw[cat_cols])
            num_block = x_raw[num_cols].to_numpy() if num_cols else np.empty((len(x_raw), 0))
            x = np.hstack([num_block, cat_block])
        else:
            enc = None
            x = x_raw.to_numpy(dtype=float)

        # NaN-drop alignment.
        mask = ~np.isnan(y) & ~np.isnan(x).any(axis=1)
        x_fit, y_fit = x[mask], y[mask]
        if len(y_fit) < 2:
            return EffectEstimate(
                value=float("nan"), method=self.name, n_observations=int(len(y_fit)),
            )

        model = LinearRegression().fit(x_fit, y_fit)

        # Build the "do(treatment=value)" counterfactual row matrix.
        x_do = x_raw.copy()
        x_do[treatment] = treatment_value
        if cat_cols and enc is not None:
            cat_do = enc.transform(x_do[cat_cols])
            num_do = x_do[num_cols].to_numpy() if num_cols else np.empty((len(x_do), 0))
            x_pred = np.hstack([num_do, cat_do])
        else:
            x_pred = x_do.to_numpy(dtype=float)

        preds = model.predict(x_pred)
        point = float(np.mean(preds))

        # Approximate SE via bootstrap-free residual variance.
        residuals = y_fit - model.predict(x_fit)
        if len(residuals) > len(features):
            se = float(np.sqrt(np.var(residuals, ddof=1) / len(residuals)))
        else:
            se = None

        return EffectEstimate(
            value=point,
            standard_error=se,
            method=self.name,
            n_observations=int(len(y_fit)),
        )
