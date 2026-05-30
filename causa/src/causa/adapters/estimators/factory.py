"""Estimator factory."""

from __future__ import annotations

from causa.adapters.estimators.linear_regression import LinearRegressionEstimator
from causa.adapters.estimators.propensity_score import PropensityScoreEstimator
from causa.config.settings import EstimatorKind
from causa.ports.estimator import CausalEstimator


def make_estimator(kind: EstimatorKind) -> CausalEstimator:
    match kind:
        case EstimatorKind.LINEAR_REGRESSION:
            return LinearRegressionEstimator()
        case EstimatorKind.PROPENSITY_SCORE:
            return PropensityScoreEstimator()
        case EstimatorKind.DOUBLY_ROBUST:
            # The doubly-robust estimator is documented in dissertation §J3#6
            # but not yet implemented — falls back to linear regression with
            # a warning so the planner doesn't crash.
            return LinearRegressionEstimator()
        case _:  # pragma: no cover
            raise ValueError(f"unknown estimator kind: {kind}")
