"""θ-threshold policies for the Counterfactual Reflection Module.

§D1, D5, J3#6: the threshold θ governs *when* an alternative-outcome
estimate is treated as evidence.  Too low and we update from LLM noise;
too high and we never update.  The ablation §J3#6 varies θ to measure
sensitivity.

Two policies ship:

- :class:`StaticThreshold` — the constant-θ default (the dissertation's
  primary configuration).
- :class:`AdaptiveThreshold` — shrinks θ with history size so the agent
  becomes increasingly receptive to small differences as its causal model
  calibrates.  Mentioned in §K1 risk mitigation.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ThresholdPolicy(Protocol):
    """A policy that returns the current θ given history size."""

    @property
    def name(self) -> str: ...

    def theta(self, n_observations: int) -> float: ...


class StaticThreshold(ThresholdPolicy):
    name = "static"

    def __init__(self, *, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"θ must be in [0, 1], got {value}")
        self._value = value

    def theta(self, n_observations: int) -> float:  # noqa: ARG002
        return self._value


class AdaptiveThreshold(ThresholdPolicy):
    """θ_n = max(θ_min, θ_0 · (1 + n/τ)^{-1}) — shrinks with history size."""

    name = "adaptive"

    def __init__(self, *, theta_0: float, theta_min: float, tau: float) -> None:
        if not 0.0 < theta_min <= theta_0 <= 1.0:
            raise ValueError("require 0 < θ_min ≤ θ_0 ≤ 1")
        if tau <= 0.0:
            raise ValueError(f"τ must be positive, got {tau}")
        self._theta_0 = theta_0
        self._theta_min = theta_min
        self._tau = tau

    def theta(self, n_observations: int) -> float:
        return max(self._theta_min, self._theta_0 / (1.0 + n_observations / self._tau))
