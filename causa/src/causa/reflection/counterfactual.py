"""Counterfactual Reflection Module — the dissertation's novelty.

The module's contract:

    Given: state, chosen action `a`, observed outcome `y`, alternative
           actions `A'`, the LLM client.
    Do:    For each `a' ∈ A'`, query the LLM `K` times for an estimated
           outcome had `a'` been chosen.  Take the median across samples
           (variance-reducing, §D3).  Compare against `y`.  For every
           alternative whose median exceeds `y` by more than θ, emit a
           **synthetic counterfactual observation** that the agent appends
           to history.

Why this is *online causal-belief update* (and not "just better prompting"):
- The synthetic observation lands in the DataFrame that DoWhy reads at the
  next decision step (§D2);
- The update is *durable* — it compounds over decisions and episodes;
- The update encodes a *structured* counterfactual claim about an SCM
  variable, not free-form rationalisation.

Failure modes are explicit:
- LLM returns non-JSON → discard, log, no update (§L4);
- All counterfactual estimates ≤ observed → no update (the correct
  behaviour, not a failure).
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any

from causa.ports.llm import LLMClient, LLMMessage, LLMRole
from causa.ports.scorer import ActionCandidate
from causa.reflection.threshold import ThresholdPolicy


SYSTEM_PROMPT = (
    "You are the counterfactual-estimation component of a debugging agent. "
    "Given a state, the action that was taken, the observed outcome score, "
    "and a list of alternative actions, produce a JSON array of objects "
    '{"action": str, "estimated_outcome": float in [0,1]} giving the outcome '
    "score you would expect if each alternative had been chosen instead, "
    "holding all other context fixed.  Be calibrated; do not just say all "
    "alternatives are better."
)


@dataclass(frozen=True)
class CounterfactualEstimate:
    """The LLM's estimate for one alternative action.

    Attributes
    ----------
    action:
        Name of the alternative action being evaluated.
    estimates:
        Raw per-sample estimates returned by the LLM (length = ``samples``).
    median:
        Median of ``estimates`` — the policy-relevant point estimate.
    """

    action: str
    estimates: tuple[float, ...]
    median: float


@dataclass
class ReflectionUpdate:
    """A single belief-update operation.

    Attributes
    ----------
    triggered:
        Whether the threshold test fired.
    chosen_action:
        Action the agent actually took.
    observed_outcome:
        Numeric outcome observed for the chosen action.
    estimates:
        Counterfactual estimates for every alternative.
    threshold_used:
        The θ in force at update time.
    synthetic_rows:
        Rows that will be appended to the observation history if
        ``triggered`` is True.  One row per triggering alternative.
    """

    triggered: bool
    chosen_action: str
    observed_outcome: float
    estimates: list[CounterfactualEstimate] = field(default_factory=list)
    threshold_used: float = 0.0
    synthetic_rows: list[dict[str, Any]] = field(default_factory=list)


class CounterfactualReflectionModule:
    """The dissertation's novel online causal-belief update loop."""

    def __init__(
        self,
        *,
        llm: LLMClient,
        threshold: ThresholdPolicy,
        samples: int,
        action_variable: str,
        outcome_variable: str,
    ) -> None:
        if samples < 1:
            raise ValueError(f"samples must be ≥ 1, got {samples}")
        self._llm = llm
        self._threshold = threshold
        self._samples = samples
        self._action_var = action_variable
        self._outcome_var = outcome_variable

    # ── public API ─────────────────────────────────────────────────────────

    def reflect(
        self,
        *,
        state: dict[str, Any],
        chosen_action: ActionCandidate,
        observed_outcome: float,
        alternatives: list[ActionCandidate],
        n_observations: int,
    ) -> ReflectionUpdate:
        """Run one reflection step; return the (possibly empty) update."""
        if not alternatives:
            return ReflectionUpdate(
                triggered=False,
                chosen_action=chosen_action.name,
                observed_outcome=observed_outcome,
            )

        estimates = self._gather_estimates(
            state=state, chosen_action=chosen_action, observed_outcome=observed_outcome,
            alternatives=alternatives,
        )
        theta = self._threshold.theta(n_observations)
        synthetic_rows: list[dict[str, Any]] = []
        triggered = False
        for est in estimates:
            if est.median - observed_outcome > theta:
                triggered = True
                synthetic_rows.append(self._synthetic_row(state, est))

        return ReflectionUpdate(
            triggered=triggered,
            chosen_action=chosen_action.name,
            observed_outcome=observed_outcome,
            estimates=estimates,
            threshold_used=theta,
            synthetic_rows=synthetic_rows,
        )

    # ── helpers ────────────────────────────────────────────────────────────

    def _gather_estimates(
        self,
        *,
        state: dict[str, Any],
        chosen_action: ActionCandidate,
        observed_outcome: float,
        alternatives: list[ActionCandidate],
    ) -> list[CounterfactualEstimate]:
        prompt = self._render_prompt(
            state=state, chosen_action=chosen_action,
            observed_outcome=observed_outcome, alternatives=alternatives,
        )
        per_action_samples: dict[str, list[float]] = {a.name: [] for a in alternatives}
        for _ in range(self._samples):
            response = self._llm.complete(
                messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
                system=SYSTEM_PROMPT,
                json_mode=True,
                max_tokens=512,
            )
            for action, est in self._parse(response.content).items():
                if action in per_action_samples:
                    per_action_samples[action].append(est)

        out: list[CounterfactualEstimate] = []
        for a in alternatives:
            samples = per_action_samples.get(a.name, [])
            if not samples:
                continue
            out.append(CounterfactualEstimate(
                action=a.name,
                estimates=tuple(samples),
                median=float(statistics.median(samples)),
            ))
        return out

    def _synthetic_row(self, state: dict[str, Any], est: CounterfactualEstimate) -> dict[str, Any]:
        """Build the row that lands in the observation history.

        We preserve the observed state's input variables, set the action
        variable to the alternative, and the outcome to the median
        counterfactual estimate.  Mediators inherit the chosen state's values
        — a deliberate simplification documented in §D3 of the dissertation.
        """
        row = dict(state)
        row[self._action_var] = est.action
        row[self._outcome_var] = est.median
        return row

    @staticmethod
    def _render_prompt(
        *,
        state: dict[str, Any],
        chosen_action: ActionCandidate,
        observed_outcome: float,
        alternatives: list[ActionCandidate],
    ) -> str:
        alt_block = "\n".join(
            f'- {{"action": "{a.name}", "description": "{a.metadata.get("description", "")}"}}'
            for a in alternatives
        )
        return (
            "[CAUSA::counterfactual]\n\n"
            f"State: {json.dumps(state, default=str)}\n"
            f"Action taken: {chosen_action.name}\n"
            f"Observed outcome score: {observed_outcome:.3f}\n\n"
            f"Alternative actions:\n{alt_block}\n\n"
            "For each alternative, estimate the outcome that would have "
            "resulted had it been chosen instead.  Return a JSON array of "
            '{"action": str, "estimated_outcome": float in [0,1]}.'
        )

    @staticmethod
    def _parse(content: str) -> dict[str, float]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        result: dict[str, float] = {}
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict) and "action" in item and "estimated_outcome" in item:
                    try:
                        result[item["action"]] = float(item["estimated_outcome"])
                    except (TypeError, ValueError):
                        continue
        return result
