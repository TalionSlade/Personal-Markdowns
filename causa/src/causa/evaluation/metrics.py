"""Evaluation metrics.

The dissertation's reportable quantities (§F4):

- **Success rate**: fraction of tasks with :class:`TaskOutcome.SOLVED`.
- **Mean outcome**: average of ``final_outcome_score`` across tasks.
  For binary outcomes this equals success rate; for continuous outcomes
  (e.g. patch quality) it captures graceful degradation.
- **Mean steps to success**: among solved tasks only, average steps used.
  Surfaces sample-efficiency improvements that don't move the success
  bar.
- **Reflection trigger rate**: per-step rate at which the Counterfactual
  Reflection Module wrote synthetic rows.  Only meaningful for the causal
  agent — for baselines this is identically zero.
- **Identifiability rate**: fraction of steps at which the DoWhy scorer
  was able to identify the effect.  Diagnostic; should be 1.0 with the
  fixed debugging SCM.
- **Mean confidence**: average ``ActionScore.confidence`` (when set);
  used for calibration analyses.

Metrics are computed *post hoc* from :class:`AgentResult` rows so the
agent loop need not be metric-aware.  This keeps the agent's contract
clean and lets us add metrics later without touching the agents.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from statistics import fmean

from causa.agents.base import AgentResult, AgentStep, DecisionTrace
from causa.domain.tasks import TaskOutcome


@dataclass(frozen=True)
class MetricRow:
    """Per-arm row of headline metrics.

    Attributes
    ----------
    agent_name:
        Identifier of the agent (or arm) the metrics describe.
    n_tasks:
        Number of task results aggregated.
    success_rate:
        Fraction of tasks with :class:`TaskOutcome.SOLVED`.
    mean_outcome:
        Mean ``final_outcome_score`` across all tasks (None for empty).
    mean_steps_to_success:
        Mean step count among solved tasks; ``None`` if none solved.
    reflection_trigger_rate:
        Per-step rate at which the reflection module appended synthetic
        rows; ``None`` if never observed (baselines).
    """

    agent_name: str
    n_tasks: int
    success_rate: float
    mean_outcome: float | None
    mean_steps_to_success: float | None
    reflection_trigger_rate: float | None


class EvaluationMetrics:
    """Pure functions over :class:`AgentResult` lists."""

    @staticmethod
    def success_rate(results: Iterable[AgentResult]) -> float:
        rs = list(results)
        if not rs:
            return 0.0
        return sum(1 for r in rs if r.outcome is TaskOutcome.SOLVED) / len(rs)

    @staticmethod
    def mean_outcome(results: Iterable[AgentResult]) -> float | None:
        rs = [r for r in results if r.final_outcome_score is not None]
        if not rs:
            return None
        return float(fmean(r.final_outcome_score for r in rs))  # type: ignore[arg-type]

    @staticmethod
    def mean_steps_to_success(results: Iterable[AgentResult]) -> float | None:
        solved = [r for r in results if r.outcome is TaskOutcome.SOLVED]
        if not solved:
            return None
        return float(fmean(r.trace.n_steps for r in solved))

    @staticmethod
    def reflection_trigger_rate(results: Iterable[AgentResult]) -> float | None:
        total_steps = 0
        triggered_steps = 0
        observed = False
        for r in results:
            for step in r.trace.steps:
                total_steps += 1
                if step.reflection and "synthetic row" in step.reflection:
                    triggered_steps += 1
                if step.reflection.startswith("reflection:"):
                    observed = True
        if not observed or total_steps == 0:
            return None
        return triggered_steps / total_steps

    @staticmethod
    def mean_confidence(results: Iterable[AgentResult]) -> float | None:
        confidences: list[float] = []
        for r in results:
            for step in r.trace.steps:
                for s in step.scores:
                    if s.confidence is not None and s.action.name == step.chosen_action:
                        confidences.append(float(s.confidence))
        if not confidences:
            return None
        return float(fmean(confidences))

    @staticmethod
    def identifiability_rate(results: Iterable[AgentResult]) -> float | None:
        total = 0
        identified = 0
        for r in results:
            for step in r.trace.steps:
                for s in step.scores:
                    if s.action.name != step.chosen_action:
                        continue
                    total += 1
                    if "via back-door" in s.rationale:
                        identified += 1
                    break
        if total == 0:
            return None
        return identified / total


def summarize_results(
    *,
    agent_name: str,
    results: Iterable[AgentResult],
) -> MetricRow:
    """Compute a :class:`MetricRow` for one agent's results."""
    rs = list(results)
    return MetricRow(
        agent_name=agent_name,
        n_tasks=len(rs),
        success_rate=EvaluationMetrics.success_rate(rs),
        mean_outcome=EvaluationMetrics.mean_outcome(rs),
        mean_steps_to_success=EvaluationMetrics.mean_steps_to_success(rs),
        reflection_trigger_rate=EvaluationMetrics.reflection_trigger_rate(rs),
    )


def _all_steps(traces: Iterable[DecisionTrace]) -> Iterator[AgentStep]:
    for t in traces:
        yield from t.steps
