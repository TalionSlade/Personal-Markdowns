"""The proposed system: CausalPlanningAgent.

Wires together the dissertation's three components into the decision loop:

    Component 1 (Causal Graph Extractor)  →  hand-authored debugging SCM
                                              for the primary experiment;
                                              :class:`LLMGraphExtractor` for
                                              novel-domain transfer (§N3).
    Component 2 (Causal Planning Layer)   →  :class:`ActionScorer` —
                                              do-calculus over the SCM.
    Component 3 (Counterfactual Reflection)→  :class:`CounterfactualReflectionModule` —
                                              the online belief-update loop.

The agent is the only place these three components are *composed*.  Every
other module is independently testable.

Why the agent does *not* extend the SCM at runtime:
    The SCM's graph is fixed at experiment start; the reflection module
    updates **only the data** the SCM sees, never the structural edges.
    This is the load-bearing distinction between Causa's notion of
    "learning" and structure-learning approaches (§A6).
"""

from __future__ import annotations

from typing import Any

from causa.agents.base import (
    AgentChoice,
    AgentStep,
    BaseAgent,
    DecisionContext,
)
from causa.domain.tasks import DebuggingTask
from causa.ports.history import ObservationHistory
from causa.ports.scorer import ActionCandidate, ActionScorer
from causa.ports.tool import DebuggingTool
from causa.reflection.counterfactual import CounterfactualReflectionModule


class CausalPlanningAgent(BaseAgent):
    """The dissertation's proposed agent.

    Parameters
    ----------
    scorer:
        Any :class:`ActionScorer` — the canonical configuration is the
        :class:`HybridActionScorer` (cold-start LLM → DoWhy steady-state).
        Swapping this is how §J3#2 ablates the do-calculus arm.
    reflection:
        The :class:`CounterfactualReflectionModule` instance.  May be
        ``None`` to ablate Component 3 (§J3#3).
    history:
        The observation history all three components read/write.
    tools, step_budget, action_variable, outcome_variable:
        Forwarded to :class:`BaseAgent`.
    name:
        Telemetry identifier; defaults to ``"causa.causal"`` but the eval
        runner overrides this for ablation variants so the result rows
        carry a self-describing arm label.
    """

    def __init__(
        self,
        *,
        scorer: ActionScorer,
        reflection: CounterfactualReflectionModule | None,
        history: ObservationHistory,
        tools: dict[str, DebuggingTool],
        step_budget: int,
        action_variable: str,
        outcome_variable: str,
        success_threshold: float = 1.0,
        name: str = "causa.causal",
    ) -> None:
        super().__init__(
            name=name,
            tools=tools,
            step_budget=step_budget,
            action_variable=action_variable,
            outcome_variable=outcome_variable,
            success_threshold=success_threshold,
        )
        self._scorer = scorer
        self._reflection = reflection
        self._history = history

    # ── policy ────────────────────────────────────────────────────────────

    def _choose_action(
        self,
        *,
        candidates: list[ActionCandidate],
        context: DecisionContext,
        task: DebuggingTask,  # noqa: ARG002
    ) -> AgentChoice:
        df = self._history.as_dataframe()
        scored = self._scorer.score(candidates, state=context.state, history=df)
        if not scored:
            raise RuntimeError(
                f"{self.name}: scorer returned no scores for step "
                f"{context.step_index}; refusing to invent an action.",
            )
        best = max(scored, key=lambda s: s.score)
        return AgentChoice(action=best.action, all_scores=scored)

    # ── bookkeeping ───────────────────────────────────────────────────────

    def _after_step(
        self,
        *,
        step: AgentStep,
        task: DebuggingTask,  # noqa: ARG002
        state: dict[str, Any],
    ) -> None:
        # 1. Append the real observation to the history (the data DoWhy
        #    will see at the next step).
        self._history.append(step.observation)

        # 2. Run the reflection module.  This is the *novel* step: for each
        #    non-chosen alternative, ask the LLM for an estimated
        #    counterfactual outcome.  Synthesise observation rows for the
        #    alternatives whose median estimate sufficiently exceeds the
        #    observed outcome, and append them to history.
        if self._reflection is None:
            return

        chosen_score = next(
            (s for s in step.scores if s.action.name == step.chosen_action), None,
        )
        if chosen_score is None:
            return
        alternatives = [s.action for s in step.scores if s.action.name != step.chosen_action]

        update = self._reflection.reflect(
            state=state,
            chosen_action=chosen_score.action,
            observed_outcome=step.outcome,
            alternatives=alternatives,
            n_observations=self._history.n_observations,
        )

        if update.triggered and update.synthetic_rows:
            self._history.append_many(update.synthetic_rows)
        step.reflection = self._format_reflection_note(update)

    # ── helpers ───────────────────────────────────────────────────────────

    def _history_size(self) -> int:
        return self._history.n_observations

    @staticmethod
    def _format_reflection_note(update: Any) -> str:
        """One-line trace annotation summarising the reflection update."""
        if not update.estimates:
            return "reflection: skipped (no alternatives)"
        if not update.triggered:
            return (
                f"reflection: no update (θ={update.threshold_used:.3f}, "
                f"max Δ={_max_delta(update):.3f})"
            )
        return (
            f"reflection: +{len(update.synthetic_rows)} synthetic row(s); "
            f"θ={update.threshold_used:.3f}; max Δ={_max_delta(update):.3f}"
        )


def _max_delta(update: Any) -> float:
    if not update.estimates:
        return 0.0
    return max(est.median - update.observed_outcome for est in update.estimates)
