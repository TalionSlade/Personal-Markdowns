"""Base agent contract.

Every agent the eval suite runs implements :class:`BaseAgent`.  The contract
is intentionally narrow:

- :meth:`BaseAgent.run` takes a :class:`DebuggingTask` and returns an
  :class:`AgentResult`;
- All policy variation (causal vs ReAct vs CoT vs no-memory) lives inside
  the subclass's :meth:`_choose_action` and :meth:`_after_step` hooks.

The split lets the **outer loop** be shared (step budget, state transition,
trace emission) while the **policy** is the variable under study.  This is
the seam the dissertation's §J3 ablations cut along, and it's what makes
the evaluation result a clean ceteris-paribus comparison.

The agent does *not* know about telemetry sinks or evaluation scoring —
both happen at higher layers.  This keeps the agent unit-testable without
fixtures for log files or metrics.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from causa.domain.tasks import DebuggingTask, TaskOutcome
from causa.ports.scorer import ActionCandidate, ActionScore
from causa.ports.tool import DebuggingTool, ToolInput, ToolOutput


# ─── outcome bookkeeping ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class DecisionContext:
    """Frozen snapshot of the world at the moment a decision is made.

    Captured *before* the action fires so the trace records what the agent
    knew when it chose — not what it knew after.

    Attributes
    ----------
    step_index:
        0-based decision index in the current task.
    state:
        Observed values for SCM observational/mediator variables.
    history_size:
        Number of rows in the observation history at decision time.
        Useful for diagnosing cold-start regime in the trace.
    """

    step_index: int
    state: dict[str, Any]
    history_size: int


@dataclass
class AgentStep:
    """One complete decision–act–observe iteration.

    Attributes
    ----------
    context:
        The :class:`DecisionContext` snapshot taken before choosing.
    scores:
        The full ranking the scorer produced, in the order returned.
        Preserved (not just the argmax) so the trace can show *what was
        considered* — §G4 explainability requirement.
    chosen_action:
        The action the agent selected.  Matches a tool name in the
        registry.
    tool_output:
        The :class:`ToolOutput` from invoking the chosen tool.
    outcome:
        Numeric outcome score for the step (the value of the SCM's
        outcome variable post-step).
    observation:
        The full SCM-shaped row appended to the observation history after
        this step.
    reflection:
        Free-form per-step annotation (used by ReAct and CoT to record
        thoughts; empty for the no-memory baseline).
    elapsed_seconds:
        Wall-clock cost of the step, end to end.
    """

    context: DecisionContext
    scores: list[ActionScore] = field(default_factory=list)
    chosen_action: str = ""
    tool_output: ToolOutput | None = None
    outcome: float = 0.0
    observation: dict[str, Any] = field(default_factory=dict)
    reflection: str = ""
    elapsed_seconds: float = 0.0


@dataclass
class DecisionTrace:
    """The agent's step-by-step record of one task attempt.

    A :class:`DecisionTrace` is the structured artifact that powers both
    debugging-the-agent (the developer view) and explainability research
    (the §G evaluation).  Each step is a complete causal episode.
    """

    task_id: str
    agent_name: str
    steps: list[AgentStep] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def append(self, step: AgentStep) -> None:
        self.steps.append(step)

    @property
    def n_steps(self) -> int:
        return len(self.steps)


@dataclass
class AgentResult:
    """Terminal result of running an agent on one task.

    Attributes
    ----------
    task_id:
        The task that was attempted.
    agent_name:
        Identifier of the agent (used to slice eval metrics).
    outcome:
        :class:`TaskOutcome` — solved / failed / budget exceeded / error.
    final_outcome_score:
        Numeric outcome at the terminal step (matches the SCM outcome
        variable's value).  ``None`` if no step ran.
    trace:
        The full step-by-step :class:`DecisionTrace`.  Optional only so
        ad-hoc callers can drop it; the eval runner always keeps it.
    """

    task_id: str
    agent_name: str
    outcome: TaskOutcome
    final_outcome_score: float | None
    trace: DecisionTrace
    error: str | None = None


# ─── base class ──────────────────────────────────────────────────────────────


class BaseAgent(ABC):
    """Shared scaffolding: decision loop, step bookkeeping, termination.

    Subclasses override the four hooks:

    - :meth:`_choose_action` — the policy under study;
    - :meth:`_after_step` — bookkeeping subclasses need (e.g. CoT's
      thought log, the Causal agent's reflection update);
    - :meth:`_initial_state` (optional) — adjust the initial observed state
      before the loop begins;
    - :meth:`_terminate_early` (optional) — agent-specific early stop
      criteria beyond outcome ≥ success threshold.

    Parameters
    ----------
    name:
        Stable identifier for telemetry and result aggregation.
    tools:
        The registered debugging tools, keyed by ``tool.name``.  The
        candidate set the scorer is shown is derived from this.
    step_budget:
        Maximum tool invocations per task.  Matches the SWE-bench
        protocol's bound.
    action_variable:
        Name of the SCM ACTION variable (e.g. ``"tool_selected"``).
    outcome_variable:
        Name of the SCM OUTCOME variable (e.g. ``"tests_passed"``).
    success_threshold:
        Outcome score at or above which the task is considered solved.
        Default ``1.0`` for binary outcomes that map pass→1.0.
    """

    def __init__(
        self,
        *,
        name: str,
        tools: dict[str, DebuggingTool],
        step_budget: int,
        action_variable: str,
        outcome_variable: str,
        success_threshold: float = 1.0,
    ) -> None:
        if step_budget < 1:
            raise ValueError(f"step_budget must be ≥ 1, got {step_budget}")
        if not tools:
            raise ValueError("tools registry must contain at least one tool")
        self.name = name
        self._tools = tools
        self._step_budget = step_budget
        self._action_var = action_variable
        self._outcome_var = outcome_variable
        self._success_threshold = success_threshold

    # ── public entry point ────────────────────────────────────────────────

    def run(self, task: DebuggingTask) -> AgentResult:
        """Run one task end-to-end; return the terminal :class:`AgentResult`."""
        trace = DecisionTrace(task_id=task.task_id, agent_name=self.name)
        state = self._initial_state(task)
        candidates = self._candidates_from_tools()
        final_outcome: float | None = None
        terminal = TaskOutcome.BUDGET_EXCEEDED
        error: str | None = None

        try:
            for step_index in range(self._step_budget):
                t0 = time.perf_counter()
                ctx = self._snapshot_context(step_index=step_index, state=state)
                choice = self._choose_action(candidates=candidates, context=ctx, task=task)
                tool = self._tools[choice.action.name]
                tool_output = tool(ToolInput(args={}), state=state)
                observation, outcome = self._record_observation(
                    state=state, choice=choice, tool_output=tool_output,
                )
                step = AgentStep(
                    context=ctx,
                    scores=choice.all_scores,
                    chosen_action=choice.action.name,
                    tool_output=tool_output,
                    outcome=outcome,
                    observation=observation,
                    elapsed_seconds=time.perf_counter() - t0,
                )
                self._after_step(step=step, task=task, state=state)
                trace.append(step)
                final_outcome = outcome

                if outcome >= self._success_threshold:
                    terminal = TaskOutcome.SOLVED
                    break
                if self._terminate_early(step=step, task=task):
                    terminal = TaskOutcome.FAILED
                    break
                # Carry observation forward as the next step's input state
                state = self._next_state(prev_state=state, observation=observation)
        except Exception as exc:  # pragma: no cover - defensive
            terminal = TaskOutcome.ERROR
            error = f"{type(exc).__name__}: {exc}"
            trace.notes.append(error)

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            outcome=terminal,
            final_outcome_score=final_outcome,
            trace=trace,
            error=error,
        )

    # ── hooks for subclasses ──────────────────────────────────────────────

    @abstractmethod
    def _choose_action(
        self,
        *,
        candidates: list[ActionCandidate],
        context: DecisionContext,
        task: DebuggingTask,
    ) -> "AgentChoice":
        """Return the chosen action plus the full ranking for the trace."""

    def _after_step(
        self,
        *,
        step: AgentStep,
        task: DebuggingTask,  # noqa: ARG002
        state: dict[str, Any],  # noqa: ARG002
    ) -> None:
        """Hook for subclasses; default no-op."""
        # Default: nothing.  Subclasses override to update history,
        # reflect, or accumulate scratchpads.
        del step  # avoid unused warnings under strict configs

    def _initial_state(self, task: DebuggingTask) -> dict[str, Any]:
        return dict(task.initial_state)

    def _terminate_early(
        self,
        *,
        step: AgentStep,  # noqa: ARG002
        task: DebuggingTask,  # noqa: ARG002
    ) -> bool:
        return False

    def _next_state(
        self,
        *,
        prev_state: dict[str, Any],
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Build the input state for the next decision step.

        The default carries the previous observational inputs forward and
        layers in the new mediators/outcome.  Subclasses can override to
        implement non-default state transitions (e.g. for the no-memory
        agent that wipes mediators).
        """
        nxt = dict(prev_state)
        nxt.update(observation)
        return nxt

    # ── shared helpers ────────────────────────────────────────────────────

    def _candidates_from_tools(self) -> list[ActionCandidate]:
        """Build the canonical candidate list from the tool registry."""
        return [
            ActionCandidate(name=t.name, metadata={"description": t.description})
            for t in self._tools.values()
        ]

    def _snapshot_context(
        self,
        *,
        step_index: int,
        state: dict[str, Any],
    ) -> DecisionContext:
        return DecisionContext(
            step_index=step_index,
            state=dict(state),
            history_size=self._history_size(),
        )

    def _history_size(self) -> int:
        """Override in subclasses that own a history; default zero."""
        return 0

    def _record_observation(
        self,
        *,
        state: dict[str, Any],
        choice: "AgentChoice",
        tool_output: ToolOutput,
    ) -> tuple[dict[str, Any], float]:
        """Compose the SCM-shaped row appended after this step.

        The base implementation makes only minimal SCM assumptions:
        - the chosen action lands in ``self._action_var``;
        - the tool's ``information_score`` lands in ``information_gained``
          if that variable exists;
        - the outcome is derived monotonically from ``information_score``
          for the default mediator-light SCMs.

        Subclasses can override for richer mechanism layering.
        """
        obs = dict(state)
        obs[self._action_var] = choice.action.name
        if "information_gained" in obs or "information_gained" not in state:
            obs["information_gained"] = round(tool_output.information_score, 3)
        # Derive a simple monotone outcome surrogate; the real SCM mechanism
        # lives in :mod:`causa.domain` and is layered in by the eval runner
        # via the optional outcome_mechanism hook.
        outcome = self._derive_outcome(obs)
        obs[self._outcome_var] = outcome
        return obs, outcome

    def _derive_outcome(self, observation: dict[str, Any]) -> float:
        """Default outcome: information_gained → monotonic outcome surrogate.

        Subclasses or evaluation harnesses can replace this with a true SCM
        forward-simulation; the default keeps the agents unit-testable in
        isolation.
        """
        info = float(observation.get("information_gained", 0.0))
        # Saturating monotone curve so repeated low-information picks stop
        # advancing — this surfaces meaningful planning differences.
        return round(min(1.0, 0.15 + 0.95 * info), 3)


# ─── glue dataclass for the policy return value ───────────────────────────────


@dataclass(frozen=True)
class AgentChoice:
    """The output of a subclass's :meth:`_choose_action`.

    Wrapping the chosen action together with the full ranking lets the
    trace record provenance — every alternative considered — without
    forcing the policy to ship a tuple convention.
    """

    action: ActionCandidate
    all_scores: list[ActionScore]
