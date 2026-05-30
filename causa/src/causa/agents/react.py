"""ReAct baseline (Yao et al. 2023).

Trajectory shape: Thought → Action → Observation → Thought → Action → … .
The agent uses an LLM to verbalise a *thought*, choose a *tool*, and
incorporate the *observation* into the next thought.

What this baseline does *not* do, by construction:

- It does not maintain a structured observation history — its memory is
  the LLM scratchpad (the running thought stream).
- It does not run counterfactual reflection.
- Its action choice is correlational; there is no do-calculus step.

These three omissions are exactly the dissertation's claimed deltas (§J3).
The implementation here uses an :class:`ActionScorer` because that is the
seam our :class:`BaseAgent` exposes; in practice the scorer for ReAct is a
thin LLM ranking with a thought-style prompt — see
:func:`make_react_scorer` below.
"""

from __future__ import annotations

import json
from typing import Any

from causa.agents.base import (
    AgentChoice,
    AgentStep,
    BaseAgent,
    DecisionContext,
)
from causa.domain.tasks import DebuggingTask
from causa.ports.llm import LLMClient, LLMMessage, LLMRole
from causa.ports.scorer import ActionCandidate, ActionScore, ActionScorer
from causa.ports.tool import DebuggingTool


REACT_SYSTEM = (
    "You are a debugging agent using the ReAct pattern.  At each step, "
    "produce one short Thought, then choose exactly one Action from the "
    "candidate tools.  Respond as JSON: "
    '{"thought": str, "action": str, "rationale": str}.  The action must '
    "be one of the candidate tool names."
)


class ReActAgent(BaseAgent):
    """Thought-Action-Observation loop with an LLM scratchpad.

    Parameters
    ----------
    llm:
        The LLM client driving thoughts and actions.
    tools, step_budget, action_variable, outcome_variable:
        Forwarded to :class:`BaseAgent`.
    name:
        Telemetry identifier; defaults to ``"causa.react"``.
    """

    def __init__(
        self,
        *,
        llm: LLMClient,
        tools: dict[str, DebuggingTool],
        step_budget: int,
        action_variable: str,
        outcome_variable: str,
        success_threshold: float = 1.0,
        name: str = "causa.react",
    ) -> None:
        super().__init__(
            name=name,
            tools=tools,
            step_budget=step_budget,
            action_variable=action_variable,
            outcome_variable=outcome_variable,
            success_threshold=success_threshold,
        )
        self._llm = llm
        self._scratchpad: list[str] = []  # thought stream — the only "memory"

    # ── policy ────────────────────────────────────────────────────────────

    def _choose_action(
        self,
        *,
        candidates: list[ActionCandidate],
        context: DecisionContext,
        task: DebuggingTask,
    ) -> AgentChoice:
        prompt = self._render_prompt(candidates=candidates, context=context, task=task)
        response = self._llm.complete(
            messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
            system=REACT_SYSTEM,
            json_mode=True,
            max_tokens=512,
        )
        thought, chosen_name, rationale = self._parse(response.content)
        # Fall back to the first candidate if the LLM picks something off-menu.
        chosen = next(
            (c for c in candidates if c.name == chosen_name),
            candidates[0],
        )
        self._scratchpad.append(f"step {context.step_index}: thought={thought}")
        scores = [
            ActionScore(
                action=c,
                score=1.0 if c.name == chosen.name else 0.0,
                rationale=rationale if c.name == chosen.name else "",
            )
            for c in candidates
        ]
        return AgentChoice(action=chosen, all_scores=scores)

    # ── bookkeeping ───────────────────────────────────────────────────────

    def _after_step(
        self,
        *,
        step: AgentStep,
        task: DebuggingTask,  # noqa: ARG002
        state: dict[str, Any],  # noqa: ARG002
    ) -> None:
        obs_summary = (
            f"info={step.observation.get('information_gained', 0.0):.2f}, "
            f"outcome={step.outcome:.2f}"
        )
        self._scratchpad.append(f"step {step.context.step_index}: observation={obs_summary}")
        step.reflection = f"react-thought: {self._scratchpad[-2]}"

    # ── prompt rendering ──────────────────────────────────────────────────

    def _render_prompt(
        self,
        *,
        candidates: list[ActionCandidate],
        context: DecisionContext,
        task: DebuggingTask,
    ) -> str:
        tool_block = "\n".join(
            f"- {c.name}: {c.metadata.get('description', '')}" for c in candidates
        )
        scratchpad = "\n".join(self._scratchpad[-8:]) or "(empty)"
        return (
            "[CAUSA::react_step]\n\n"
            f"Task: {task.description}\n"
            f"Step {context.step_index} state: {json.dumps(context.state, default=str)}\n\n"
            f"Recent thoughts/observations:\n{scratchpad}\n\n"
            f"Candidate tools:\n{tool_block}\n\n"
            'Respond with {"thought": str, "action": str, "rationale": str}.'
        )

    @staticmethod
    def _parse(content: str) -> tuple[str, str, str]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return ("", "", "")
        if not isinstance(data, dict):
            return ("", "", "")
        return (
            str(data.get("thought", "")),
            str(data.get("action", "")),
            str(data.get("rationale", "")),
        )


# ─── optional helper: a scorer-shaped wrapper for symmetric experiments ─────


class ReActScorer(ActionScorer):
    """Wrap a :class:`ReActAgent` step in :class:`ActionScorer` clothing.

    Lets the eval harness reuse :class:`CausalPlanningAgent` with a ReAct-
    style scorer for §J3#2 ablations (does do-calculus add value beyond
    well-prompted LLM action selection?).
    """

    name: str = "react_scorer"

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    def score(
        self,
        candidates: list[ActionCandidate],
        *,
        state: dict[str, Any],
        history: Any,  # noqa: ARG002
    ) -> list[ActionScore]:
        prompt = (
            "[CAUSA::react_step]\n\n"
            f"State: {json.dumps(state, default=str)}\n"
            f"Candidate tools: {[c.name for c in candidates]}\n\n"
            'Respond with {"thought": str, "action": str, "rationale": str}.'
        )
        response = self._llm.complete(
            messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
            system=REACT_SYSTEM,
            json_mode=True,
            max_tokens=256,
        )
        try:
            data = json.loads(response.content)
        except json.JSONDecodeError:
            data = {}
        winner = str(data.get("action", "")) if isinstance(data, dict) else ""
        rationale = str(data.get("rationale", "")) if isinstance(data, dict) else ""
        return [
            ActionScore(
                action=c,
                score=1.0 if c.name == winner else 0.0,
                rationale=rationale if c.name == winner else "",
            )
            for c in candidates
        ]
