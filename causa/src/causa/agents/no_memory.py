"""NoMemoryAgent — the *floor* baseline for §J3.

This agent has:

- An LLM that sees the current state and the tool list;
- No scratchpad;
- No history;
- No reflection.

Every step is *iid* from the agent's perspective.  Its purpose is to
establish the no-memory performance floor so the gains attributable to
each of (a) ReAct's scratchpad, (b) CoT's reasoning chain, and (c) Causa's
structured causal memory are all separable in the ablation report.

A reader who finds this trivial is reading it correctly — the point is to
keep all moving parts except the policy *identical* to the other arms.
"""

from __future__ import annotations

import json
from typing import Any

from causa.agents.base import AgentChoice, BaseAgent, DecisionContext
from causa.domain.tasks import DebuggingTask
from causa.ports.llm import LLMClient, LLMMessage, LLMRole
from causa.ports.scorer import ActionCandidate, ActionScore
from causa.ports.tool import DebuggingTool


NM_SYSTEM = (
    "You are a single-step debugging agent.  Given the current task state "
    "and a list of candidate tools, pick the best tool to invoke now.  You "
    "have no memory of previous steps.  Reply with JSON: "
    '{"action": str, "rationale": str}.'
)


class NoMemoryAgent(BaseAgent):
    """LLM-only, memoryless agent.

    Parameters
    ----------
    llm:
        The LLM client.
    tools, step_budget, action_variable, outcome_variable:
        Forwarded to :class:`BaseAgent`.
    name:
        Telemetry identifier; defaults to ``"causa.no_memory"``.
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
        name: str = "causa.no_memory",
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
            system=NM_SYSTEM,
            json_mode=True,
            max_tokens=256,
        )
        chosen_name, rationale = self._parse(response.content)
        chosen = next(
            (c for c in candidates if c.name == chosen_name),
            candidates[0],
        )
        scores = [
            ActionScore(
                action=c,
                score=1.0 if c.name == chosen.name else 0.0,
                rationale=rationale if c.name == chosen.name else "",
            )
            for c in candidates
        ]
        return AgentChoice(action=chosen, all_scores=scores)

    # ── state transitions: the wipe ──────────────────────────────────────

    def _next_state(
        self,
        *,
        prev_state: dict[str, Any],
        observation: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        """Wipe mediators so each step starts from the task's observational
        inputs only.  This is what makes the agent *memoryless* in the
        operational sense — even our default mediator-carry-forward is
        suppressed."""
        keep = {"error_message_type", "codebase_structure", "context_available"}
        return {k: v for k, v in prev_state.items() if k in keep}

    # ── prompt rendering ─────────────────────────────────────────────────

    @staticmethod
    def _render_prompt(
        *,
        candidates: list[ActionCandidate],
        context: DecisionContext,
        task: DebuggingTask,
    ) -> str:
        tool_block = "\n".join(
            f"- {c.name}: {c.metadata.get('description', '')}" for c in candidates
        )
        return (
            "[CAUSA::no_memory_step]\n\n"
            f"Task: {task.description}\n"
            f"Current state: {json.dumps(context.state, default=str)}\n\n"
            f"Candidate tools:\n{tool_block}\n\n"
            'Reply with {"action": str, "rationale": str}.'
        )

    @staticmethod
    def _parse(content: str) -> tuple[str, str]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return ("", "")
        if not isinstance(data, dict):
            return ("", "")
        return (str(data.get("action", "")), str(data.get("rationale", "")))
