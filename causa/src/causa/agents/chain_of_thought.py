"""Chain-of-Thought baseline (Wei et al. 2022).

Distinct from ReAct in one important way: the agent emits its *entire*
reasoning chain in a single LLM call per step, rather than interleaving
short thoughts with tool calls.

In our adaptation for tool-using debugging, the chain has two parts:

1. A free-form reasoning block ("let's think step by step …").
2. A final boxed answer naming the chosen tool.

The boxed-answer parsing is the well-known CoT-for-tools recipe.  We do not
modify state-tracking — the LLM context carries the chain across steps
exactly the way it would across CoT prompt turns.

What this baseline does *not* do (and the dissertation contrasts):
- No structured causal model;
- No back-door adjustment;
- No counterfactual reflection;
- No durable history beyond the LLM context.
"""

from __future__ import annotations

import json
import re
from typing import Any

from causa.agents.base import (
    AgentChoice,
    AgentStep,
    BaseAgent,
    DecisionContext,
)
from causa.domain.tasks import DebuggingTask
from causa.ports.llm import LLMClient, LLMMessage, LLMRole
from causa.ports.scorer import ActionCandidate, ActionScore
from causa.ports.tool import DebuggingTool


COT_SYSTEM = (
    "You are a debugging agent using chain-of-thought reasoning.  At each "
    "step, write out your reasoning step by step, then conclude with a "
    'final answer in the form: ```final\\n{"action": "<tool_name>"}\\n```. '
    "Use only the provided tool names."
)


# Recognise the boxed final answer the model is asked to emit.
_FINAL_BLOCK = re.compile(r"```final\s*(\{.*?\})\s*```", re.DOTALL)


class ChainOfThoughtAgent(BaseAgent):
    """One-shot CoT-per-step debugging agent.

    Parameters
    ----------
    llm:
        The LLM client.
    tools, step_budget, action_variable, outcome_variable:
        Forwarded to :class:`BaseAgent`.
    name:
        Telemetry identifier; defaults to ``"causa.cot"``.
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
        name: str = "causa.cot",
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
        self._chain: list[str] = []  # running CoT trace (LLM context only)

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
            system=COT_SYSTEM,
            json_mode=False,  # CoT prefers free-form prose + boxed answer
            max_tokens=1024,
        )
        reasoning, chosen_name = self._parse(response.content)
        chosen = next(
            (c for c in candidates if c.name == chosen_name),
            candidates[0],
        )
        self._chain.append(f"step {context.step_index}: {reasoning[:240]}")
        scores = [
            ActionScore(
                action=c,
                score=1.0 if c.name == chosen.name else 0.0,
                rationale="chain-of-thought selection" if c.name == chosen.name else "",
            )
            for c in candidates
        ]
        return AgentChoice(action=chosen, all_scores=scores)

    def _after_step(
        self,
        *,
        step: AgentStep,
        task: DebuggingTask,  # noqa: ARG002
        state: dict[str, Any],  # noqa: ARG002
    ) -> None:
        # CoT keeps its reasoning chain in the LLM context — we mirror the
        # last reasoning fragment into the structured step trace so the
        # post-hoc evaluator can score explanation quality (§G).
        step.reflection = (
            f"cot-chain ({step.context.step_index}): "
            f"{self._chain[-1] if self._chain else '(empty)'}"
        )

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
        chain_block = "\n".join(self._chain[-6:]) or "(no prior reasoning)"
        return (
            "[CAUSA::cot_step]\n\n"
            f"Task: {task.description}\n"
            f"Step {context.step_index} state: {json.dumps(context.state, default=str)}\n\n"
            f"Prior reasoning chain:\n{chain_block}\n\n"
            f"Candidate tools:\n{tool_block}\n\n"
            "Let's think step by step.  Then conclude with a boxed final "
            "answer choosing exactly one tool name."
        )

    @staticmethod
    def _parse(content: str) -> tuple[str, str]:
        # 1. Try the structured ```final``` block first.
        m = _FINAL_BLOCK.search(content)
        if m:
            try:
                payload = json.loads(m.group(1))
                if isinstance(payload, dict):
                    return (content[: m.start()].strip(), str(payload.get("action", "")))
            except json.JSONDecodeError:
                pass
        # 2. Mock LLM (and well-behaved real LLMs) may return pure JSON.
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return (content.strip(), "")
        if isinstance(payload, dict) and "action" in payload:
            return (str(payload.get("reasoning", content)).strip(), str(payload["action"]))
        return (content.strip(), "")
