"""LLM-only action scorer.

This is the *correlational* baseline the dissertation argues against, used:

- as the **cold-start** scorer when history is too thin for DoWhy (§C5);
- as the **ablation arm** for §J3#2 — does the do-calculus machinery earn
  its complexity beyond a well-prompted LLM?

The prompt template includes a marker token so the mock LLM client returns
a canned-but-context-aware response without API cost.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from causa.ports.llm import LLMClient, LLMMessage, LLMRole
from causa.ports.scorer import ActionCandidate, ActionScore, ActionScorer


SYSTEM_PROMPT = (
    "You are the action-scoring component of a debugging agent.  Given the "
    "current state, the list of candidate tools, and a brief snippet of past "
    "observations, return a JSON array ranking each candidate tool by its "
    "estimated probability of leading to a passing test suite if invoked now. "
    "Be terse; rationales should be one short sentence."
)


class LLMActionScorer(ActionScorer):
    """Score candidate actions via an LLM ranking call."""

    name: str = "llm_scorer"

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    def score(
        self,
        candidates: list[ActionCandidate],
        *,
        state: dict[str, Any],
        history: pd.DataFrame,
    ) -> list[ActionScore]:
        prompt = self._render_prompt(candidates, state=state, history=history)
        response = self._llm.complete(
            messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
            system=SYSTEM_PROMPT,
            json_mode=True,
            max_tokens=512,
        )
        scores_by_name = self._parse(response.content)
        out: list[ActionScore] = []
        for c in candidates:
            raw = scores_by_name.get(c.name, {})
            score = float(raw.get("score", 0.5))
            out.append(ActionScore(
                action=c,
                score=score,
                rationale=str(raw.get("rationale", "LLM-only correlational ranking")),
            ))
        return out

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _render_prompt(
        candidates: list[ActionCandidate],
        *,
        state: dict[str, Any],
        history: pd.DataFrame,
    ) -> str:
        cand_block = "\n".join(f"- {c.name}: {c.metadata.get('description', '')}"
                               for c in candidates)
        hist_block = (
            history.tail(8).to_csv(index=False)
            if not history.empty else "(none)"
        )
        state_block = json.dumps(state, indent=2, default=str)
        return (
            "[CAUSA::score_actions]\n\n"
            f"State:\n{state_block}\n\n"
            f"Candidate tools:\n{cand_block}\n\n"
            f"Recent observations (last 8 rows):\n{hist_block}\n\n"
            'Respond with a JSON array of objects {"action": str, "score": float in [0,1], '
            '"rationale": str}.'
        )

    @staticmethod
    def _parse(content: str) -> dict[str, dict[str, Any]]:
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            return {}
        if isinstance(parsed, list):
            return {item["action"]: item for item in parsed if "action" in item}
        if isinstance(parsed, dict) and "actions" in parsed:
            return {item["action"]: item for item in parsed["actions"]}
        return {}
