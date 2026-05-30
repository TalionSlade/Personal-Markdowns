"""A deterministic mock LLM client.

Used by every unit test and as the default in :file:`.env.example`.  The
mock returns canned-but-context-aware responses for the three prompt
templates Causa generates:

- *counterfactual estimation* prompts (Component 3)
- *causal edge extraction* prompts (Component 1)
- *LLM-scoring fallback* prompts (Component 2 cold-start)

Each path is keyed off a marker phrase in the prompt — see the matching
templates in :mod:`causa.reflection`, :mod:`causa.extraction`, and
:mod:`causa.planning.llm_scorer`.

The mock is intentionally **deterministic** (no randomness, no time) so the
test suite is reproducible and CI never flakes.
"""

from __future__ import annotations

import hashlib
import json
import re

from causa.ports.llm import LLMClient, LLMMessage, LLMResponse


class MockLLMClient(LLMClient):
    """A deterministic, context-aware mock LLM client."""

    def __init__(self, *, model: str = "mock-llm-v1") -> None:
        self._model = model

    @property
    def model(self) -> str:
        return self._model

    # ── public API ─────────────────────────────────────────────────────────

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        system: str | None = None,
        json_mode: bool = False,
    ) -> LLMResponse:
        prompt = self._render(messages, system)
        if "[CAUSA::counterfactual]" in prompt:
            content = self._counterfactual_response(prompt)
        elif "[CAUSA::extract_edges]" in prompt:
            content = self._extract_edges_response(prompt)
        elif "[CAUSA::score_actions]" in prompt:
            content = self._score_actions_response(prompt)
        else:
            content = self._default_response(prompt)
        return LLMResponse(
            content=content,
            model=self._model,
            input_tokens=len(prompt) // 4,
            output_tokens=len(content) // 4,
            finish_reason="stop",
        )

    # ── canned response generators (deterministic on prompt hash) ──────────

    def _counterfactual_response(self, prompt: str) -> str:
        """Return JSON list of {action, estimated_outcome} for the alternatives."""
        alternatives = re.findall(r'"action":\s*"([^"]+)"', prompt)
        if not alternatives:
            # fall back to scanning for bullets
            alternatives = re.findall(r"-\s+([a-z_]+)", prompt)
        seed = int(hashlib.sha1(prompt.encode(), usedforsecurity=False).hexdigest(), 16)
        out = []
        for i, a in enumerate(dict.fromkeys(alternatives)):  # preserve order, dedupe
            # deterministic but varied "estimate" in [0.30, 0.85]
            est = 0.30 + 0.55 * (((seed >> (i * 4)) & 0xF) / 15.0)
            out.append({"action": a, "estimated_outcome": round(est, 3)})
        return json.dumps(out)

    def _extract_edges_response(self, prompt: str) -> str:
        """Propose plausible edges for the dissertation's debugging SCM."""
        # The mock uses the canonical edges of the 9-node debugging SCM.
        edges = [
            {"source": "error_message_type", "target": "hypothesis_space"},
            {"source": "codebase_structure", "target": "hypothesis_space"},
            {"source": "hypothesis_space", "target": "tool_selected"},
            {"source": "tool_selected", "target": "information_gained"},
            {"source": "context_available", "target": "information_gained"},
            {"source": "information_gained", "target": "root_cause_identified"},
            {"source": "root_cause_identified", "target": "patch_quality"},
            {"source": "patch_quality", "target": "tests_passed"},
        ]
        return json.dumps({"edges": edges})

    def _score_actions_response(self, prompt: str) -> str:
        """Return a deterministic JSON ranking of actions."""
        actions = re.findall(r"-\s+([a-z_]+)", prompt)
        seed = int(hashlib.sha1(prompt.encode(), usedforsecurity=False).hexdigest(), 16)
        out = []
        for i, a in enumerate(dict.fromkeys(actions)):
            score = 0.20 + 0.70 * (((seed >> (i * 3)) & 0x7) / 7.0)
            out.append({"action": a, "score": round(score, 3),
                        "rationale": f"prior strength {score:.2f}"})
        return json.dumps(out)

    def _default_response(self, prompt: str) -> str:
        return f"[mock-llm-response] received {len(prompt)} chars"

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _render(messages: list[LLMMessage], system: str | None) -> str:
        parts: list[str] = []
        if system:
            parts.append(f"[system]\n{system}\n")
        for m in messages:
            parts.append(f"[{m.role.value}]\n{m.content}\n")
        return "\n".join(parts)
