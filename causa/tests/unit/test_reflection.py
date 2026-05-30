"""Tests for the Counterfactual Reflection Module — the novel contribution."""

from __future__ import annotations

from causa.ports.scorer import ActionCandidate
from causa.reflection.counterfactual import CounterfactualReflectionModule
from causa.reflection.threshold import (
    AdaptiveThreshold,
    StaticThreshold,
)


class _StubLLM:
    """LLM stub that emits a single canned JSON response per call.

    Lets each test pin the median estimates without ferrying through
    the mock client's marker keys.
    """

    def __init__(self, content: str) -> None:
        self._content = content

    def complete(self, **_: object):  # noqa: ANN003
        from dataclasses import dataclass

        @dataclass
        class _R:
            content: str
            input_tokens: int = 0
            output_tokens: int = 0

        return _R(content=self._content)


def _candidates(*names: str) -> list[ActionCandidate]:
    return [ActionCandidate(name=n, metadata={"description": ""}) for n in names]


# ── static threshold ────────────────────────────────────────────────────────


def test_static_threshold_is_constant():
    t = StaticThreshold(value=0.2)
    assert t.theta(0) == 0.2
    assert t.theta(100) == 0.2


def test_static_threshold_rejects_out_of_range():
    import pytest
    with pytest.raises(ValueError):
        StaticThreshold(value=1.5)


def test_adaptive_threshold_shrinks_with_history():
    t = AdaptiveThreshold(theta_0=0.4, theta_min=0.05, tau=10.0)
    assert t.theta(0) == 0.4
    assert t.theta(20) < 0.4
    assert t.theta(10_000) >= 0.05


# ── reflection module ───────────────────────────────────────────────────────


def test_reflection_does_not_fire_when_no_alternatives():
    llm = _StubLLM(content="[]")
    mod = CounterfactualReflectionModule(
        llm=llm, threshold=StaticThreshold(value=0.05), samples=1,
        action_variable="tool_selected", outcome_variable="tests_passed",
    )
    update = mod.reflect(
        state={"foo": "bar"},
        chosen_action=ActionCandidate(name="x"),
        observed_outcome=0.4,
        alternatives=[],
        n_observations=0,
    )
    assert not update.triggered
    assert update.synthetic_rows == []


def test_reflection_fires_when_estimate_exceeds_threshold():
    """Median estimate 0.8 vs observed 0.3 → Δ=0.5 > θ=0.05."""
    canned = '[{"action": "alt", "estimated_outcome": 0.8}]'
    llm = _StubLLM(content=canned)
    mod = CounterfactualReflectionModule(
        llm=llm, threshold=StaticThreshold(value=0.05), samples=1,
        action_variable="tool_selected", outcome_variable="tests_passed",
    )
    update = mod.reflect(
        state={"error_message_type": "type_error"},
        chosen_action=ActionCandidate(name="chosen"),
        observed_outcome=0.3,
        alternatives=_candidates("alt"),
        n_observations=5,
    )
    assert update.triggered
    assert len(update.synthetic_rows) == 1
    row = update.synthetic_rows[0]
    assert row["tool_selected"] == "alt"
    assert row["tests_passed"] == 0.8
    assert row["error_message_type"] == "type_error"


def test_reflection_does_not_fire_when_estimate_below_observed():
    canned = '[{"action": "alt", "estimated_outcome": 0.20}]'
    llm = _StubLLM(content=canned)
    mod = CounterfactualReflectionModule(
        llm=llm, threshold=StaticThreshold(value=0.05), samples=1,
        action_variable="tool_selected", outcome_variable="tests_passed",
    )
    update = mod.reflect(
        state={"error_message_type": "type_error"},
        chosen_action=ActionCandidate(name="chosen"),
        observed_outcome=0.4,  # observed > estimate
        alternatives=_candidates("alt"),
        n_observations=5,
    )
    assert not update.triggered
    assert update.synthetic_rows == []


def test_reflection_discards_non_json_response():
    llm = _StubLLM(content="not even close to JSON")
    mod = CounterfactualReflectionModule(
        llm=llm, threshold=StaticThreshold(value=0.05), samples=1,
        action_variable="tool_selected", outcome_variable="tests_passed",
    )
    update = mod.reflect(
        state={},
        chosen_action=ActionCandidate(name="chosen"),
        observed_outcome=0.3,
        alternatives=_candidates("alt"),
        n_observations=0,
    )
    assert update.estimates == []  # parser returned nothing
    assert not update.triggered
