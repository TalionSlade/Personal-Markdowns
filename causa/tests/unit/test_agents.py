"""Tests for the four agents — pin behavioural invariants of each arm."""

from __future__ import annotations

import pandas as pd

from causa.adapters.history.pandas_history import PandasObservationHistory
from causa.adapters.tools.registry import make_tool_registry
from causa.agents.causal import CausalPlanningAgent
from causa.agents.chain_of_thought import ChainOfThoughtAgent
from causa.agents.no_memory import NoMemoryAgent
from causa.agents.react import ReActAgent
from causa.domain.scm_debugging import (
    build_debugging_scm,
    debugging_observation_schema,
)
from causa.domain.tasks import DebuggingTask, TaskOutcome
from causa.ports.llm import LLMMessage, LLMResponse
from causa.ports.scorer import (
    ActionCandidate,
    ActionScore,
    ActionScorer,
)


# ─── stubs (no network, no API cost) ─────────────────────────────────────────


class _DummyLLM:
    """Always picks the first tool — deterministic across all baselines."""

    model = "dummy-llm"

    def complete(self, messages, *, system=None, json_mode=False,  # noqa: ANN001, ARG002
                 max_tokens=1024, temperature=0.0):
        return LLMResponse(
            content='{"action": "code_search", "rationale": "stub", "thought": ""}',
            model=self.model,
            input_tokens=0,
            output_tokens=0,
        )


class _MaxScorer(ActionScorer):
    """Scorer that ranks the first candidate at 1.0, rest at 0.0."""

    name = "stub_max"

    def score(self, candidates, *, state, history):  # noqa: ANN001, ARG002
        return [
            ActionScore(action=c, score=1.0 if i == 0 else 0.0, rationale="stub")
            for i, c in enumerate(candidates)
        ]


def _task() -> DebuggingTask:
    return DebuggingTask(
        task_id="t1",
        description="example",
        initial_state={
            "error_message_type": "type_error",
            "codebase_structure": "small_flat",
            "context_available": "partial",
        },
    )


# ─── baseline tests ──────────────────────────────────────────────────────────


def test_causal_agent_records_step_per_iteration():
    scm = build_debugging_scm()
    history = PandasObservationHistory(schema=debugging_observation_schema())
    agent = CausalPlanningAgent(
        scorer=_MaxScorer(),
        reflection=None,
        history=history,
        tools=make_tool_registry(),
        step_budget=4,
        action_variable=scm.action_variable.name,
        outcome_variable=scm.outcome_variable.name,
        success_threshold=2.0,  # impossible threshold → fills budget
    )
    result = agent.run(_task())
    assert result.outcome is TaskOutcome.BUDGET_EXCEEDED
    assert result.trace.n_steps == 4
    # Each step landed in the history.
    assert history.n_observations == 4


def test_causal_agent_solves_when_threshold_reachable():
    """With the default outcome surrogate, an outcome ≥ 1.0 immediately
    terminates the task as SOLVED.  We accept any pass."""
    scm = build_debugging_scm()
    history = PandasObservationHistory(schema=debugging_observation_schema())
    agent = CausalPlanningAgent(
        scorer=_MaxScorer(),
        reflection=None,
        history=history,
        tools=make_tool_registry(),
        step_budget=8,
        action_variable=scm.action_variable.name,
        outcome_variable=scm.outcome_variable.name,
        success_threshold=0.1,  # trivially met
    )
    result = agent.run(_task())
    assert result.outcome is TaskOutcome.SOLVED
    assert result.trace.n_steps == 1


def test_react_agent_wipes_to_first_tool_via_dummy_llm():
    scm = build_debugging_scm()
    agent = ReActAgent(
        llm=_DummyLLM(),
        tools=make_tool_registry(),
        step_budget=2,
        action_variable=scm.action_variable.name,
        outcome_variable=scm.outcome_variable.name,
        success_threshold=2.0,
    )
    result = agent.run(_task())
    assert result.trace.n_steps == 2
    # Every step picked code_search (the dummy LLM's canned answer).
    assert all(step.chosen_action == "code_search" for step in result.trace.steps)


def test_no_memory_agent_wipes_mediators_between_steps():
    scm = build_debugging_scm()
    agent = NoMemoryAgent(
        llm=_DummyLLM(),
        tools=make_tool_registry(),
        step_budget=2,
        action_variable=scm.action_variable.name,
        outcome_variable=scm.outcome_variable.name,
        success_threshold=2.0,
    )
    result = agent.run(_task())
    # The state at step 1 must lack the mediators that step 0's observation
    # would have left behind.
    step_1_state = result.trace.steps[1].context.state
    assert "information_gained" not in step_1_state


def test_cot_agent_records_chain_reflection():
    scm = build_debugging_scm()
    agent = ChainOfThoughtAgent(
        llm=_DummyLLM(),
        tools=make_tool_registry(),
        step_budget=1,
        action_variable=scm.action_variable.name,
        outcome_variable=scm.outcome_variable.name,
        success_threshold=2.0,
    )
    result = agent.run(_task())
    assert result.trace.n_steps == 1
    assert result.trace.steps[0].reflection.startswith("cot-chain")
