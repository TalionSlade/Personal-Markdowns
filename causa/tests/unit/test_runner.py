"""Tests for :class:`EvaluationRunner` — serial path, observer ordering,
and parallel parity.

The runner is the *exact* surface that the dissertation's evaluation
pipeline calls.  These tests pin three properties:

1. The observer sees ``run_started`` → ``on_step``\\* → ``on_task_result``
   → ``run_finished`` in order on the serial path.
2. Per-task ``on_task_result`` order matches submission order even in the
   parallel path (so trace files stay readable).
3. Aggregate metrics are produced for the report.
"""

from __future__ import annotations

from causa.agents.base import (
    AgentResult,
    AgentStep,
    DecisionContext,
    DecisionTrace,
)
from causa.domain.tasks import DebuggingTask, TaskOutcome
from causa.evaluation.runner import EvaluationRunner, RunObserver
from causa.ports.scorer import ActionCandidate, ActionScore


# ─── stub agent (no tools, no LLM) ──────────────────────────────────────────


class _StubAgent:
    """A no-op agent that produces a fixed 1-step solved trace per task.

    Duck-typed against BaseAgent.run — sufficient for runner-only tests
    because the runner never inspects the agent beyond ``agent.run(task)``.
    """

    name = "stub-agent"

    def run(self, task: DebuggingTask) -> AgentResult:
        candidate = ActionCandidate(name="noop")
        score = ActionScore(action=candidate, score=1.0, rationale="stub")
        step = AgentStep(
            context=DecisionContext(step_index=0, state={}, history_size=0),
            scores=[score],
            chosen_action="noop",
            tool_output=None,
            outcome=1.0,
            observation={},
            reflection="",
            elapsed_seconds=0.001,
        )
        trace = DecisionTrace(
            task_id=task.task_id, agent_name=self.name, steps=[step],
        )
        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            outcome=TaskOutcome.SOLVED,
            final_outcome_score=1.0,
            trace=trace,
        )


class _RecordingObserver:
    """Records the sequence of observer callbacks for assertion."""

    def __init__(self) -> None:
        self.events: list[str] = []
        self.step_task_ids: list[str] = []
        self.result_task_ids: list[str] = []

    def on_run_started(self, *, task_count: int) -> None:
        self.events.append(f"start:{task_count}")

    def on_step(self, *, task_id: str, step: AgentStep) -> None:  # noqa: ARG002
        self.events.append(f"step:{task_id}")
        self.step_task_ids.append(task_id)

    def on_task_result(self, *, result: AgentResult) -> None:
        self.events.append(f"result:{result.task_id}")
        self.result_task_ids.append(result.task_id)

    def on_run_finished(self) -> None:
        self.events.append("finish")


def _task(task_id: str) -> DebuggingTask:
    return DebuggingTask(task_id=task_id, description="t")


# ─── tests ──────────────────────────────────────────────────────────────────


def test_runner_observer_sees_canonical_event_order():
    obs = _RecordingObserver()
    runner = EvaluationRunner(
        agent_factory=_StubAgent, parallelism=1, observer=obs,
    )
    runner.run([_task("a"), _task("b")])
    assert obs.events == [
        "start:2",
        "step:a",
        "result:a",
        "step:b",
        "result:b",
        "finish",
    ]


def test_runner_satisfies_runobserver_protocol():
    """Type-narrowing protocol check — observer must be RunObserver."""
    obs = _RecordingObserver()
    assert isinstance(obs, RunObserver)


def test_runner_emits_results_in_submission_order_parallel():
    obs = _RecordingObserver()
    runner = EvaluationRunner(
        agent_factory=_StubAgent, parallelism=4, observer=obs,
    )
    tasks = [_task(f"t-{i}") for i in range(10)]
    runner.run(tasks)
    assert obs.result_task_ids == [t.task_id for t in tasks]


def test_runner_report_aggregates_metrics():
    runner = EvaluationRunner(agent_factory=_StubAgent, parallelism=1)
    report = runner.run([_task("a"), _task("b"), _task("c")])
    assert report.agent_name == "stub-agent"
    assert len(report.results) == 3
    assert report.metrics is not None
    # All three solved → 100% success rate.
    assert report.metrics.success_rate == 1.0


def test_runner_invokes_progress_callback():
    seen: list[tuple[int, int]] = []
    runner = EvaluationRunner(
        agent_factory=_StubAgent,
        parallelism=1,
        progress_callback=lambda done, total: seen.append((done, total)),
    )
    runner.run([_task("a"), _task("b")])
    assert seen == [(1, 2), (2, 2)]


def test_runner_rejects_invalid_parallelism():
    import pytest
    with pytest.raises(ValueError):
        EvaluationRunner(agent_factory=_StubAgent, parallelism=0)


def test_runner_swallows_observer_exceptions():
    """An observer that raises must not abort the run."""

    class _BadObserver:
        def on_run_started(self, *, task_count: int) -> None:
            raise RuntimeError("boom")

        def on_step(self, *, task_id, step) -> None:  # noqa: ANN001, ARG002
            raise RuntimeError("boom")

        def on_task_result(self, *, result) -> None:  # noqa: ANN001, ARG002
            raise RuntimeError("boom")

        def on_run_finished(self) -> None:
            raise RuntimeError("boom")

    runner = EvaluationRunner(
        agent_factory=_StubAgent, parallelism=1, observer=_BadObserver(),
    )
    report = runner.run([_task("a")])
    assert report.metrics is not None
