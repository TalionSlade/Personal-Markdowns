"""Tests for the telemetry layer: TraceWriter, NullTraceWriter, auditor.

A trace must:
* round-trip — every event written can be parsed back
* be self-framed — start with ``run_started`` and end with ``run_finished``
* survive truncation — the auditor skips a partial trailing line silently

The auditor must:
* count steps and tasks correctly
* identify cold-start vs steady-state by rationale substrings
* flag reflection triggers via "synthetic row" substring
"""

from __future__ import annotations

import json
from pathlib import Path

from causa.agents.base import (
    AgentResult,
    AgentStep,
    DecisionContext,
    DecisionTrace,
)
from causa.domain.tasks import TaskOutcome
from causa.ports.scorer import ActionCandidate, ActionScore
from causa.telemetry import (
    NullTraceWriter,
    TraceWriter,
    audit_trace,
    build_run_event,
    build_step_event,
)
from causa.telemetry.events import TRACE_VERSION, build_result_event, new_run_id


# ─── fixtures (hand-rolled — no pytest fixtures for portability) ────────────


def _step(*, idx: int = 0, action: str = "code_search",
          rationale: str = "via back-door {Z}", outcome: float = 0.5,
          reflection: str = "no update") -> AgentStep:
    candidate = ActionCandidate(name=action)
    score = ActionScore(
        action=candidate, score=0.8,
        rationale=rationale, adjustment_set=frozenset({"Z"}),
    )
    return AgentStep(
        context=DecisionContext(step_index=idx, state={"k": "v"}, history_size=idx),
        scores=[score],
        chosen_action=action,
        tool_output=None,
        observation={"chosen_action": action, "outcome": outcome},
        outcome=outcome,
        reflection=reflection,
        elapsed_seconds=0.01,
    )


def _result(task_id: str = "t-1", agent_name: str = "causal") -> AgentResult:
    trace = DecisionTrace(
        task_id=task_id, agent_name=agent_name,
        steps=[_step(idx=0), _step(idx=1)],
    )
    return AgentResult(
        task_id=task_id,
        agent_name=agent_name,
        outcome=TaskOutcome.SOLVED,
        final_outcome_score=0.7,
        trace=trace,
    )


# ─── event builders ──────────────────────────────────────────────────────────


def test_run_event_carries_schema_version():
    ev = build_run_event(
        kind="run_started", run_id="r1", agent_name="causal", task_count=4,
    )
    assert ev["trace_version"] == TRACE_VERSION
    assert ev["kind"] == "run_started"
    assert ev["agent_name"] == "causal"
    assert ev["task_count"] == 4
    assert "timestamp" in ev


def test_step_event_serialises_adjustment_set():
    ev = build_step_event(
        run_id="r1", task_id="t1", agent_name="causal", step=_step(),
    )
    assert ev["kind"] == "step"
    assert ev["chosen_action"] == "code_search"
    assert ev["scores"][0]["adjustment_set"] == ["Z"]


def test_result_event_carries_outcome_value():
    ev = build_result_event(run_id="r1", result=_result())
    assert ev["kind"] == "task_result"
    assert ev["outcome"] == TaskOutcome.SOLVED.value
    assert ev["n_steps"] == 2


# ─── writer round-trip ──────────────────────────────────────────────────────


def test_writer_round_trip_writes_jsonl(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path=path, agent_name="causal")
    writer.start(task_count=1, seed=42)
    writer.emit_step(task_id="t-1", step=_step(idx=0))
    writer.emit_result(result=_result())
    writer.finish()

    lines = [json.loads(l) for l in path.read_text().splitlines() if l]
    kinds = [e["kind"] for e in lines]
    assert kinds[0] == "run_started"
    assert "step" in kinds
    assert "task_result" in kinds
    assert kinds[-1] == "run_finished"


def test_writer_idempotent_finish(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path=path, agent_name="x")
    writer.start()
    writer.finish()
    writer.finish()  # second finish is a no-op (was already closed)
    lines = path.read_text().splitlines()
    assert sum(1 for l in lines if '"run_finished"' in l) == 1


def test_writer_run_id_stable_when_supplied(tmp_path: Path):
    rid = new_run_id()
    writer = TraceWriter(path=tmp_path / "t.jsonl", run_id=rid, agent_name="x")
    assert writer.run_id == rid


def test_null_writer_is_quiet_drop_in():
    nw = NullTraceWriter()
    nw.start(task_count=10)
    nw.emit_step(task_id="t-1", step=_step())
    nw.emit_result(result=_result())
    nw.finish()
    # No file created, no exception.
    assert nw.path == Path("/dev/null")


# ─── auditor ─────────────────────────────────────────────────────────────────


def test_auditor_counts_steps_and_tasks(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path=path, agent_name="causal") as writer:
        for tid in ("t-1", "t-2"):
            writer.emit_step(task_id=tid, step=_step(idx=0))
            writer.emit_step(task_id=tid, step=_step(idx=1))
            writer.emit_result(result=_result(task_id=tid))
    audit = audit_trace(path)
    assert audit.n_steps == 4
    assert audit.n_tasks == 2
    assert audit.action_counts["code_search"] == 4
    assert audit.agent_name == "causal"


def test_auditor_classifies_cold_vs_steady_via_rationale(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path=path, agent_name="causal") as writer:
        writer.emit_step(
            task_id="t-1",
            step=_step(rationale="[cold-start LLM] no data yet"),
        )
        writer.emit_step(
            task_id="t-1",
            step=_step(rationale="via back-door {Z}"),
        )
    audit = audit_trace(path)
    assert audit.cold_start_steps == 1
    assert audit.steady_state_steps == 1


def test_auditor_counts_reflection_triggers(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path=path, agent_name="causal") as writer:
        writer.emit_step(task_id="t-1",
                         step=_step(reflection="appended synthetic row"))
        writer.emit_step(task_id="t-1",
                         step=_step(reflection="no update"))
    audit = audit_trace(path)
    assert audit.reflection_triggers == 1


def test_auditor_skips_corrupt_lines(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    with TraceWriter(path=path, agent_name="causal") as writer:
        writer.emit_step(task_id="t-1", step=_step())
    # Append a corrupt line — auditor must not crash.
    with path.open("a", encoding="utf-8") as fh:
        fh.write("{not-valid-json\n")
    audit = audit_trace(path)
    assert audit.n_steps == 1


def test_auditor_records_errors(tmp_path: Path):
    path = tmp_path / "trace.jsonl"
    result = _result()
    object.__setattr__(result, "error", "boom")
    with TraceWriter(path=path, agent_name="causal") as writer:
        writer.emit_result(result=result)
    audit = audit_trace(path)
    assert audit.errors == ["boom"]
