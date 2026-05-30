"""Trace event schema (v1).

Each agent step is one event.  Each evaluation run is bracketed by
``run_started`` and ``run_finished`` events so the trace file is self-
describing.

The schema is *closed* — adding fields is fine; renaming or removing them
requires bumping :data:`TRACE_VERSION`.
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from causa.agents.base import AgentResult, AgentStep
from causa.ports.scorer import ActionScore


TRACE_VERSION: int = 1


# ─── one trace event ─────────────────────────────────────────────────────────


class TraceEvent(dict[str, Any]):
    """A JSON-serialisable trace event.

    Subclassing ``dict`` (not adding a separate Pydantic model) keeps the
    write path zero-copy and avoids dragging Pydantic into the trace
    plumbing.
    """


# ─── builders ────────────────────────────────────────────────────────────────


def build_run_event(
    *,
    kind: str,
    run_id: str,
    agent_name: str,
    task_count: int | None = None,
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> TraceEvent:
    """Build a ``run_started`` / ``run_finished`` framing event."""
    event = TraceEvent(
        trace_version=TRACE_VERSION,
        kind=kind,
        timestamp=time.time(),
        run_id=run_id,
        agent_name=agent_name,
    )
    if task_count is not None:
        event["task_count"] = task_count
    if seed is not None:
        event["seed"] = seed
    if extra:
        event["extra"] = extra
    return event


def build_step_event(
    *,
    run_id: str,
    task_id: str,
    agent_name: str,
    step: AgentStep,
) -> TraceEvent:
    """Build a ``step`` event from one :class:`AgentStep`."""
    return TraceEvent(
        trace_version=TRACE_VERSION,
        kind="step",
        timestamp=time.time(),
        run_id=run_id,
        task_id=task_id,
        agent_name=agent_name,
        step_index=step.context.step_index,
        history_size=step.context.history_size,
        state=_safe_json(step.context.state),
        scores=[_score_payload(s) for s in step.scores],
        chosen_action=step.chosen_action,
        tool_output=_tool_payload(step.tool_output),
        observation=_safe_json(step.observation),
        outcome=step.outcome,
        reflection=step.reflection,
        elapsed_seconds=round(step.elapsed_seconds, 4),
    )


def build_result_event(
    *,
    run_id: str,
    result: AgentResult,
) -> TraceEvent:
    """Build a terminal ``task_result`` event summarising one task."""
    return TraceEvent(
        trace_version=TRACE_VERSION,
        kind="task_result",
        timestamp=time.time(),
        run_id=run_id,
        task_id=result.task_id,
        agent_name=result.agent_name,
        outcome=result.outcome.value,
        final_outcome_score=result.final_outcome_score,
        n_steps=result.trace.n_steps,
        notes=list(result.trace.notes),
        error=result.error,
    )


def new_run_id() -> str:
    """Generate a fresh run identifier (used by the CLI and tests)."""
    return uuid.uuid4().hex[:12]


# ─── coercion helpers ────────────────────────────────────────────────────────


def _score_payload(score: ActionScore) -> dict[str, Any]:
    return {
        "action": score.action.name,
        "score": float(score.score),
        "confidence": (
            float(score.confidence) if score.confidence is not None else None
        ),
        "rationale": score.rationale,
        "adjustment_set": sorted(score.adjustment_set),
    }


def _tool_payload(output: Any) -> dict[str, Any] | None:
    if output is None:
        return None
    return {
        "success": bool(output.success),
        "information_score": float(output.information_score),
        "elapsed_seconds": float(output.elapsed_seconds),
        "payload": _safe_json(output.payload),
    }


def _safe_json(value: Any) -> Any:  # noqa: ANN401
    """Make a payload JSON-friendly without dragging custom encoders into
    the writer."""
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_safe_json(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)
