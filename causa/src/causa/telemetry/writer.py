"""JSON Lines trace writer.

Append-only, fsync-optional, thread-safe.  Designed to be passed into the
evaluation runner so each :class:`AgentStep` is persisted as it happens.

A :class:`NullTraceWriter` is provided for tests and CLI dry-runs — same
interface, no-ops on write.

Why thread-safety: the runner uses a thread pool when parallelism > 1, so
multiple agents may emit events concurrently.  A single lock around the
file handle is sufficient (events are tiny; fsync is opt-in).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import IO, Any, Protocol

from causa.agents.base import AgentResult, AgentStep
from causa.telemetry.events import (
    build_result_event,
    build_run_event,
    build_step_event,
    new_run_id,
)


class TraceSink(Protocol):
    """Internal protocol — anything with .write/.flush/.close behaves."""

    def write(self, event: dict[str, Any]) -> None: ...
    def flush(self) -> None: ...
    def close(self) -> None: ...


class TraceWriter:
    """JSON Lines trace writer with optional fsync-per-write durability.

    Parameters
    ----------
    path:
        Output ``.jsonl`` file.  Created on first write; parent directory
        is created if missing.
    run_id:
        Stable identifier; auto-generated if ``None``.
    agent_name:
        Identifier of the agent writing the trace; recorded in framing
        events.
    fsync_each_step:
        When True, ``fsync`` the underlying file after every event.
        Slower but survives a crash mid-run.  Default ``False`` because
        the eval runs typically finish quickly enough that batched fsync
        on close is fine.
    """

    def __init__(
        self,
        *,
        path: Path,
        run_id: str | None = None,
        agent_name: str = "unknown",
        fsync_each_step: bool = False,
    ) -> None:
        self._path = path
        self._run_id = run_id or new_run_id()
        self._agent_name = agent_name
        self._fsync = fsync_each_step
        self._lock = threading.Lock()
        self._handle: IO[str] | None = None
        self._started = False

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(
        self,
        *,
        task_count: int | None = None,
        seed: int | None = None,
        extra: dict[str, Any] | None = None,
    ) -> "TraceWriter":
        """Open the file and write the ``run_started`` framing event."""
        if self._started:
            return self
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self._path.open("a", encoding="utf-8")
        self._write_event(build_run_event(
            kind="run_started",
            run_id=self._run_id,
            agent_name=self._agent_name,
            task_count=task_count,
            seed=seed,
            extra=extra,
        ))
        self._started = True
        return self

    def finish(self, *, extra: dict[str, Any] | None = None) -> None:
        """Write the ``run_finished`` framing event and close the file."""
        if not self._started:
            return
        self._write_event(build_run_event(
            kind="run_finished",
            run_id=self._run_id,
            agent_name=self._agent_name,
            extra=extra,
        ))
        if self._handle is not None:
            self._handle.flush()
            self._handle.close()
            self._handle = None

    def __enter__(self) -> "TraceWriter":
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        self.finish()

    # ── writes ────────────────────────────────────────────────────────────

    def emit_step(self, *, task_id: str, step: AgentStep) -> None:
        """Persist one :class:`AgentStep` event."""
        self._write_event(build_step_event(
            run_id=self._run_id,
            task_id=task_id,
            agent_name=self._agent_name,
            step=step,
        ))

    def emit_result(self, *, result: AgentResult) -> None:
        """Persist one :class:`AgentResult` summary event."""
        self._write_event(build_result_event(run_id=self._run_id, result=result))

    # ── RunObserver implementation (causa.evaluation.runner.RunObserver) ──

    def on_run_started(self, *, task_count: int) -> None:
        self.start(task_count=task_count)

    def on_step(self, *, task_id: str, step: AgentStep) -> None:
        self.emit_step(task_id=task_id, step=step)

    def on_task_result(self, *, result: AgentResult) -> None:
        self.emit_result(result=result)

    def on_run_finished(self) -> None:
        self.finish()

    @property
    def run_id(self) -> str:
        return self._run_id

    @property
    def path(self) -> Path:
        return self._path

    # ── internals ─────────────────────────────────────────────────────────

    def _write_event(self, event: dict[str, Any]) -> None:
        if self._handle is None:
            # Lazy-open if start() was skipped (allowed for ad-hoc use).
            self.start()
        assert self._handle is not None  # narrow type for type-checkers
        line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self._handle.write(line + "\n")
            if self._fsync:
                self._handle.flush()
                import os
                os.fsync(self._handle.fileno())


class NullTraceWriter:
    """A drop-in :class:`TraceWriter` that discards all events.

    Used in tests and in CLI ``--no-trace`` mode so the surrounding code
    doesn't grow a conditional around every emit.
    """

    run_id: str = "null"
    path: Path = Path("/dev/null")

    def start(self, **kwargs: Any) -> "NullTraceWriter":  # noqa: ARG002
        return self

    def finish(self, **kwargs: Any) -> None:  # noqa: ARG002
        return None

    def __enter__(self) -> "NullTraceWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def emit_step(self, *, task_id: str, step: AgentStep) -> None:  # noqa: ARG002
        return None

    def emit_result(self, *, result: AgentResult) -> None:  # noqa: ARG002
        return None

    def on_run_started(self, *, task_count: int) -> None:  # noqa: ARG002
        return None

    def on_step(self, *, task_id: str, step: AgentStep) -> None:  # noqa: ARG002
        return None

    def on_task_result(self, *, result: AgentResult) -> None:  # noqa: ARG002
        return None

    def on_run_finished(self) -> None:
        return None
