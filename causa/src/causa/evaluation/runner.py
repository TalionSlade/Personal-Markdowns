"""Evaluation runner.

Drives one :class:`BaseAgent` over a task list, optionally invoking a
``before_task`` reset hook so per-task agent state is wiped between tasks.

The runner is intentionally synchronous.  SWE-bench tasks are not the
bottleneck (the LLM call is), so async wouldn't help, and synchronous code
is dramatically easier to inspect when a trace looks wrong.

Concurrency is provided via the optional ``parallelism`` parameter, which
runs tasks in a :class:`concurrent.futures.ThreadPoolExecutor` — safe
because the agent is stateless across tasks once :meth:`reset` is called
on the supplied agent factory.
"""

from __future__ import annotations

import concurrent.futures
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from causa.agents.base import AgentResult, AgentStep, BaseAgent
from causa.domain.tasks import DebuggingTask
from causa.evaluation.metrics import MetricRow, summarize_results


# ─── observer port ───────────────────────────────────────────────────────────


@runtime_checkable
class RunObserver(Protocol):
    """Hook surface for tying telemetry, logging, or progress bars into a
    run.  Each method is best-effort; an observer that raises will not
    abort the run (errors are swallowed)."""

    def on_run_started(self, *, task_count: int) -> None: ...
    def on_step(self, *, task_id: str, step: AgentStep) -> None: ...
    def on_task_result(self, *, result: AgentResult) -> None: ...
    def on_run_finished(self) -> None: ...


# ─── runner report ───────────────────────────────────────────────────────────


@dataclass
class RunnerReport:
    """Aggregate result of running one agent over one task list.

    Attributes
    ----------
    agent_name:
        Identifier of the agent that ran.
    results:
        Per-task :class:`AgentResult` rows, in submission order.
    metrics:
        Aggregate :class:`MetricRow`.
    elapsed_seconds:
        Total wall-clock cost.
    """

    agent_name: str
    results: list[AgentResult] = field(default_factory=list)
    metrics: MetricRow | None = None
    elapsed_seconds: float = 0.0


# ─── runner ──────────────────────────────────────────────────────────────────


AgentFactory = Callable[[], BaseAgent]
"""Factory returning a *fresh* agent — the runner calls this once per task
when ``parallelism > 1`` so each thread holds its own agent.  For
``parallelism == 1`` the same agent can be reused and the factory is still
useful for clarity."""


class EvaluationRunner:
    """Run an agent over a task list and aggregate results.

    Parameters
    ----------
    agent_factory:
        Factory producing fresh :class:`BaseAgent` instances.  Called once
        per task in the parallel path and once total in the serial path.
    parallelism:
        Maximum simultaneous tasks.  ``1`` = strictly serial; larger
        values use a :class:`ThreadPoolExecutor`.
    progress_callback:
        Optional ``(completed, total) → None`` notifier for the CLI.
    """

    def __init__(
        self,
        *,
        agent_factory: AgentFactory,
        parallelism: int = 1,
        progress_callback: Callable[[int, int], None] | None = None,
        observer: RunObserver | None = None,
    ) -> None:
        if parallelism < 1:
            raise ValueError(f"parallelism must be ≥ 1, got {parallelism}")
        self._agent_factory = agent_factory
        self._parallelism = parallelism
        self._progress = progress_callback
        self._observer = observer

    def run(self, tasks: Iterable[DebuggingTask]) -> RunnerReport:
        task_list = list(tasks)
        self._observe(lambda o: o.on_run_started(task_count=len(task_list)))
        start = time.perf_counter()
        if self._parallelism == 1:
            results = self._run_serial(task_list)
        else:
            results = self._run_parallel(task_list)
        elapsed = time.perf_counter() - start
        self._observe(lambda o: o.on_run_finished())

        # All agents in the factory share an identifier; pull it from the
        # first result for the report header.
        agent_name = results[0].agent_name if results else "unknown"
        return RunnerReport(
            agent_name=agent_name,
            results=results,
            metrics=summarize_results(agent_name=agent_name, results=results),
            elapsed_seconds=elapsed,
        )

    # ── execution paths ──────────────────────────────────────────────────

    def _run_serial(self, tasks: list[DebuggingTask]) -> list[AgentResult]:
        agent = self._agent_factory()
        results: list[AgentResult] = []
        for i, task in enumerate(tasks, start=1):
            result = agent.run(task)
            self._emit_trace(task_id=task.task_id, result=result)
            results.append(result)
            self._notify(i, len(tasks))
        return results

    def _run_parallel(self, tasks: list[DebuggingTask]) -> list[AgentResult]:
        results: list[AgentResult | None] = [None] * len(tasks)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self._parallelism,
        ) as pool:
            future_to_index = {
                pool.submit(self._run_one, task): i
                for i, task in enumerate(tasks)
            }
            completed = 0
            for fut in concurrent.futures.as_completed(future_to_index):
                idx = future_to_index[fut]
                results[idx] = fut.result()
                completed += 1
                self._notify(completed, len(tasks))
        # Indices were populated in any order; cast away the None now that
        # we know every slot is filled.
        ordered = [r for r in results if r is not None]
        # Emit trace events in submission order so the JSON Lines file is
        # human-readable.  We do this *after* the parallel run so the
        # observer is invoked serially (writer locks are otherwise the
        # only contention point).
        for task, result in zip(tasks, ordered, strict=False):
            self._emit_trace(task_id=task.task_id, result=result)
        return ordered

    def _run_one(self, task: DebuggingTask) -> AgentResult:
        agent = self._agent_factory()
        return agent.run(task)

    def _notify(self, completed: int, total: int) -> None:
        if self._progress is not None:
            self._progress(completed, total)

    def _emit_trace(self, *, task_id: str, result: AgentResult) -> None:
        if self._observer is None:
            return
        for step in result.trace.steps:
            self._observe(lambda o, s=step: o.on_step(task_id=task_id, step=s))
        self._observe(lambda o: o.on_task_result(result=result))

    def _observe(self, fn: Callable[[RunObserver], None]) -> None:
        if self._observer is None:
            return
        try:
            fn(self._observer)
        except Exception:  # pragma: no cover - observer must not break runs
            pass
