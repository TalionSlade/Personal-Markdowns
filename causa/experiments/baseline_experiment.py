"""Baseline experiment — compare all agent arms on synthetic debugging tasks.

Runs CAUSAL, CAUSAL_NO_REFLECTION, LLM_SCORER, REACT, COT, NO_MEMORY on
60 synthetic tasks covering all combinations of the three observational SCM
inputs.  Uses MockLLMClient (no API cost, fully reproducible).

Usage
-----
    cd causa
    python experiments/baseline_experiment.py

Output
------
    - Console results table (copy straight into the mid-sem report).
    - experiments/results/baseline_YYYYMMDD_HHMMSS.json  (for audit/reuse).
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from itertools import product
from pathlib import Path

# Ensure src/ is on the path when run from the repo root.
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from causa.adapters.history.pandas_history import PandasObservationHistory
from causa.cli.runtime import AgentArm, build_agent
from causa.config.settings import CausaSettings
from causa.domain.scm_debugging import (
    build_debugging_scm,
    debugging_action_levels,
    debugging_observation_schema,
)
from causa.domain.tasks import DebuggingTask, TaskOutcome
from causa.evaluation.metrics import EvaluationMetrics, summarize_results
from causa.evaluation.runner import EvaluationRunner


# ─── task generation ──────────────────────────────────────────────────────────

_ERROR_TYPES = (
    "type_error", "value_error", "ref_error", "import_error",
    "attribute_error", "assertion_error", "logic_error", "runtime_error",
)
_CODEBASE_STRUCTURES = ("small_flat", "medium_modular", "large_layered")
_CONTEXT_LEVELS = ("none", "partial", "rich")


def make_synthetic_tasks(n_repeats: int = 1) -> list[DebuggingTask]:
    """Return 70 × n_repeats synthetic debugging tasks.

    Each repeat re-uses the same 70 combinations (8×3×3 − 2 OOD holdouts) with
    a repeat-index suffix in the task_id and description so they are treated as
    independent samples by the evaluation runner.  n_repeats=2 → 140 tasks,
    n_repeats=3 → 210 tasks, etc.
    """
    tasks = []
    i = 0
    for rep in range(n_repeats):
        for err, cb, ctx in product(_ERROR_TYPES, _CODEBASE_STRUCTURES, _CONTEXT_LEVELS):
            # Skip the 2 OOD holdouts (logic_error/runtime_error × large_layered × none).
            if err in ("logic_error", "runtime_error") and cb == "large_layered" and ctx == "none":
                continue
            suffix = f" [rep {rep + 1}/{n_repeats}]" if n_repeats > 1 else ""
            tasks.append(DebuggingTask(
                task_id=f"synth/{i:03d}",
                description=(
                    f"Debugging task: {err} in a {cb} codebase "
                    f"with {ctx} context available.{suffix}"
                ),
                initial_state={
                    "error_message_type": err,
                    "codebase_structure": cb,
                    "context_available": ctx,
                },
                partition_axes={
                    "error_type": err,
                    "codebase": cb,
                    "context": ctx,
                },
            ))
            i += 1
    return tasks


# ─── arm runner ───────────────────────────────────────────────────────────────


@dataclass
class ArmResult:
    arm: str
    n_tasks: int
    success_rate: float
    mean_outcome: float | None
    mean_steps_to_success: float | None
    reflection_trigger_rate: float | None
    elapsed_seconds: float


def run_arm(
    arm: AgentArm,
    tasks: list[DebuggingTask],
    *,
    settings: CausaSettings,
    verbose: bool = False,
) -> ArmResult:
    scm = build_debugging_scm()
    history = PandasObservationHistory(schema=debugging_observation_schema())

    def agent_factory():
        return build_agent(arm, settings=settings, scm=scm, history=history)

    completed = 0
    def _progress(done: int, total: int) -> None:
        nonlocal completed
        completed = done
        if verbose:
            print(f"\r    task {done}/{total}", end="", flush=True)

    runner = EvaluationRunner(
        agent_factory=agent_factory, parallelism=1, progress_callback=_progress,
    )
    t0 = time.perf_counter()
    report = runner.run(tasks)
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"\r    {completed}/{completed} tasks done" + " " * 10)

    m = report.metrics
    return ArmResult(
        arm=arm.value,
        n_tasks=m.n_tasks if m else len(tasks),
        success_rate=m.success_rate if m else 0.0,
        mean_outcome=m.mean_outcome if m else None,
        mean_steps_to_success=m.mean_steps_to_success if m else None,
        reflection_trigger_rate=m.reflection_trigger_rate if m else None,
        elapsed_seconds=round(elapsed, 2),
    )


# ─── display ─────────────────────────────────────────────────────────────────


def _fmt(val: float | None, decimals: int = 3, pct: bool = False) -> str:
    if val is None:
        return "—"
    if pct:
        return f"{val * 100:.1f}%"
    return f"{val:.{decimals}f}"


def print_results_table(results: list[ArmResult]) -> None:
    col_widths = [24, 8, 13, 13, 19, 21, 10]
    headers = [
        "Arm", "N", "Success %", "Mean out.", "Steps (solved)", "Reflect. rate", "Time (s)",
    ]
    sep = "+" + "+".join("-" * w for w in col_widths) + "+"
    fmt_row = "|" + "|".join(f"{{:<{w}}}" for w in col_widths) + "|"

    print()
    print("=" * 80)
    print("  CAUSA — Baseline Experiment Results")
    print("=" * 80)
    print(sep)
    print(fmt_row.format(*headers))
    print(sep)
    for r in results:
        row = [
            r.arm,
            str(r.n_tasks),
            _fmt(r.success_rate, pct=True),
            _fmt(r.mean_outcome),
            _fmt(r.mean_steps_to_success, decimals=2),
            _fmt(r.reflection_trigger_rate, pct=True) if r.reflection_trigger_rate is not None else "N/A (baseline)",
            str(r.elapsed_seconds),
        ]
        print(fmt_row.format(*row))
    print(sep)
    print()


# ─── main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider", default="auto",
        help="LLM provider: mock | openai | anthropic | auto (reads .env, default mock)",
    )
    parser.add_argument("--n-tasks", type=int, default=None,
                        help="Cap on number of tasks (default: all 70)")
    parser.add_argument("--fast", action="store_true",
                        help="Fast mode: 1 reflection sample, step_budget=6 (for live demos)")
    args = parser.parse_args()

    # Resolve provider: 'auto' reads from env/.env; anything else overrides.
    env_kwargs: dict = dict(
        warm_start_prior_size=40,
        dowhy_min_history=10,
        step_budget=6 if args.fast else 12,
        reflection_threshold=0.15,
        reflection_samples=1 if args.fast else 3,
        random_seed=42,
        success_threshold=0.9,
    )
    if args.provider != "auto":
        env_kwargs["llm_provider"] = args.provider
        if args.provider == "mock":
            env_kwargs["llm_model"] = "mock-llm-v1"

    settings = CausaSettings(**env_kwargs)

    print(f"\n  Provider : {settings.llm_provider.value}")
    print(f"  Model    : {settings.llm_model}")

    all_tasks = make_synthetic_tasks()
    tasks = all_tasks[:args.n_tasks] if args.n_tasks else all_tasks
    print(f"  Tasks    : {len(tasks)}\n")

    arms_to_run = [
        AgentArm.CAUSAL,
        AgentArm.CAUSAL_NO_REFLECTION,
        AgentArm.LLM_SCORER,
        AgentArm.REACT,
        AgentArm.COT,
        AgentArm.NO_MEMORY,
    ]

    is_live = settings.llm_provider.value != "mock"
    results: list[ArmResult] = []
    for arm in arms_to_run:
        print(f"  Running arm: {arm.value:30s}", end="", flush=True)
        r = run_arm(arm, tasks, settings=settings, verbose=is_live)
        results.append(r)
        print(f"  done  success={r.success_rate*100:.1f}%  ({r.elapsed_seconds}s)")

    print_results_table(results)

    # Persist results for report/audit use.
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    provider_tag = settings.llm_provider.value
    out_path = out_dir / f"baseline_{provider_tag}_{stamp}.json"
    out_path.write_text(
        json.dumps(
            {
                "timestamp": stamp,
                "provider": provider_tag,
                "model": settings.llm_model,
                "n_tasks": len(tasks),
                "settings": settings.model_dump(mode="json"),
                "results": [asdict(r) for r in results],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Results saved -> {out_path}")


if __name__ == "__main__":
    main()
