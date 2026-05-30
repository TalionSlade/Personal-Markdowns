"""Causa CLI — typer-based command dispatcher.

Commands are grouped by domain (SCM, agent, eval, trace).  Each command is
a thin wrapper that:

1. Loads :class:`CausaSettings` (env + .env);
2. Wires the requested adapters;
3. Hands off to a library function defined in the relevant module.

This keeps the CLI a transport layer, not a place where logic lives.  Unit
tests target the library code directly; integration tests invoke the CLI
via :class:`typer.testing.CliRunner`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from causa.cli.runtime import (
    AgentArm,
    build_agent,
    load_tasks,
    make_history,
)
from causa.config.settings import LLMProvider, load_settings
from causa.domain.scm_debugging import build_debugging_scm
from causa.evaluation.partition import (
    PartitionLabel,
    SplitPartitioner,
    default_policy,
)
from causa.evaluation.runner import EvaluationRunner
from causa.extraction.llm_extractor import LLMGraphExtractor
from causa.telemetry.auditor import audit_trace
from causa.telemetry.writer import NullTraceWriter, TraceWriter


app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    rich_markup_mode="rich",
    help="Causa — Causal Planning for LLM Agents.",
)
console = Console()


# ─── scm-show ────────────────────────────────────────────────────────────────


@app.command("scm-show")
def scm_show(
    fmt: str = typer.Option(
        "table",
        "--format",
        "-f",
        help="Output format: table | dot | json.",
    ),
) -> None:
    """Render the canonical 9-node debugging SCM."""
    scm = build_debugging_scm()
    if fmt == "dot":
        typer.echo(scm.graph.to_dot())
        return
    if fmt == "json":
        typer.echo(json.dumps({
            "nodes": list(scm.graph.nodes()),
            "edges": [
                {"source": e.source, "target": e.target, "semantics": e.semantics}
                for e in scm.graph.edges()
            ],
        }, indent=2))
        return

    table = Table(title=f"SCM: {scm.name}", show_lines=True)
    table.add_column("Variable", style="bold")
    table.add_column("Role")
    table.add_column("Domain")
    table.add_column("Levels", overflow="fold")
    for name in scm.graph.topological_order():
        v = scm.variables[name]
        levels = ", ".join(v.levels) if v.levels else "—"
        table.add_row(name, v.role.value, v.domain.value, levels)
    console.print(table)

    edge_table = Table(title="Edges (8)", show_lines=False)
    edge_table.add_column("Source")
    edge_table.add_column("Target")
    edge_table.add_column("Semantics", overflow="fold")
    for e in scm.graph.edges():
        edge_table.add_row(e.source, e.target, e.semantics or "—")
    console.print(edge_table)


# ─── scm-extract ─────────────────────────────────────────────────────────────


@app.command("scm-extract")
def scm_extract(
    description_file: Path = typer.Argument(
        ...,
        exists=True,
        file_okay=True,
        readable=True,
        help="Plain-text file with the domain description.",
    ),
    variables: str = typer.Option(
        ...,
        "--variables",
        "-v",
        help="Comma-separated variable names.",
    ),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Optional JSON output path.",
    ),
) -> None:
    """Run the LLM Graph Extractor on a free-form domain description."""
    from causa.cli.runtime import build_llm  # local import keeps startup fast

    settings = load_settings()
    llm = build_llm(settings)
    extractor = LLMGraphExtractor(llm=llm)
    description = description_file.read_text(encoding="utf-8")
    variable_list = [v.strip() for v in variables.split(",") if v.strip()]
    graph, accepted, rejected = extractor.extract(
        variables=variable_list, domain_description=description,
    )
    payload = {
        "variables": variable_list,
        "accepted": [
            {"source": e.source, "target": e.target, "semantics": e.semantics}
            for e in accepted
        ],
        "rejected": [
            {"source": e.source, "target": e.target, "semantics": e.semantics}
            for e in rejected
        ],
        "graph_edges": [
            {"source": e.source, "target": e.target}
            for e in graph.edges()
        ],
    }
    if out is not None:
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        console.print(f"Wrote {out}.")
    else:
        console.print_json(data=payload)


# ─── run ─────────────────────────────────────────────────────────────────────


@app.command("run")
def run_single(
    arm: AgentArm = typer.Option(
        AgentArm.CAUSAL, "--arm", "-a", help="Which agent to run.",
    ),
    tasks_path: Path = typer.Option(
        ...,
        "--tasks",
        "-t",
        exists=True,
        readable=True,
        help="JSON Lines file of tasks (SWE-bench format).",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Process at most N tasks.",
    ),
    parallelism: int = typer.Option(
        1, "--parallelism", "-p", help="Threadpool size for parallel runs.",
    ),
    trace_dir: Optional[Path] = typer.Option(
        None,
        "--trace-dir",
        help="Directory to write JSON Lines trace files into.",
    ),
) -> None:
    """Run one agent arm over a task file and print the metric row."""
    settings = load_settings()
    tasks = load_tasks(tasks_path, limit=limit)
    scm = build_debugging_scm()
    history = make_history(scm)

    agent = build_agent(arm, settings=settings, scm=scm, history=history)
    factory = lambda: agent  # reuse — runner only needs a callable

    writer: NullTraceWriter | TraceWriter
    if trace_dir is None:
        writer = NullTraceWriter()
    else:
        trace_dir.mkdir(parents=True, exist_ok=True)
        writer = TraceWriter(
            path=trace_dir / f"{arm.value}.jsonl",
            agent_name=agent.name,
        )

    runner = EvaluationRunner(
        agent_factory=factory, parallelism=parallelism, observer=writer,
    )
    report = runner.run(tasks)
    _print_metric_row(report.metrics)


# ─── eval ────────────────────────────────────────────────────────────────────


@app.command("eval")
def eval_suite(
    tasks_path: Path = typer.Option(
        ...,
        "--tasks",
        "-t",
        exists=True,
        readable=True,
        help="JSON Lines file of tasks.",
    ),
    arms: list[AgentArm] = typer.Option(
        list(AgentArm),
        "--arm",
        help="Subset of arms to run; repeat the flag.",
    ),
    limit: Optional[int] = typer.Option(
        None, "--limit", "-n", help="Process at most N tasks per arm.",
    ),
    partition: bool = typer.Option(
        True, "--partition/--no-partition",
        help="Split tasks into IID and OOD subsets before evaluation.",
    ),
    trace_dir: Optional[Path] = typer.Option(
        None, "--trace-dir", help="Optional trace output directory.",
    ),
) -> None:
    """Run a full ablation suite and print one row per arm × partition."""
    settings = load_settings()
    tasks = load_tasks(tasks_path, limit=limit)

    if partition:
        partitioner = SplitPartitioner(default_policy())
        iid, ood, _unlabeled = partitioner.split(tasks)
        partitions: dict[str, list] = {
            PartitionLabel.IID.value: iid,
            PartitionLabel.OOD.value: ood,
        }
    else:
        partitions = {"all": tasks}

    table = Table(title="Causa ablation suite", show_lines=True)
    table.add_column("Arm", style="bold")
    table.add_column("Partition")
    table.add_column("N")
    table.add_column("Success")
    table.add_column("Mean outcome")
    table.add_column("Steps→solve")
    table.add_column("Reflection rate")

    for arm in arms:
        scm = build_debugging_scm()
        for partition_name, partition_tasks in partitions.items():
            history = make_history(scm)
            agent = build_agent(arm, settings=settings, scm=scm, history=history)
            writer = _make_writer(trace_dir, arm.value, partition_name, agent.name)
            runner = EvaluationRunner(
                agent_factory=lambda agent=agent: agent,
                observer=writer,
            )
            report = runner.run(partition_tasks)
            m = report.metrics
            table.add_row(
                arm.value, partition_name, str(m.n_tasks),
                f"{m.success_rate:.3f}",
                f"{m.mean_outcome:.3f}" if m.mean_outcome is not None else "—",
                f"{m.mean_steps_to_success:.2f}" if m.mean_steps_to_success is not None else "—",
                f"{m.reflection_trigger_rate:.3f}" if m.reflection_trigger_rate is not None else "—",
            )

    console.print(table)


# ─── trace-audit ─────────────────────────────────────────────────────────────


@app.command("trace-audit")
def trace_audit_cmd(
    trace_path: Path = typer.Argument(
        ..., exists=True, readable=True, help="JSON Lines trace file.",
    ),
    fmt: str = typer.Option(
        "table", "--format", "-f", help="Output format: table | json.",
    ),
) -> None:
    """Summarise a JSON Lines trace file."""
    audit = audit_trace(trace_path)
    if fmt == "json":
        console.print_json(data=audit.to_dict())
        return

    table = Table(title=f"Trace: {trace_path.name}", show_lines=False)
    table.add_column("Field", style="bold")
    table.add_column("Value")
    table.add_row("run_id", str(audit.run_id))
    table.add_row("agent", str(audit.agent_name))
    table.add_row("n_tasks", str(audit.n_tasks))
    table.add_row("n_steps", str(audit.n_steps))
    table.add_row("reflection_triggers", str(audit.reflection_triggers))
    table.add_row("cold_start_steps", str(audit.cold_start_steps))
    table.add_row("steady_state_steps", str(audit.steady_state_steps))
    table.add_row(
        "top_actions",
        ", ".join(f"{a}={c}" for a, c in audit.action_counts.most_common(5)) or "—",
    )
    if audit.errors:
        table.add_row("errors", "; ".join(audit.errors[:3]))
    console.print(table)


# ─── helpers ─────────────────────────────────────────────────────────────────


def _make_writer(
    trace_dir: Path | None,
    arm: str,
    partition: str,
    agent_name: str,
) -> NullTraceWriter | TraceWriter:
    if trace_dir is None:
        return NullTraceWriter()
    trace_dir.mkdir(parents=True, exist_ok=True)
    return TraceWriter(
        path=trace_dir / f"{arm}.{partition}.jsonl",
        agent_name=agent_name,
    )


def _print_metric_row(metrics) -> None:  # noqa: ANN001
    if metrics is None:
        console.print("[yellow]No results produced.[/yellow]")
        return
    table = Table(show_header=True, show_lines=False)
    table.add_column("Metric", style="bold")
    table.add_column("Value")
    table.add_row("agent_name", metrics.agent_name)
    table.add_row("n_tasks", str(metrics.n_tasks))
    table.add_row("success_rate", f"{metrics.success_rate:.3f}")
    if metrics.mean_outcome is not None:
        table.add_row("mean_outcome", f"{metrics.mean_outcome:.3f}")
    if metrics.mean_steps_to_success is not None:
        table.add_row("mean_steps_to_success", f"{metrics.mean_steps_to_success:.2f}")
    if metrics.reflection_trigger_rate is not None:
        table.add_row("reflection_trigger_rate", f"{metrics.reflection_trigger_rate:.3f}")
    console.print(table)


# ─── entry point for ``python -m causa`` and ``causa`` console script ────────


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
