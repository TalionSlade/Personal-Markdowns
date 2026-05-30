"""Trace auditor — read .jsonl traces and produce audit summaries.

§G2 requires a *post hoc* auditor that loads a trace file and answers:

- How many steps used a back-door identified scorer (vs cold-start)?
- How many reflection updates fired across the run, and on which steps?
- What was the distribution of chosen actions?
- Were there any error events?

This is the function the explanation-quality human-eval study loads to
pull representative trace fragments for the rubric reviewers.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


@dataclass
class TraceAudit:
    """Aggregate summary of a single trace file."""

    path: Path
    run_id: str | None = None
    agent_name: str | None = None
    n_steps: int = 0
    n_tasks: int = 0
    action_counts: Counter[str] = field(default_factory=Counter)
    reflection_triggers: int = 0
    cold_start_steps: int = 0
    steady_state_steps: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "run_id": self.run_id,
            "agent_name": self.agent_name,
            "n_steps": self.n_steps,
            "n_tasks": self.n_tasks,
            "action_counts": dict(self.action_counts),
            "reflection_triggers": self.reflection_triggers,
            "cold_start_steps": self.cold_start_steps,
            "steady_state_steps": self.steady_state_steps,
            "errors": list(self.errors),
        }


def audit_trace(path: Path) -> TraceAudit:
    """Load a JSON Lines trace and return a :class:`TraceAudit`.

    Robust to truncated/partial files — invalid lines are skipped
    silently (matching the "partial files are still useful" property of
    the JSON Lines format).
    """
    audit = TraceAudit(path=path)
    for event in _iter_events(path):
        kind = event.get("kind")
        if kind == "run_started":
            audit.run_id = event.get("run_id")
            audit.agent_name = event.get("agent_name")
        elif kind == "step":
            audit.n_steps += 1
            action = event.get("chosen_action", "")
            if action:
                audit.action_counts[action] += 1
            reflection = event.get("reflection", "") or ""
            if "synthetic row" in reflection:
                audit.reflection_triggers += 1
            for score in event.get("scores", []):
                if score.get("action") != action:
                    continue
                rationale = score.get("rationale", "")
                if "[cold-start LLM]" in rationale:
                    audit.cold_start_steps += 1
                elif "[steady-state DoWhy]" in rationale or "via back-door" in rationale:
                    audit.steady_state_steps += 1
                break
        elif kind == "task_result":
            audit.n_tasks += 1
            err = event.get("error")
            if err:
                audit.errors.append(str(err))
    return audit


def _iter_events(path: Path) -> Iterable[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
