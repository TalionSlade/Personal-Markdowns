"""SWE-bench task loader.

Translates raw SWE-bench instance records into the lightweight
:class:`DebuggingTask` rows the agent actually sees.  The instance schema
we read is the public Princeton-NLP SWE-bench Verified format.

Design choices:

- The loader **does not** check out repositories or run tests.  Those are
  the eval-driver's concern.  This module turns one JSONL row into one
  :class:`DebuggingTask` so the agent loop can run end-to-end against a
  cache file alone.
- Partition axis assignment uses lightweight heuristics over the
  instance metadata (repo name, problem statement keywords).  The result
  is stable, auditable, and good enough for the OOD partitioner.
- The loader is robust to missing keys: anything absent gets a sentinel
  ``"unknown"`` rather than raising — that lets us run the harness
  against partial caches during development.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable

from causa.domain.tasks import DebuggingTask


# Heuristic regexes for axis classification.  Conservative: anything that
# doesn't match falls through to "unknown".
_LANG_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("python",  re.compile(r"\.py\b|pyproject|django|flask|sklearn|pandas", re.I)),
    ("javascript", re.compile(r"\.js\b|npm|node|react|vue", re.I)),
    ("typescript", re.compile(r"\.ts\b|tsconfig|nestjs", re.I)),
    ("java",    re.compile(r"\.java\b|maven|gradle|spring", re.I)),
    ("go",      re.compile(r"\.go\b|go\.mod|gopkg", re.I)),
    ("rust",    re.compile(r"\.rs\b|cargo\.toml|rustc", re.I)),
)

_FRAMEWORK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("django",  re.compile(r"\bdjango\b", re.I)),
    ("flask",   re.compile(r"\bflask\b", re.I)),
    ("sklearn", re.compile(r"\bsklearn|scikit-learn\b", re.I)),
    ("pandas",  re.compile(r"\bpandas\b", re.I)),
    ("react",   re.compile(r"\breact(?:\.js)?\b", re.I)),
    ("spring",  re.compile(r"\bspring(?:boot|framework)?\b", re.I)),
    ("rails",   re.compile(r"\brails\b|active(?:record|support)", re.I)),
)

_BUG_TYPE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("type_error",      re.compile(r"\bTypeError\b|type error|wrong type", re.I)),
    ("value_error",     re.compile(r"\bValueError\b|invalid value", re.I)),
    ("attribute_error", re.compile(r"\bAttributeError\b|has no attribute", re.I)),
    ("assertion_error", re.compile(r"\bAssertionError\b|expected .* got", re.I)),
    ("runtime_error",   re.compile(r"\bRuntimeError\b|segfault|crash", re.I)),
    ("logic_error",     re.compile(r"\b(?:logic|incorrect|wrong) (?:bug|behaviour|result)", re.I)),
    ("import_error",    re.compile(r"\bImportError\b|module not found", re.I)),
    ("ref_error",       re.compile(r"\bNameError\b|undefined|undeclared", re.I)),
)

_SIZE_BOUNDS = (("small", 1_000), ("medium", 50_000), ("large", float("inf")))


def load_swebench_tasks(
    path: Path,
    *,
    limit: int | None = None,
) -> list[DebuggingTask]:
    """Load a SWE-bench JSONL cache file into :class:`DebuggingTask` rows."""
    if not path.exists():
        raise FileNotFoundError(f"SWE-bench cache not found: {path}")
    tasks: list[DebuggingTask] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if not line.strip():
                continue
            if limit is not None and i >= limit:
                break
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            tasks.append(_record_to_task(obj))
    return tasks


def load_iter(records: Iterable[dict[str, Any]]) -> list[DebuggingTask]:
    """Translate an in-memory iterable of records — useful in tests."""
    return [_record_to_task(r) for r in records]


# ─── translation ─────────────────────────────────────────────────────────────


def _record_to_task(record: dict[str, Any]) -> DebuggingTask:
    instance_id = str(record.get("instance_id", record.get("id", "unknown")))
    problem = str(record.get("problem_statement", record.get("description", "")))
    repo = str(record.get("repo", ""))
    size_hint = int(record.get("repo_size_loc", 0) or 0)

    axes = {
        "language":      _classify(problem + " " + repo, _LANG_PATTERNS),
        "framework":     _classify(problem + " " + repo, _FRAMEWORK_PATTERNS),
        "bug_type":      _classify(problem, _BUG_TYPE_PATTERNS),
        "codebase_size": _bucket_size(size_hint),
    }
    error_type = axes["bug_type"] if axes["bug_type"] != "unknown" else "runtime_error"
    initial_state = {
        "error_message_type": error_type,
        "codebase_structure": _structure_from_size(axes["codebase_size"]),
        "context_available":  _context_from_problem(problem),
    }
    return DebuggingTask(
        task_id=instance_id,
        description=problem,
        initial_state=initial_state,
        partition_axes=axes,
    )


def _classify(text: str, patterns: tuple[tuple[str, re.Pattern[str]], ...]) -> str:
    for label, pat in patterns:
        if pat.search(text):
            return label
    return "unknown"


def _bucket_size(loc: int) -> str:
    if loc <= 0:
        return "unknown"
    for label, ceiling in _SIZE_BOUNDS:
        if loc < ceiling:
            return label
    return "large"


def _structure_from_size(size_label: str) -> str:
    return {
        "small": "small_flat",
        "medium": "medium_modular",
        "large": "large_layered",
    }.get(size_label, "medium_modular")


def _context_from_problem(problem: str) -> str:
    n_words = len(problem.split())
    if n_words >= 400:
        return "rich"
    if n_words >= 120:
        return "partial"
    return "none"
