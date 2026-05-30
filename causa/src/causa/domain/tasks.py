"""Typed task structures for evaluation.

A :class:`DebuggingTask` is the unit of work the agent operates on.  The
fields here are deliberately small — what the agent *sees* — and not the
SWE-bench instance metadata; the latter lives in :mod:`causa.evaluation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


@dataclass(frozen=True)
class DebuggingTask:
    """One debugging instance the agent attempts to solve.

    Attributes
    ----------
    task_id:
        Unique identifier (e.g. ``swe-bench/sklearn-1234``).
    description:
        Natural-language problem statement (the GitHub issue body).
    initial_state:
        Values for the SCM's observational inputs at task start
        (``error_message_type``, ``codebase_structure``, ``context_available``).
    partition_axes:
        Distribution-shift partition labels — used by the eval suite to
        bucket tasks into in/out-of-distribution sets.  Keys typically
        include ``language``, ``framework``, ``bug_type``, ``codebase_size``.
    """

    task_id: str
    description: str
    initial_state: dict[str, Any] = field(default_factory=dict)
    partition_axes: dict[str, str] = field(default_factory=dict)


class TaskOutcome(str, Enum):
    """Terminal outcome of a task attempt."""

    SOLVED = "solved"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    ERROR = "error"
