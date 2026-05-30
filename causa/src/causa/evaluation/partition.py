"""Distribution-shift partitioner.

§F2 of the dissertation: the central claim is robustness under shift.  To
test it, every SWE-bench task is labelled along four axes:

    language      — python | js | ts | java | go | rust | …
    framework     — django | flask | react | rails | spring | … | none
    bug_type      — type | value | runtime | logic | assertion | …
    codebase_size — small | medium | large

A :class:`PartitionPolicy` defines *which* combination of axes constitutes
the In-Distribution (IID) set and which constitutes Out-Of-Distribution
(OOD) sets.  The default split — used in the primary experiment — leaves
out the largest codebase-size bucket *and* the framework that contributes
the most tasks, so OOD covers both *novel framework* and *novel size*.

The partitioner is deterministic given the task list; it does not depend
on any randomness.  This is important: the OOD generalisation result must
be reproducible without seed bookkeeping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from causa.domain.tasks import DebuggingTask


class PartitionAxis(str, Enum):
    """One of the four distribution-shift axes."""

    LANGUAGE = "language"
    FRAMEWORK = "framework"
    BUG_TYPE = "bug_type"
    CODEBASE_SIZE = "codebase_size"


class PartitionLabel(str, Enum):
    """Membership label for a partitioned task."""

    IID = "iid"
    OOD = "ood"


@dataclass(frozen=True)
class PartitionPolicy:
    """Declarative policy for splitting tasks into IID and OOD.

    Parameters
    ----------
    held_out:
        Mapping ``axis → set of values``.  A task whose ``partition_axes``
        contain *any* held-out value on *any* axis is classified OOD.
    require_axes:
        Axes that MUST be present on every task; tasks missing one are
        flagged as ``unlabeled`` (returned separately so the caller can
        decide whether to drop them).
    name:
        Human-readable identifier carried into the eval report.
    """

    held_out: dict[PartitionAxis, frozenset[str]] = field(default_factory=dict)
    require_axes: tuple[PartitionAxis, ...] = (
        PartitionAxis.LANGUAGE,
        PartitionAxis.FRAMEWORK,
        PartitionAxis.BUG_TYPE,
        PartitionAxis.CODEBASE_SIZE,
    )
    name: str = "default"

    def assign(self, task: DebuggingTask) -> PartitionLabel | None:
        """Return the label, or ``None`` if the task is unlabelable."""
        for axis in self.require_axes:
            if axis.value not in task.partition_axes:
                return None
        for axis, held in self.held_out.items():
            if task.partition_axes.get(axis.value, "") in held:
                return PartitionLabel.OOD
        return PartitionLabel.IID


@dataclass(frozen=True)
class SplitPartitioner:
    """Apply a :class:`PartitionPolicy` to a task list and return splits."""

    policy: PartitionPolicy

    def split(
        self,
        tasks: Iterable[DebuggingTask],
    ) -> tuple[list[DebuggingTask], list[DebuggingTask], list[DebuggingTask]]:
        """Return ``(iid, ood, unlabeled)`` task lists."""
        iid: list[DebuggingTask] = []
        ood: list[DebuggingTask] = []
        unlabeled: list[DebuggingTask] = []
        for t in tasks:
            label = self.policy.assign(t)
            if label is None:
                unlabeled.append(t)
            elif label is PartitionLabel.IID:
                iid.append(t)
            else:
                ood.append(t)
        return iid, ood, unlabeled


# ─── canonical policies ──────────────────────────────────────────────────────


def default_policy() -> PartitionPolicy:
    """The dissertation's primary OOD split (§F2).

    Holds out the largest codebase bucket *and* a chosen framework so the
    OOD set tests *both* size and framework novelty simultaneously.
    """
    return PartitionPolicy(
        held_out={
            PartitionAxis.CODEBASE_SIZE: frozenset({"large"}),
            PartitionAxis.FRAMEWORK: frozenset({"spring"}),
        },
        name="primary_size+framework",
    )


def language_only_policy() -> PartitionPolicy:
    """Stricter shift: hold out an entire language for cross-language transfer."""
    return PartitionPolicy(
        held_out={PartitionAxis.LANGUAGE: frozenset({"rust", "go"})},
        name="cross_language",
    )


def bug_type_policy() -> PartitionPolicy:
    """Hold out a bug type — the agent has never seen it during the
    warm-start prior but must still infer correct interventional choices."""
    return PartitionPolicy(
        held_out={PartitionAxis.BUG_TYPE: frozenset({"logic_error"})},
        name="held_out_bug_type",
    )
