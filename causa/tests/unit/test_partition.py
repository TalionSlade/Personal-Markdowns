"""Tests for the distribution-shift partitioner."""

from __future__ import annotations

from causa.domain.tasks import DebuggingTask
from causa.evaluation.partition import (
    PartitionAxis,
    PartitionLabel,
    PartitionPolicy,
    SplitPartitioner,
    default_policy,
)


def _task(task_id: str, **axes: str) -> DebuggingTask:
    return DebuggingTask(task_id=task_id, description="", partition_axes=axes)


def test_iid_task_is_classified_iid():
    policy = default_policy()
    t = _task(
        "iid-1", language="python", framework="django",
        bug_type="type_error", codebase_size="small",
    )
    assert policy.assign(t) is PartitionLabel.IID


def test_held_out_framework_marks_task_ood():
    policy = default_policy()
    t = _task(
        "ood-1", language="java", framework="spring",
        bug_type="type_error", codebase_size="small",
    )
    assert policy.assign(t) is PartitionLabel.OOD


def test_missing_required_axis_returns_none():
    policy = default_policy()
    t = _task("u-1", language="python")  # framework / bug_type / size missing
    assert policy.assign(t) is None


def test_split_partitioner_yields_three_buckets():
    policy = default_policy()
    splitter = SplitPartitioner(policy)
    tasks = [
        _task("iid-1", language="python", framework="django", bug_type="t", codebase_size="small"),
        _task("ood-1", language="java", framework="spring", bug_type="t", codebase_size="small"),
        _task("u-1",   language="python"),
    ]
    iid, ood, unlabeled = splitter.split(tasks)
    assert [t.task_id for t in iid] == ["iid-1"]
    assert [t.task_id for t in ood] == ["ood-1"]
    assert [t.task_id for t in unlabeled] == ["u-1"]


def test_custom_policy_holds_out_size_only():
    policy = PartitionPolicy(
        held_out={PartitionAxis.CODEBASE_SIZE: frozenset({"large"})},
        name="size-only",
    )
    big = _task("big", language="x", framework="y", bug_type="z", codebase_size="large")
    small = _task("sml", language="x", framework="y", bug_type="z", codebase_size="small")
    assert policy.assign(big) is PartitionLabel.OOD
    assert policy.assign(small) is PartitionLabel.IID
