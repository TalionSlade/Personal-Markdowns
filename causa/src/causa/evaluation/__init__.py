"""Evaluation harness — the experimental scaffolding around the agents.

Four modules:

- :mod:`causa.evaluation.partition` — the four-axis distribution-shift
  partitioner that splits SWE-bench into IID and OOD subsets (§F2).
- :mod:`causa.evaluation.runner` — orchestrates an agent over a task list
  and aggregates :class:`AgentResult` rows.
- :mod:`causa.evaluation.metrics` — the metric set (success rate, mean
  outcome, steps to success, reflection trigger rate, identifiability
  rate, calibration).
- :mod:`causa.evaluation.stats` — paired permutation tests and BCa
  bootstrap confidence intervals (§F3).

The runner is intentionally a pure function of (agent, task list,
settings); the data plane (where results land) is the caller's concern,
which keeps the runner unit-testable without storage fixtures.
"""

from __future__ import annotations

from causa.evaluation.metrics import EvaluationMetrics, MetricRow, summarize_results
from causa.evaluation.partition import (
    PartitionAxis,
    PartitionLabel,
    PartitionPolicy,
    SplitPartitioner,
)
from causa.evaluation.runner import EvaluationRunner, RunnerReport
from causa.evaluation.stats import (
    BootstrapInterval,
    PairedTestResult,
    bca_bootstrap_mean,
    paired_permutation_test,
)
from causa.evaluation.swebench import load_iter, load_swebench_tasks

__all__ = [
    "BootstrapInterval",
    "EvaluationMetrics",
    "EvaluationRunner",
    "MetricRow",
    "PairedTestResult",
    "PartitionAxis",
    "PartitionLabel",
    "PartitionPolicy",
    "RunnerReport",
    "SplitPartitioner",
    "bca_bootstrap_mean",
    "load_iter",
    "load_swebench_tasks",
    "paired_permutation_test",
    "summarize_results",
]
