"""Telemetry — JSON Lines causal-trace emitter.

§G of the dissertation requires every agent step to be *auditable*: a
reviewer should be able to read a trace and answer:

- "Which back-door set was used?"
- "Which alternatives did the scorer rank and how?"
- "Did the reflection module fire?  What synthetic rows did it write?"
- "What was the tool output?"

The :class:`TraceWriter` emits one JSON Lines event per agent step and
optional ``RUN_*`` framing events around the run.  JSON Lines was picked
because:

- Append-only fits the step-by-step write pattern;
- Each line is parseable in isolation — partial files are still useful;
- Trivial to load into pandas/Polars for the explainability study.

Schema versioning is explicit (``trace_version: 1``) so we can evolve the
event shape without breaking historical traces.
"""

from __future__ import annotations

from causa.telemetry.auditor import TraceAudit, audit_trace
from causa.telemetry.events import TraceEvent, build_run_event, build_step_event
from causa.telemetry.writer import NullTraceWriter, TraceWriter

__all__ = [
    "NullTraceWriter",
    "TraceAudit",
    "TraceEvent",
    "TraceWriter",
    "audit_trace",
    "build_run_event",
    "build_step_event",
]
