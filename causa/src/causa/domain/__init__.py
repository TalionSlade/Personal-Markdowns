"""Domain-specific artefacts for software debugging.

The contents of this package are the *application of the framework* to the
dissertation's chosen domain.  They are deliberately kept separate from
:mod:`causa.core` so the framework can be reused for other domains (§N3 in
the dissertation) without disturbing the debugging instantiation.

- :mod:`causa.domain.scm_debugging` — the hand-authored 9-node SCM
- :mod:`causa.domain.tools` — the canonical 7-tool action space
- :mod:`causa.domain.tasks` — typed task structures for evaluation
"""

from __future__ import annotations

from causa.domain.scm_debugging import (
    DEBUGGING_SCM_NAME,
    build_debugging_scm,
    debugging_action_levels,
    debugging_observation_schema,
)
from causa.domain.tasks import DebuggingTask, TaskOutcome

__all__ = [
    "DEBUGGING_SCM_NAME",
    "DebuggingTask",
    "TaskOutcome",
    "build_debugging_scm",
    "debugging_action_levels",
    "debugging_observation_schema",
]
