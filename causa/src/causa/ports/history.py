"""Observation history protocol.

The agent's observation history is the *data* DoWhy reads when estimating
:math:`P(Y \\mid do(X))` (§B4 in the dissertation).  We declare it as a
port so we can swap pandas (default) for SQLite or Parquet snapshots later
without touching the planning layer.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class ObservationHistory(Protocol):
    """Append-only history of agent observations.

    Each *row* is one decision-step's observed variable assignments.  The
    history is the back-door-adjustment data DoWhy is configured against.
    """

    @property
    def n_observations(self) -> int:
        """Number of rows currently in the history."""
        ...

    def append(self, observation: dict[str, Any]) -> None:
        """Append one observation.  ``observation`` keys must equal SCM vars."""
        ...

    def append_many(self, observations: list[dict[str, Any]]) -> None:
        """Bulk append (used by the warm-start prior generator)."""
        ...

    def as_dataframe(self) -> pd.DataFrame:
        """Return the history as a pandas DataFrame.

        Implementations may return a snapshot (recommended) or a view, but
        callers MUST treat the result as read-only.
        """
        ...

    def window(self, k: int) -> pd.DataFrame:
        """Return the *last* k observations as a DataFrame.

        Used by the planning layer to bound DoWhy query latency under long
        runs (§B4 in the dissertation).
        """
        ...

    def clear(self) -> None:
        """Reset the history (used by the eval runner between tasks)."""
        ...
