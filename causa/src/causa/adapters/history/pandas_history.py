"""Pandas-backed observation history.

Append-only DataFrame with O(1) appends amortised via row-buffering.  The
buffer is flushed to the DataFrame on first read so callers see a single
authoritative snapshot.

Memory note: at SWE-bench scale (≤ ~10k decisions per evaluation run) this
fits comfortably in RAM.  For longer runs the windowed view
(:meth:`window`) keeps DoWhy queries fast.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import pandas as pd


class PandasObservationHistory:
    """In-memory append-only history of SCM observations.

    The class is *not* thread-safe — the agent loop is sequential.

    Notes
    -----
    Schema:
        The DataFrame's columns are the SCM's variable names.  Schema is
        established on first :meth:`append` from the keys of the first
        observation; subsequent appends are checked for schema equality.
    """

    def __init__(self, *, schema: Iterable[str] | None = None) -> None:
        self._schema: list[str] | None = list(schema) if schema is not None else None
        self._rows: list[dict[str, Any]] = []
        self._df_cache: pd.DataFrame | None = None

    # ── reads ──────────────────────────────────────────────────────────────

    @property
    def n_observations(self) -> int:
        return len(self._rows)

    def as_dataframe(self) -> pd.DataFrame:
        if self._df_cache is not None and len(self._df_cache) == len(self._rows):
            return self._df_cache
        df = pd.DataFrame(self._rows, columns=self._schema) if self._rows else pd.DataFrame(
            columns=self._schema or [],
        )
        self._df_cache = df
        return df

    def window(self, k: int) -> pd.DataFrame:
        if k <= 0:
            raise ValueError(f"window size must be positive, got {k}")
        if k >= self.n_observations:
            return self.as_dataframe()
        df = pd.DataFrame(self._rows[-k:], columns=self._schema)
        return df

    # ── writes ─────────────────────────────────────────────────────────────

    def append(self, observation: dict[str, Any]) -> None:
        self._validate(observation)
        self._rows.append(dict(observation))
        self._df_cache = None

    def append_many(self, observations: list[dict[str, Any]]) -> None:
        for o in observations:
            self._validate(o)
        self._rows.extend(dict(o) for o in observations)
        self._df_cache = None

    def clear(self) -> None:
        self._rows.clear()
        self._df_cache = None

    # ── helpers ────────────────────────────────────────────────────────────

    def _validate(self, observation: dict[str, Any]) -> None:
        if self._schema is None:
            self._schema = list(observation.keys())
            return
        if set(observation.keys()) != set(self._schema):
            missing = set(self._schema) - set(observation.keys())
            extra = set(observation.keys()) - set(self._schema)
            raise ValueError(
                f"observation schema mismatch: missing={missing}, extra={extra}"
            )

    def __repr__(self) -> str:
        return f"PandasObservationHistory(n={self.n_observations}, schema={self._schema})"
