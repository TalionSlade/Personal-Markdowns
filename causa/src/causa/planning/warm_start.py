"""Synthetic warm-start prior generator.

§B8 of the dissertation: before the agent has acted, DoWhy has no data and
cannot estimate effects.  The warm-start prior seeds the observation
history with synthetic rows generated from the SCM's structural form.

Two strategies:

- **Mechanistic**: if the SCM has structural equations
  (:attr:`SCM.equations`), simulate them with NumPy.  This is the
  preferred path for the debugging SCM, which we give simple sigmoidal
  mechanisms in :mod:`causa.domain.scm_debugging` (TODO: real f's; for
  now we use a structured-noise generator that respects the DAG).
- **LLM-elicited**: ask the LLM to generate plausible rows consistent with
  the SCM's variables.  Used when no mechanisms are present.

The warm-start prior is a *known limitation* (it's the LLM's prior, not
the true data distribution).  The §J3#4 ablation measures its
contribution.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from causa.core.scm import SCM, VariableDomain, VariableType


class SyntheticWarmStart:
    """Generate synthetic observations consistent with an SCM's DAG.

    The generator follows topological order — each variable's value depends
    on its parents in the SCM, so the synthetic distribution is consistent
    with the graph's conditional-independence structure (which is the
    invariant DoWhy needs to identify effects).

    Determinism: given ``seed``, the output is reproducible.
    """

    def __init__(self, *, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate(self, scm: SCM, n: int) -> list[dict[str, Any]]:
        if n < 0:
            raise ValueError(f"n must be ≥ 0, got {n}")
        order = scm.graph.topological_order()
        rows: list[dict[str, Any]] = []
        for _ in range(n):
            row: dict[str, Any] = {}
            for var_name in order:
                var = scm.variables[var_name]
                row[var_name] = self._sample(var, row, scm)
            rows.append(row)
        return rows

    # ── private samplers ───────────────────────────────────────────────────

    def _sample(self, var: Any, row: dict[str, Any], scm: SCM) -> Any:  # noqa: ANN401
        """Sample one variable given its parents' values."""
        parents = scm.graph.parents(var.name)
        parent_signature = "|".join(f"{p}={row.get(p)!r}" for p in sorted(parents))
        h = int(hashlib.sha1(
            (var.name + parent_signature).encode(), usedforsecurity=False,
        ).hexdigest()[:8], 16)
        # Mix in RNG so we don't collapse across rows
        h ^= int(self._rng.integers(0, 2**31))

        match var.domain:
            case VariableDomain.CATEGORICAL | VariableDomain.ORDINAL:
                return var.levels[h % len(var.levels)]
            case VariableDomain.BINARY:
                # Bias outcome variables on patch_quality if present
                base = 0.5
                if var.role == VariableType.OUTCOME and "patch_quality" in row:
                    pq = float(row["patch_quality"])
                    base = 0.15 + 0.7 * pq
                return var.levels[1 if self._rng.random() < base else 0] if var.levels else (
                    1 if self._rng.random() < base else 0
                )
            case VariableDomain.CONTINUOUS:
                # Derive from parents when available
                base = 0.5
                if "information_gained" in parents and "information_gained" in row:
                    base = 0.2 + 0.7 * float(row["information_gained"])
                elif "root_cause_identified" in parents and "root_cause_identified" in row:
                    rci = row["root_cause_identified"]
                    base = 0.7 if (rci == "yes" or rci == 1) else 0.3
                # tool-conditioned noise
                value = float(np.clip(self._rng.normal(loc=base, scale=0.1), 0.0, 1.0))
                return round(value, 3)
            case _:  # pragma: no cover
                return None
