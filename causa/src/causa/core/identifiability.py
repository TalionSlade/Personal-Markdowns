"""Identification via the back-door (and front-door) criterion.

Identifiability — Pearl 2009, §3.3 — is the *graphical* property of a causal
query that decides whether :math:`P(Y \\mid do(X))` can be computed from
observational data plus the DAG alone.  If a query is *not* identifiable, no
amount of data fixes it — only an experiment does.

This module implements:

- **Back-door adjustment** — the workhorse used by DoWhy and by the
  Causal Planning Layer.  A set :math:`Z` satisfies the back-door
  criterion w.r.t. an ordered pair :math:`(X, Y)` iff:

    1. :math:`Z` blocks every back-door path between :math:`X` and :math:`Y`,
       i.e. every path with an arrow into :math:`X`;
    2. :math:`Z` contains no descendants of :math:`X`.

  If such :math:`Z` exists,

  .. math::
      P(Y \\mid do(X=x)) \\;=\\; \\sum_z P(Y \\mid X=x, Z=z) \\, P(Z=z).

- **Front-door** — only used as a contingency when back-door fails due to
  an unobserved confounder (dissertation §A5).

We compute admissible back-door sets *constructively*: try the empty set,
then singletons of confounders, then unions.  The agent's planning layer
calls :func:`identify_effect` once per candidate action.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from itertools import combinations

from causa.core.graph import CausalGraph
from causa.core.intervention import Intervention, mutilate


class IdentificationStrategy(str, Enum):
    BACKDOOR = "backdoor"
    FRONTDOOR = "frontdoor"
    UNIDENTIFIABLE = "unidentifiable"


@dataclass(frozen=True)
class BackdoorSet:
    """An admissible back-door adjustment set."""

    variables: frozenset[str]
    minimal: bool = False

    def __len__(self) -> int:
        return len(self.variables)


@dataclass
class IdentificationResult:
    """Outcome of an identifiability query for ``P(Y | do(X))``.

    Attributes
    ----------
    treatment:
        Name of the do-variable.
    outcome:
        Name of the outcome variable.
    strategy:
        Which adjustment strategy succeeded — back-door, front-door, or
        unidentifiable.
    adjustment_set:
        The variables to adjust on (empty for an effect with no back-door
        paths).  Frozen so it can be cached on a per-graph basis.
    candidate_sets:
        All admissible adjustment sets considered (useful in the dissertation
        for the "minimal vs full" adjustment comparison).
    """

    treatment: str
    outcome: str
    strategy: IdentificationStrategy
    adjustment_set: frozenset[str] = frozenset()
    candidate_sets: list[BackdoorSet] = field(default_factory=list)

    @property
    def identifiable(self) -> bool:
        return self.strategy is not IdentificationStrategy.UNIDENTIFIABLE


# ── back-door identification ──────────────────────────────────────────────────


def _is_backdoor_set(graph: CausalGraph, treatment: str, outcome: str,
                    z: set[str]) -> bool:
    """Test the two-clause back-door criterion (Pearl 2009 def 3.3.1).

    Clause 1 — :math:`Z` contains no descendants of treatment.
    Clause 2 — :math:`Z` blocks every back-door path from treatment to outcome,
              which is equivalent to:
              treatment and outcome are d-separated by :math:`Z` in the graph
              obtained by *removing outgoing edges from treatment*.
    """
    if z & graph.descendants(treatment):
        return False
    # Build G_X̲ (delete arrows *out of* X) for the back-door check.
    g_lower = graph.copy()
    for child in graph.children(treatment):
        g_lower.remove_edge(treatment, child)
    return g_lower.d_separated({treatment}, {outcome}, z)


def _candidate_pool(graph: CausalGraph, treatment: str, outcome: str) -> set[str]:
    """Variables that may appear in a back-door set.

    Heuristic: all non-descendants of the treatment, excluding the outcome
    itself.  Keeps the search space tractable for SCMs of dissertation scale
    (≤ 20 nodes).
    """
    pool = set(graph.nodes) - {treatment, outcome}
    pool -= graph.descendants(treatment)
    return pool


def _find_backdoor_sets(graph: CausalGraph, treatment: str, outcome: str,
                        max_size: int = 4) -> list[BackdoorSet]:
    """Enumerate admissible back-door sets up to ``max_size``.

    We start from size 0 (the empty set — is the effect already identifiable
    without adjustment?), then 1, 2, …, ``max_size``.  The first size at
    which we find admissible sets gives us the *minimal* ones.

    For the 9-node debugging SCM, max_size=4 is comfortably generous.
    """
    pool = _candidate_pool(graph, treatment, outcome)
    found: list[BackdoorSet] = []
    minimal_size: int | None = None
    for k in range(max_size + 1):
        if minimal_size is not None and k > minimal_size:
            break
        for combo in combinations(sorted(pool), k):
            z = set(combo)
            if _is_backdoor_set(graph, treatment, outcome, z):
                found.append(BackdoorSet(
                    variables=frozenset(z),
                    minimal=(minimal_size is None or k == minimal_size),
                ))
                if minimal_size is None:
                    minimal_size = k
    return found


def identify_effect(graph: CausalGraph, treatment: str, outcome: str,
                    *, prefer_minimal: bool = True) -> IdentificationResult:
    """Identify :math:`P(outcome \\mid do(treatment))` from the graph.

    Attempts back-door adjustment first (the common case in practice).
    Front-door is not yet implemented — returns ``UNIDENTIFIABLE`` if
    back-door fails.  When back-door succeeds, the *minimal* adjustment set
    is selected by default (smaller sets give lower-variance estimates with
    finite samples).

    Parameters
    ----------
    graph:
        The DAG over the SCM's variables.
    treatment:
        Name of the do-variable.
    outcome:
        Name of the outcome variable.
    prefer_minimal:
        If True (default) and several minimal-size sets exist, return the
        lexicographically first.  If False, the *full* admissible set
        (union of all minimal ones) is returned — useful when the
        estimator (e.g. a forest) handles more variables gracefully.

    Returns
    -------
    :class:`IdentificationResult` with the chosen adjustment set.

    Notes
    -----
    This is *graph-only* identification — no data is touched.  Numerical
    estimation lives in :mod:`causa.planning.dowhy_scorer`.
    """
    if treatment not in graph or outcome not in graph:
        raise KeyError(
            f"treatment {treatment!r} or outcome {outcome!r} not in graph"
        )

    # Sanity: outcome must be reachable from treatment, else effect is trivially 0
    if outcome not in graph.descendants(treatment):
        return IdentificationResult(
            treatment=treatment, outcome=outcome,
            strategy=IdentificationStrategy.BACKDOOR,
            adjustment_set=frozenset(),  # no causal effect → no adjustment needed
            candidate_sets=[BackdoorSet(frozenset(), minimal=True)],
        )

    candidates = _find_backdoor_sets(graph, treatment, outcome)
    if not candidates:
        return IdentificationResult(
            treatment=treatment, outcome=outcome,
            strategy=IdentificationStrategy.UNIDENTIFIABLE,
            adjustment_set=frozenset(),
            candidate_sets=[],
        )

    minimal = [c for c in candidates if c.minimal]
    chosen: frozenset[str]
    if prefer_minimal:
        # Lex-first among minimal sets for determinism
        chosen = sorted(minimal, key=lambda c: tuple(sorted(c.variables)))[0].variables
    else:
        chosen = frozenset.union(*(c.variables for c in minimal)) if minimal else frozenset()
    return IdentificationResult(
        treatment=treatment, outcome=outcome,
        strategy=IdentificationStrategy.BACKDOOR,
        adjustment_set=chosen,
        candidate_sets=candidates,
    )


def post_intervention_graph(graph: CausalGraph, intervention: Intervention) -> CausalGraph:
    """Convenience wrapper around :func:`causa.core.intervention.mutilate`.

    Returned graph is the one used by *all* downstream identification logic;
    we keep this re-export so callers don't reach across modules for it.
    """
    return mutilate(graph, intervention)
