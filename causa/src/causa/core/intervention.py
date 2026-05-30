"""Pearl's do(·) operator — the surgical view of intervention.

Pearl 2009, §3.2 — *Causal Effect as Intervention*:

    The intervention ``do(X = x)`` replaces the structural equation for X
    with the constant assignment X := x, severing every incoming edge to X.
    The resulting *mutilated* graph represents the post-intervention world.

This module gives us two things:

1. A typed :class:`Intervention` value object capturing
   ``do(variable = value)`` so the planning layer's audit trace can record
   every interventional query precisely.
2. A pure function :func:`mutilate` that returns the mutilated graph
   :math:`G_{\\overline{X}}` used by :mod:`causa.core.identifiability` for
   back-door / front-door checks.

We deliberately *do not* couple this to DoWhy here — DoWhy operates on the
mutilated graph internally; we keep the primitive accessible to tests and to
alternative estimators (the ablation §J3 in the dissertation requires
swappable estimators).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from causa.core.graph import CausalGraph


@dataclass(frozen=True)
class Intervention:
    """A single ``do(variable = value)`` assignment.

    Attributes
    ----------
    variable:
        Name of the variable being forced.
    value:
        The forced value.  Type is the SCM-defined domain of the variable.
    """

    variable: str
    value: Any

    def __str__(self) -> str:
        return f"do({self.variable}={self.value!r})"


def mutilate(graph: CausalGraph, intervention: Intervention) -> CausalGraph:
    """Return the post-intervention mutilated graph :math:`G_{\\overline{X}}`.

    Drops every incoming edge to the intervention variable.  Outgoing edges
    are preserved (the agent still expects the intervened variable to
    influence its downstream effects).

    Parameters
    ----------
    graph:
        The original causal graph.
    intervention:
        The ``do(X=x)`` assignment.

    Returns
    -------
    A new :class:`CausalGraph` with all edges entering ``intervention.variable``
    removed.  The input graph is left untouched.

    Examples
    --------
    >>> g = CausalGraph.from_edges([("A","B"), ("B","C"), ("D","B")])
    >>> g2 = mutilate(g, Intervention("B", 1))
    >>> sorted((e.source, e.target) for e in g2.edges)
    [('B', 'C')]
    """
    mutilated = graph.copy()
    for parent in graph.parents(intervention.variable):
        mutilated.remove_edge(parent, intervention.variable)
    return mutilated


def mutilate_multiple(graph: CausalGraph, interventions: list[Intervention]) -> CausalGraph:
    """Mutilate w.r.t. a *set* of simultaneous interventions.

    Useful for multi-step planning extensions (future work, §N1 in the
    dissertation).  Not used by the M.Tech-scope one-step greedy planner.
    """
    out = graph.copy()
    for iv in interventions:
        for parent in graph.parents(iv.variable):
            if parent in out:
                try:
                    out.remove_edge(parent, iv.variable)
                except Exception:  # noqa: BLE001
                    pass  # already removed by a prior intervention
    return out
