"""DAG wrapper with causal-inference invariants.

A *causal graph* is a Directed Acyclic Graph whose edges encode direct causal
influence.  This module wraps :class:`networkx.DiGraph` with:

- Strict acyclicity enforcement at construction time
- Convenience accessors for parents, children, ancestors, descendants
- d-separation, used downstream by :mod:`causa.core.identifiability`
- An immutable :meth:`copy` so downstream code never mutates the SCM by accident

We deliberately keep this layer dependency-light (NetworkX only) — it is the
*pure domain* and must be safe to import inside tight inference loops.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import networkx as nx


class GraphInvariantError(ValueError):
    """Raised when a graph operation would break a causal-graph invariant."""


@dataclass(frozen=True)
class CausalEdge:
    """An immutable directed edge in a causal graph.

    Attributes
    ----------
    source:
        Name of the parent (cause) node.
    target:
        Name of the child (effect) node.
    semantics:
        Free-form human-readable description (used in the auditable trace).
    """

    source: str
    target: str
    semantics: str = ""

    def __post_init__(self) -> None:
        if self.source == self.target:
            raise GraphInvariantError(f"self-loop: {self.source} -> {self.target}")


class CausalGraph:
    """A DAG wrapper enforcing causal-graph invariants.

    The class is intentionally thin: it exposes a small, precisely-typed
    surface to the rest of the system, and delegates raw algorithms to
    NetworkX.  Every mutation is checked for cycle creation.

    Invariants
    ----------
    1. The underlying :class:`networkx.DiGraph` is acyclic at all times.
    2. Edges carry optional ``semantics`` metadata for the causal trace.
    3. Self-loops are forbidden.
    """

    def __init__(self, nodes: Iterable[str] | None = None,
                 edges: Iterable[CausalEdge] | None = None) -> None:
        self._g: nx.DiGraph = nx.DiGraph()
        for n in nodes or ():
            self._g.add_node(n)
        for e in edges or ():
            self.add_edge(e)

    # ── construction ────────────────────────────────────────────────────────

    @classmethod
    def from_edges(cls, edges: Iterable[tuple[str, str]]) -> CausalGraph:
        """Convenience constructor from ``(parent, child)`` tuples."""
        g = cls()
        for src, tgt in edges:
            g.add_edge(CausalEdge(src, tgt))
        return g

    def add_node(self, name: str) -> None:
        self._g.add_node(name)

    def add_edge(self, edge: CausalEdge) -> None:
        """Add a directed edge; raise on cycle creation."""
        self._g.add_node(edge.source)
        self._g.add_node(edge.target)
        self._g.add_edge(edge.source, edge.target, semantics=edge.semantics)
        if not nx.is_directed_acyclic_graph(self._g):
            self._g.remove_edge(edge.source, edge.target)
            raise GraphInvariantError(
                f"adding edge {edge.source}->{edge.target} creates a cycle"
            )

    def remove_edge(self, source: str, target: str) -> None:
        self._g.remove_edge(source, target)

    # ── accessors ───────────────────────────────────────────────────────────

    @property
    def nodes(self) -> tuple[str, ...]:
        return tuple(self._g.nodes)

    @property
    def edges(self) -> tuple[CausalEdge, ...]:
        return tuple(
            CausalEdge(u, v, semantics=self._g[u][v].get("semantics", ""))
            for u, v in self._g.edges
        )

    def parents(self, node: str) -> set[str]:
        return set(self._g.predecessors(node))

    def children(self, node: str) -> set[str]:
        return set(self._g.successors(node))

    def ancestors(self, node: str) -> set[str]:
        return set(nx.ancestors(self._g, node))

    def descendants(self, node: str) -> set[str]:
        return set(nx.descendants(self._g, node))

    def topological_order(self) -> list[str]:
        return list(nx.topological_sort(self._g))

    # ── d-separation ────────────────────────────────────────────────────────

    def d_separated(self, x: set[str], y: set[str], z: set[str]) -> bool:
        """Return True iff ``z`` d-separates ``x`` from ``y`` in this DAG.

        Notes
        -----
        d-separation is the graphical criterion for conditional independence
        in a Bayesian network.  We use NetworkX's implementation, which is
        the standard moralised-ancestral-graph algorithm (Geiger, Verma &
        Pearl, 1990).
        """
        return nx.is_d_separator(self._g, x, y, z)

    # ── operations ──────────────────────────────────────────────────────────

    def copy(self) -> CausalGraph:
        """Return a deep, independent copy."""
        new = CausalGraph()
        new._g = self._g.copy()
        return new

    def to_networkx(self) -> nx.DiGraph:
        """Escape hatch for libraries (e.g. DoWhy) that need a raw DiGraph."""
        return self._g.copy()

    def to_dot(self, *, intervention_node: str | None = None) -> str:
        """Render to Graphviz DOT for ad-hoc visual inspection.

        The intervention node (the do-variable) is highlighted in purple
        so the rendered graph reads like the dissertation figures.
        """
        lines = ["digraph CausalGraph {", "  rankdir=LR;", "  node [shape=box, style=rounded];"]
        for n in self.topological_order():
            if n == intervention_node:
                lines.append(f'  "{n}" [style="rounded,filled", fillcolor="#3c228a", fontcolor="white"];')
            else:
                lines.append(f'  "{n}";')
        for e in self.edges:
            lines.append(f'  "{e.source}" -> "{e.target}";')
        lines.append("}")
        return "\n".join(lines)

    # ── dunders ─────────────────────────────────────────────────────────────

    def __iter__(self) -> Iterator[str]:
        return iter(self.topological_order())

    def __len__(self) -> int:
        return self._g.number_of_nodes()

    def __contains__(self, item: str) -> bool:
        return item in self._g

    def __repr__(self) -> str:
        return f"CausalGraph(nodes={len(self)}, edges={self._g.number_of_edges()})"
