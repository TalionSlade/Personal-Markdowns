"""Tests for the CausalGraph wrapper around NetworkX."""

from __future__ import annotations

import pytest

from causa.core.graph import CausalEdge, CausalGraph, GraphInvariantError


def test_empty_graph_has_zero_nodes():
    g = CausalGraph()
    assert len(g) == 0
    assert len(g.edges) == 0


def test_add_edge_creates_nodes():
    g = CausalGraph()
    g.add_edge(CausalEdge("a", "b"))
    assert "a" in g.nodes
    assert "b" in g.nodes
    assert len(g.edges) == 1


def test_self_loop_rejected_at_edge_construction():
    """:class:`CausalEdge` raises on self-loop in ``__post_init__``."""
    with pytest.raises(GraphInvariantError):
        CausalEdge("a", "a")


def test_cycle_rejected():
    g = CausalGraph()
    g.add_edge(CausalEdge("a", "b"))
    g.add_edge(CausalEdge("b", "c"))
    with pytest.raises(GraphInvariantError):
        g.add_edge(CausalEdge("c", "a"))
    # And after the failure, the graph is still in its pre-cycle state.
    assert len(g.edges) == 2


def test_topological_order_is_consistent():
    g = CausalGraph()
    g.add_edge(CausalEdge("a", "b"))
    g.add_edge(CausalEdge("b", "c"))
    g.add_edge(CausalEdge("a", "c"))
    order = g.topological_order()
    assert order.index("a") < order.index("b") < order.index("c")


def test_parents_returns_direct_parents_only():
    g = CausalGraph()
    g.add_edge(CausalEdge("a", "b"))
    g.add_edge(CausalEdge("b", "c"))
    assert g.parents("c") == {"b"}
    assert g.parents("b") == {"a"}
    assert g.parents("a") == set()


def test_descendants_includes_transitive_children():
    g = CausalGraph()
    g.add_edge(CausalEdge("a", "b"))
    g.add_edge(CausalEdge("b", "c"))
    g.add_edge(CausalEdge("a", "d"))
    assert g.descendants("a") == {"b", "c", "d"}


def test_d_separation_chain():
    """In a→b→c, conditioning on b d-separates a from c."""
    g = CausalGraph.from_edges([("a", "b"), ("b", "c")])
    assert g.d_separated({"a"}, {"c"}, {"b"})
    assert not g.d_separated({"a"}, {"c"}, set())


def test_d_separation_collider():
    """In a→b←c, b is a collider; conditioning on it *unblocks* a and c."""
    g = CausalGraph.from_edges([("a", "b"), ("c", "b")])
    assert g.d_separated({"a"}, {"c"}, set())
    assert not g.d_separated({"a"}, {"c"}, {"b"})


def test_to_dot_highlights_intervention_node():
    g = CausalGraph.from_edges([("a", "b"), ("b", "c")])
    dot = g.to_dot(intervention_node="b")
    assert "#3c228a" in dot  # the brand purple is the highlight colour
    assert '"b"' in dot
    assert '"a" -> "b"' in dot


def test_copy_is_independent():
    g = CausalGraph.from_edges([("a", "b")])
    g2 = g.copy()
    g2.add_edge(CausalEdge("b", "c"))
    assert "c" not in g.nodes
    assert "c" in g2.nodes
