"""Tests for the do(·) operator and graph mutilation."""

from __future__ import annotations

from causa.core.graph import CausalGraph
from causa.core.intervention import Intervention, mutilate, mutilate_multiple


def test_mutilate_drops_incoming_edges_only():
    """do(B=1) removes edges *into* B; outgoing edges from B survive."""
    g = CausalGraph.from_edges([("A", "B"), ("B", "C"), ("D", "B")])
    g2 = mutilate(g, Intervention("B", 1))
    edges = sorted((e.source, e.target) for e in g2.edges)
    assert edges == [("B", "C")]


def test_mutilate_returns_new_graph():
    """The input graph must not be mutated by the function."""
    g = CausalGraph.from_edges([("A", "B"), ("B", "C")])
    pre_edges = set((e.source, e.target) for e in g.edges)
    _ = mutilate(g, Intervention("B", "x"))
    post_edges = set((e.source, e.target) for e in g.edges)
    assert pre_edges == post_edges


def test_mutilate_multiple_handles_overlapping_parents():
    g = CausalGraph.from_edges([
        ("A", "B"), ("B", "C"), ("D", "B"), ("D", "C"),
    ])
    g2 = mutilate_multiple(g, [Intervention("B", 1), Intervention("C", 2)])
    edges = sorted((e.source, e.target) for e in g2.edges)
    # All four original edges target B or C, so all are dropped.
    assert edges == []


def test_intervention_str_includes_variable_and_value():
    assert str(Intervention("tool_selected", "test_runner")) == "do(tool_selected='test_runner')"
