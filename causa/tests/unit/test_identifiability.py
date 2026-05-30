"""Tests for back-door identification.

The canonical confounder pattern is

    Z → X → Y
    Z → Y

The back-door criterion: adjusting for Z identifies P(Y | do(X)).  The
empty set does *not* identify it.  These tests pin those properties.
"""

from __future__ import annotations

from causa.core.graph import CausalGraph
from causa.core.identifiability import (
    IdentificationStrategy,
    identify_effect,
)


def test_no_confounder_returns_empty_set_identifiable():
    """When the only path X→Y has no back-door, ∅ identifies the effect."""
    g = CausalGraph.from_edges([("X", "Y")])
    result = identify_effect(g, "X", "Y")
    assert result.identifiable
    assert result.strategy is IdentificationStrategy.BACKDOOR
    assert result.adjustment_set == frozenset()


def test_confounded_chain_requires_z_adjustment():
    g = CausalGraph.from_edges([("Z", "X"), ("X", "Y"), ("Z", "Y")])
    result = identify_effect(g, "X", "Y")
    assert result.identifiable
    assert result.adjustment_set == frozenset({"Z"})


def test_unreachable_outcome_returns_trivial_effect():
    """When X does not lead to Y, the effect is identifiable as 0 with ∅."""
    g = CausalGraph.from_edges([("A", "B"), ("X", "C")])
    g.add_node("Y")
    result = identify_effect(g, "X", "Y")
    assert result.identifiable
    assert result.adjustment_set == frozenset()


def test_two_independent_confounders_yield_pair():
    """Two confounders that each open a back-door require *both* in Z."""
    g = CausalGraph.from_edges([
        ("Z1", "X"), ("Z1", "Y"),
        ("Z2", "X"), ("Z2", "Y"),
        ("X", "Y"),
    ])
    result = identify_effect(g, "X", "Y")
    assert result.identifiable
    assert result.adjustment_set == frozenset({"Z1", "Z2"})


def test_descendant_of_treatment_excluded_from_adjustment():
    """Conditioning on a descendant of X (a mediator) is invalid; the
    algorithm must pick the parent confounder instead."""
    g = CausalGraph.from_edges([
        ("Z", "X"), ("Z", "Y"),
        ("X", "M"), ("M", "Y"),  # M is a descendant
    ])
    result = identify_effect(g, "X", "Y")
    assert result.identifiable
    assert "M" not in result.adjustment_set
    assert result.adjustment_set == frozenset({"Z"})


def test_minimal_is_preferred_over_superset():
    g = CausalGraph.from_edges([
        ("Z", "X"), ("Z", "Y"),
        ("U", "Z"),  # superset {Z, U} also blocks but is non-minimal
        ("X", "Y"),
    ])
    result = identify_effect(g, "X", "Y")
    assert result.adjustment_set == frozenset({"Z"})
