"""Tests for the SCM data structure and builder."""

from __future__ import annotations

import pytest

from causa.core.scm import (
    SCM,
    SCMBuilder,
    StructuralEquation,
    Variable,
    VariableDomain,
    VariableType,
)


def test_variable_requires_levels_for_categorical():
    with pytest.raises(ValueError, match="categorical"):
        Variable(
            name="x", role=VariableType.OBSERVATIONAL,
            domain=VariableDomain.CATEGORICAL, levels=(),
        )


def test_binary_with_wrong_levels_rejected():
    with pytest.raises(ValueError, match="binary"):
        Variable(
            name="y", role=VariableType.OUTCOME,
            domain=VariableDomain.BINARY, levels=("one", "two", "three"),
        )


def test_builder_constructs_minimal_valid_scm():
    b = SCMBuilder("test")
    b.observational("x", VariableDomain.CATEGORICAL, levels=("a", "b"))
    b.action("a", VariableDomain.CATEGORICAL, levels=("hit", "stand"))
    b.outcome("y", VariableDomain.BINARY, levels=("lose", "win"))
    b.edge("x", "a")
    b.edge("a", "y")
    scm = b.build()
    assert scm.action_variable.name == "a"
    assert scm.outcome_variable.name == "y"


def test_validate_rejects_two_action_variables():
    b = SCMBuilder("test")
    b.observational("x", VariableDomain.CATEGORICAL, levels=("a", "b"))
    b.action("a1", VariableDomain.CATEGORICAL, levels=("p", "q"))
    b.action("a2", VariableDomain.CATEGORICAL, levels=("p", "q"))
    b.outcome("y", VariableDomain.BINARY, levels=("lose", "win"))
    b.edge("a1", "y")
    with pytest.raises(ValueError, match="ACTION"):
        b.build()


def test_set_equation_must_match_graph_parents():
    b = SCMBuilder("test")
    b.observational("x", VariableDomain.CATEGORICAL, levels=("a", "b"))
    b.action("a", VariableDomain.CATEGORICAL, levels=("p", "q"))
    b.outcome("y", VariableDomain.BINARY, levels=("lose", "win"))
    b.edge("x", "a")
    b.edge("a", "y")
    scm = b.build()
    # equation parent list disagrees with the graph
    with pytest.raises(ValueError, match="disagree"):
        scm.set_equation(StructuralEquation(target="y", parents=("x",), f=None))


def test_summary_returns_one_line_per_count_section():
    b = SCMBuilder("test")
    b.observational("x", VariableDomain.CATEGORICAL, levels=("a", "b"))
    b.action("a", VariableDomain.CATEGORICAL, levels=("p", "q"))
    b.outcome("y", VariableDomain.BINARY, levels=("lose", "win"))
    b.edge("x", "a")
    b.edge("a", "y")
    scm = b.build()
    text = scm.summary()
    assert "variables: 3" in text
    assert "edges: 2" in text
