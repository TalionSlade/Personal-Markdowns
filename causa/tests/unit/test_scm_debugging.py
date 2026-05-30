"""Tests pinning the canonical 9-node debugging SCM.

The SCM is the *load-bearing artifact* of the dissertation; these tests
fail loudly the moment a teammate edits the graph without updating the
intended structure.
"""

from __future__ import annotations

from causa.core.identifiability import identify_effect
from causa.core.scm import VariableType
from causa.domain.scm_debugging import (
    build_debugging_scm,
    debugging_action_levels,
    debugging_observation_schema,
)


CANONICAL_NODES = {
    "error_message_type", "codebase_structure", "context_available",
    "hypothesis_space", "tool_selected",
    "information_gained", "root_cause_identified", "patch_quality",
    "tests_passed",
}


def test_node_set_is_canonical():
    scm = build_debugging_scm()
    assert set(scm.graph.nodes) == CANONICAL_NODES


def test_edge_count_is_eight():
    scm = build_debugging_scm()
    assert len(scm.graph.edges) == 8


def test_action_and_outcome_are_pinned():
    scm = build_debugging_scm()
    assert scm.action_variable.name == "tool_selected"
    assert scm.outcome_variable.name == "tests_passed"
    assert scm.action_variable.role is VariableType.ACTION
    assert scm.outcome_variable.role is VariableType.OUTCOME


def test_schema_is_topological():
    """The observation-schema column order must respect the DAG."""
    scm = build_debugging_scm()
    schema = debugging_observation_schema()
    topo = scm.graph.topological_order()
    schema_idx = {name: i for i, name in enumerate(schema)}
    topo_idx = {name: i for i, name in enumerate(topo)}
    # For every edge u→v in the graph, schema(u) must precede schema(v).
    for e in scm.graph.edges:
        assert schema_idx[e.source] < schema_idx[e.target]
        assert topo_idx[e.source] < topo_idx[e.target]


def test_effect_on_outcome_is_identifiable():
    """P(tests_passed | do(tool_selected = a)) must be identifiable
    on the hand-authored graph."""
    scm = build_debugging_scm()
    result = identify_effect(
        scm.graph, "tool_selected", "tests_passed",
    )
    assert result.identifiable


def test_action_levels_match_seven_canonical_tools():
    levels = debugging_action_levels()
    assert len(levels) == 7
    assert "patch_generator" in levels
    assert "test_runner" in levels
