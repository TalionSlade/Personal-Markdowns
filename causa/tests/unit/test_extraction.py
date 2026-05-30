"""Tests for the LLM-driven causal-graph extractor.

The extractor is the cold-start path for *novel* domains (§N3 domain
transfer).  The invariants it must defend:

* unknown variables silently rejected (no crash)
* self-loops silently rejected (no GraphInvariantError leaking out)
* edges that would form a cycle filtered into the ``rejected`` bucket
* the returned graph is always a valid DAG
* a non-JSON LLM response collapses to an empty edge set
"""

from __future__ import annotations

import json

from causa.extraction.llm_extractor import LLMGraphExtractor, ProposedEdge
from causa.ports.llm import LLMResponse


class _ScriptedLLM:
    """LLM stub that returns a single canned JSON payload."""

    model = "scripted-llm"

    def __init__(self, payload: str) -> None:
        self._payload = payload
        self.calls: list[str] = []

    def complete(self, messages, *, system=None, json_mode=False,  # noqa: ANN001, ARG002
                 max_tokens=1024, temperature=0.0):
        self.calls.append(messages[0].content)
        return LLMResponse(
            content=self._payload,
            model=self.model,
            input_tokens=10,
            output_tokens=10,
        )


def test_extractor_builds_dag_from_canonical_edges():
    payload = json.dumps({
        "edges": [
            {"source": "A", "target": "B", "semantics": "causes"},
            {"source": "B", "target": "C", "semantics": "causes"},
        ],
    })
    extractor = LLMGraphExtractor(llm=_ScriptedLLM(payload))
    graph, accepted, rejected = extractor.extract(
        variables=["A", "B", "C"], domain_description="toy chain",
    )
    assert len(graph.edges) == 2
    assert {e.source for e in graph.edges} == {"A", "B"}
    assert len(accepted) == 2
    assert rejected == []


def test_extractor_rejects_cycle_inducing_edges():
    payload = json.dumps({
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "B", "target": "C"},
            {"source": "C", "target": "A"},  # would close the cycle
        ],
    })
    extractor = LLMGraphExtractor(llm=_ScriptedLLM(payload))
    graph, accepted, rejected = extractor.extract(
        variables=["A", "B", "C"], domain_description="cycle-prone domain",
    )
    assert len(graph.edges) == 2  # cycle edge dropped
    assert len(accepted) == 2
    assert len(rejected) == 1
    assert rejected[0].source == "C" and rejected[0].target == "A"


def test_extractor_filters_unknown_variables():
    payload = json.dumps({
        "edges": [
            {"source": "A", "target": "B"},
            {"source": "ZZZ", "target": "B"},   # ZZZ not in variable set
            {"source": "A", "target": "QQQ"},   # QQQ not in variable set
        ],
    })
    extractor = LLMGraphExtractor(llm=_ScriptedLLM(payload))
    graph, accepted, rejected = extractor.extract(
        variables=["A", "B"], domain_description="filter-unknowns",
    )
    assert len(accepted) == 1
    assert len(rejected) == 2
    assert len(graph.edges) == 1


def test_extractor_handles_invalid_json_gracefully():
    extractor = LLMGraphExtractor(llm=_ScriptedLLM("not-json-at-all"))
    graph, accepted, rejected = extractor.extract(
        variables=["A", "B"], domain_description="garbage payload",
    )
    assert len(graph.edges) == 0
    assert accepted == []
    assert rejected == []


def test_extractor_passes_marker_to_llm():
    payload = json.dumps({"edges": []})
    llm = _ScriptedLLM(payload)
    extractor = LLMGraphExtractor(llm=llm)
    extractor.extract(variables=["A"], domain_description="any")
    assert llm.calls, "LLM was never invoked"
    assert "[CAUSA::extract_edges]" in llm.calls[0]


def test_proposed_edge_is_frozen():
    """Proposed edges are immutable — required for safe deduplication."""
    e = ProposedEdge(source="A", target="B", semantics="x")
    try:
        e.source = "C"  # type: ignore[misc]
    except Exception:  # noqa: BLE001
        return
    raise AssertionError("ProposedEdge should be frozen")
