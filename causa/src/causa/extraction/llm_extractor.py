"""LLM-driven causal-edge extractor.

Given a *domain description* (free-form text identifying variables and
their context), this extractor:

1. Prompts the LLM with the variable list and a request for candidate
   directed edges plus per-edge semantics;
2. Parses the response into a list of :class:`ProposedEdge`;
3. Filters edges that would introduce cycles;
4. Builds a validated :class:`CausalGraph`.

For the dissertation's debugging domain, the canonical edges are
hand-authored — the extractor is exercised on *novel* domains for the
domain-transfer experiments (§N3).
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from causa.core.graph import CausalEdge, CausalGraph, GraphInvariantError
from causa.ports.llm import LLMClient, LLMMessage, LLMRole


SYSTEM_PROMPT = (
    "You are a causal-graph extractor.  Given a list of variable names plus "
    "a domain description, propose directed causal edges between them.  "
    'Return a JSON object {"edges": [{"source": str, "target": str, '
    '"semantics": str}, ...]}.  Use only the provided variable names.  '
    "Do not propose self-loops.  Aim for a sparse graph: prefer leaving an "
    "edge out if you are unsure."
)


@dataclass(frozen=True)
class ProposedEdge:
    """An LLM-proposed edge — may not survive cycle filtering."""

    source: str
    target: str
    semantics: str = ""


class LLMGraphExtractor:
    """Component 1: produce a :class:`CausalGraph` from a textual description."""

    def __init__(self, *, llm: LLMClient) -> None:
        self._llm = llm

    def extract(
        self,
        *,
        variables: list[str],
        domain_description: str,
    ) -> tuple[CausalGraph, list[ProposedEdge], list[ProposedEdge]]:
        """Extract a graph from a description.

        Returns
        -------
        graph:
            The validated DAG (cycle-free).
        accepted:
            Edges that survived cycle filtering.
        rejected:
            Edges discarded because they would have created a cycle.
        """
        prompt = self._render_prompt(variables=variables, description=domain_description)
        response = self._llm.complete(
            messages=[LLMMessage(role=LLMRole.USER, content=prompt)],
            system=SYSTEM_PROMPT,
            json_mode=True,
            max_tokens=2048,
        )
        proposed = self._parse(response.content)
        return self._build_graph(variables, proposed)

    # ── helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _render_prompt(*, variables: list[str], description: str) -> str:
        return (
            "[CAUSA::extract_edges]\n\n"
            f"Domain description:\n{description}\n\n"
            f"Variables:\n" + "\n".join(f"- {v}" for v in variables)
        )

    @staticmethod
    def _parse(content: str) -> list[ProposedEdge]:
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return []
        edges = data.get("edges", []) if isinstance(data, dict) else []
        out: list[ProposedEdge] = []
        for e in edges:
            if not isinstance(e, dict):
                continue
            src, tgt = e.get("source"), e.get("target")
            if not isinstance(src, str) or not isinstance(tgt, str):
                continue
            out.append(ProposedEdge(source=src, target=tgt,
                                    semantics=str(e.get("semantics", ""))))
        return out

    @staticmethod
    def _build_graph(
        variables: list[str],
        proposed: list[ProposedEdge],
    ) -> tuple[CausalGraph, list[ProposedEdge], list[ProposedEdge]]:
        var_set = set(variables)
        graph = CausalGraph(nodes=variables)
        accepted: list[ProposedEdge] = []
        rejected: list[ProposedEdge] = []
        for edge in proposed:
            if edge.source not in var_set or edge.target not in var_set:
                rejected.append(edge)
                continue
            try:
                graph.add_edge(CausalEdge(edge.source, edge.target, edge.semantics))
            except GraphInvariantError:
                rejected.append(edge)
            else:
                accepted.append(edge)
        return graph, accepted, rejected
