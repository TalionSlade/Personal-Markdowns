"""Causal Graph Extractor (Component 1).

LLM-assisted construction of a causal DAG from a domain description.  Used
when the framework is applied to a *new* domain — the debugging SCM is
hand-authored, but the extractor is what lets the framework generalise
(§4 of the abstract).
"""

from __future__ import annotations

from causa.extraction.llm_extractor import LLMGraphExtractor, ProposedEdge

__all__ = ["LLMGraphExtractor", "ProposedEdge"]
