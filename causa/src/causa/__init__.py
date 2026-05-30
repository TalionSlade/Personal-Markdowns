"""Causa — Causal Planning for LLM Agents.

A reference implementation of Pearl's do-calculus action selection inside an
LLM agent loop, with online causal-belief update via counterfactual reflection,
evaluated against SWE-bench under controlled distribution shift.

Author : Arpan Ghosh (BITS Pilani, M.Tech. AI/ML, May 2026)
Thesis : Causal Planning for LLM Agents — A Framework for Robust
         Decision-Making Under Distribution Shift
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Arpan Ghosh"
__all__ = ["__version__", "__author__"]
