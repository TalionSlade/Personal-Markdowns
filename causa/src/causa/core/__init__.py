"""Pure causal-inference primitives.

This package is deliberately I/O-free, dependency-light, and side-effect free:
- :mod:`causa.core.graph` — DAG wrapper around NetworkX with invariants
- :mod:`causa.core.scm`   — Structural Causal Model (U, V, F, P(U))
- :mod:`causa.core.intervention` — Pearl's do(·) operator
- :mod:`causa.core.identifiability` — Back-door / front-door criteria

These primitives are what the rest of the system composes against.  They are
written to be testable as pure functions and citable in the dissertation.
"""

from __future__ import annotations

from causa.core.graph import CausalGraph
from causa.core.identifiability import BackdoorSet, IdentificationResult, identify_effect
from causa.core.intervention import Intervention, mutilate
from causa.core.scm import SCM, StructuralEquation, Variable, VariableType

__all__ = [
    "BackdoorSet",
    "CausalGraph",
    "IdentificationResult",
    "Intervention",
    "SCM",
    "StructuralEquation",
    "Variable",
    "VariableType",
    "identify_effect",
    "mutilate",
]
