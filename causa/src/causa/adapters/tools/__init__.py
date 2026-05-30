"""Debugging-tool adapters.

The dissertation's 6–8 debugging tools (§B6).  All ship as **stubbed**
adapters by default — they return realistic, deterministic synthetic
outputs so the agent loop can run end-to-end without subprocess overhead.
Real adapters (subprocess pytest, ripgrep, …) live alongside the stubs and
can be swapped via :func:`make_tool_registry`.
"""

from __future__ import annotations

from causa.adapters.tools.registry import make_tool_registry
from causa.adapters.tools.stubs import (
    StubCodeSearchTool,
    StubDocumentationLookupTool,
    StubLogInspectorTool,
    StubPatchGeneratorTool,
    StubRegressionCheckerTool,
    StubStaticAnalyzerTool,
    StubTestRunnerTool,
)

__all__ = [
    "StubCodeSearchTool",
    "StubDocumentationLookupTool",
    "StubLogInspectorTool",
    "StubPatchGeneratorTool",
    "StubRegressionCheckerTool",
    "StubStaticAnalyzerTool",
    "StubTestRunnerTool",
    "make_tool_registry",
]
