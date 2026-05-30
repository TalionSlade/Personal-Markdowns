"""Tool registry — default debugging tool set."""

from __future__ import annotations

from causa.adapters.tools.stubs import (
    StubCodeSearchTool,
    StubDocumentationLookupTool,
    StubLogInspectorTool,
    StubPatchGeneratorTool,
    StubRegressionCheckerTool,
    StubStaticAnalyzerTool,
    StubTestRunnerTool,
)
from causa.ports.tool import DebuggingTool


def make_tool_registry() -> dict[str, DebuggingTool]:
    """Return the canonical 7-tool debugging action space."""
    tools: list[DebuggingTool] = [
        StubCodeSearchTool(),
        StubStaticAnalyzerTool(),
        StubTestRunnerTool(),
        StubLogInspectorTool(),
        StubDocumentationLookupTool(),
        StubPatchGeneratorTool(),
        StubRegressionCheckerTool(),
    ]
    return {t.name: t for t in tools}
