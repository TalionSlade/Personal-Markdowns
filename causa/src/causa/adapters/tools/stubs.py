"""Deterministic stub debugging tools.

Each stub returns a realistic :class:`ToolOutput` keyed off the task state's
hash.  Outputs include an ``information_score`` that varies by tool and by
state — exactly what the SCM's ``information_gained`` node needs to be
non-trivial.  No I/O, no subprocesses.

The seven tools below are the canonical set referenced throughout the
dissertation (§B6).  Adding new tools requires only adding a stub and
registering it in :mod:`causa.adapters.tools.registry`.
"""

from __future__ import annotations

import hashlib

from causa.ports.tool import DebuggingTool, ToolInput, ToolOutput


def _seed(state: dict[str, object], tool_name: str) -> int:
    h = hashlib.sha1(
        (repr(sorted(state.items())) + "|" + tool_name).encode(),
        usedforsecurity=False,
    ).hexdigest()
    return int(h, 16)


def _normalize(seed: int, lo: float, hi: float, *, shift: int = 0) -> float:
    return lo + (hi - lo) * (((seed >> shift) & 0xFF) / 255.0)


class _StubBase(DebuggingTool):
    name: str = ""
    description: str = ""
    _base_low: float = 0.20
    _base_high: float = 0.60
    _affinity: tuple[str, ...] = ()  # error types this tool is particularly good for

    def __call__(self, input: ToolInput, *, state: dict[str, object]) -> ToolOutput:
        seed = _seed(state, self.name)
        info = _normalize(seed, self._base_low, self._base_high)
        error_type = str(state.get("error_message_type", ""))
        if any(a in error_type for a in self._affinity):
            info = min(1.0, info + 0.25)
        return ToolOutput(
            success=True,
            payload={"args": input.args, "stub": self.name},
            information_score=round(info, 3),
            elapsed_seconds=round(_normalize(seed, 0.05, 0.45, shift=8), 3),
        )


class StubCodeSearchTool(_StubBase):
    name = "code_search"
    description = (
        "Locate definitions, references, or string matches across the repository."
    )
    _affinity = ("ref_error", "import_error")


class StubStaticAnalyzerTool(_StubBase):
    name = "static_analyzer"
    description = "Run a static analyser (type-checking, lint) over a target file."
    _affinity = ("type_error",)


class StubTestRunnerTool(_StubBase):
    name = "test_runner"
    description = "Execute the project's test suite and return failure traces."
    _affinity = ("assertion_error", "logic_error")


class StubLogInspectorTool(_StubBase):
    name = "log_inspector"
    description = "Inspect runtime logs from a recent execution."
    _affinity = ("runtime_error",)


class StubDocumentationLookupTool(_StubBase):
    name = "documentation_lookup"
    description = "Look up library/framework documentation for a symbol."
    _affinity = ("attribute_error", "import_error")


class StubPatchGeneratorTool(_StubBase):
    name = "patch_generator"
    description = "Generate a candidate code patch resolving the root cause."
    _base_low = 0.35
    _base_high = 0.85


class StubRegressionCheckerTool(_StubBase):
    name = "regression_checker"
    description = "Verify a candidate patch does not break other tests."
    _base_low = 0.25
    _base_high = 0.55
