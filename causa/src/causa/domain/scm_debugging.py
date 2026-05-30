"""The canonical 9-node, 8-edge SCM for software debugging.

This is the **load-bearing artifact** of the dissertation (§5 of the
abstract).  Edges and variable types are documented inline so the rendered
graph reads identically across:

- the auditable causal trace produced at runtime;
- the diagrams in :mod:`docs/wiki/diagrams/02-scm-debugging-domain.html`;
- the dissertation's slide 5.

If you edit this file, you almost certainly also need to edit:
- the HTML diagram cited above,
- the warm-start prior (`causa.planning.warm_start`) so synthetic priors
  remain coherent with the edges,
- the unit test :file:`tests/unit/test_scm_debugging.py`.
"""

from __future__ import annotations

from causa.core.scm import SCM, SCMBuilder, VariableDomain

DEBUGGING_SCM_NAME = "debugging_scm_v1"


# ─── observation schema (DataFrame column order = topological order) ──────────


def debugging_observation_schema() -> tuple[str, ...]:
    """Topological column order for the observation history DataFrame."""
    return (
        "error_message_type",
        "codebase_structure",
        "context_available",
        "hypothesis_space",
        "tool_selected",
        "information_gained",
        "root_cause_identified",
        "patch_quality",
        "tests_passed",
    )


# ─── canonical levels for categorical variables ──────────────────────────────


_ERROR_TYPES = (
    "type_error",
    "value_error",
    "ref_error",
    "import_error",
    "attribute_error",
    "assertion_error",
    "logic_error",
    "runtime_error",
)

_CODEBASE_STRUCTURES = ("small_flat", "medium_modular", "large_layered")

_CONTEXT_LEVELS = ("none", "partial", "rich")

_HYPOTHESIS_CLUSTERS = (
    "wrong_type",
    "missing_dep",
    "off_by_one",
    "edge_case",
    "config_mismatch",
    "logic_flaw",
)

_TOOLS = (
    "code_search",
    "static_analyzer",
    "test_runner",
    "log_inspector",
    "documentation_lookup",
    "patch_generator",
    "regression_checker",
)


def debugging_action_levels() -> tuple[str, ...]:
    return _TOOLS


# ─── the SCM ─────────────────────────────────────────────────────────────────


def build_debugging_scm() -> SCM:
    """Construct the canonical 9-node debugging SCM.

    Returns
    -------
    A fully-validated :class:`SCM`.

    The graph is identical to the one rendered on slide 5 of the viva
    deck.  Three observational inputs feed ``hypothesis_space``, which
    drives ``tool_selected`` (the do-variable).  Together with
    ``context_available``, the tool produces ``information_gained``, which
    feeds ``root_cause_identified``, ``patch_quality``, and finally the
    observable success signal ``tests_passed``.
    """
    b = SCMBuilder(DEBUGGING_SCM_NAME)

    # Observational inputs
    b.observational(
        "error_message_type", VariableDomain.CATEGORICAL,
        levels=_ERROR_TYPES,
        description="The kind of error encountered.  Sensed input.",
    )
    b.observational(
        "codebase_structure", VariableDomain.CATEGORICAL,
        levels=_CODEBASE_STRUCTURES,
        description="High-level structural classification of the repository.",
    )
    b.observational(
        "context_available", VariableDomain.ORDINAL,
        levels=_CONTEXT_LEVELS,
        description="How much surrounding context (open files, recent edits) is in scope.",
    )

    # Latent hypothesis state
    b.hypothesis(
        "hypothesis_space", VariableDomain.CATEGORICAL,
        levels=_HYPOTHESIS_CLUSTERS,
        description=(
            "Coarsened representation of the agent's current belief over "
            "candidate root causes (§A5 in the FAQ)."
        ),
    )

    # The do-variable
    b.action(
        "tool_selected", VariableDomain.CATEGORICAL,
        levels=_TOOLS,
        description=(
            "The tool the agent chooses to invoke at this step.  This is "
            "the do-variable — DoWhy computes P(tests_passed | do(tool_selected = a))."
        ),
    )

    # Mediators
    b.observational(
        "information_gained", VariableDomain.CONTINUOUS,
        description=(
            "Score in [0,1] measuring the novelty/usefulness of the tool's "
            "output.  Scored by an LLM-judge in production."
        ),
    )
    b.observational(
        "root_cause_identified", VariableDomain.BINARY,
        levels=("no", "yes"),
        description="Whether the agent has correctly localised the bug.",
    )
    b.observational(
        "patch_quality", VariableDomain.CONTINUOUS,
        description="Score in [0,1] for the quality of the proposed fix.",
    )

    # The outcome
    b.outcome(
        "tests_passed", VariableDomain.BINARY,
        levels=("fail", "pass"),
        description=(
            "Observable success signal — whether the patch makes the SWE-bench "
            "test suite pass.  Primary objective of the agent."
        ),
    )

    # Edges (8 total, left → right in the canonical layout)
    b.edge("error_message_type",  "hypothesis_space",      "error kind shapes candidate root causes")
    b.edge("codebase_structure",  "hypothesis_space",      "structural shape biases plausible failure modes")
    b.edge("hypothesis_space",    "tool_selected",         "beliefs about the bug drive tool choice")
    b.edge("tool_selected",       "information_gained",    "tool determines what is learned")
    b.edge("context_available",   "information_gained",    "surrounding context amplifies tool yield")
    b.edge("information_gained",  "root_cause_identified", "information accumulates into root-cause certainty")
    b.edge("root_cause_identified", "patch_quality",       "knowing the cause produces a better fix")
    b.edge("patch_quality",       "tests_passed",          "fix quality determines test outcome")

    return b.build()
