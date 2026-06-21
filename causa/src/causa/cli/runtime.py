"""Runtime wiring helpers — the dependency-injection layer for the CLI.

Centralising the construction logic here means:

- the CLI commands stay declarative — they list inputs, not assembly steps;
- the ablation arms (§J3) are a one-place edit when adding a new arm;
- tests can call :func:`build_agent` directly without booting Typer.

If you add a new agent arm, the change is:

1. Add a member to :class:`AgentArm`.
2. Add the corresponding branch in :func:`build_agent`.
3. (Optional) add the arm's name to the CLI's ``--arm`` enum help text.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

from causa.adapters.history.pandas_history import PandasObservationHistory
from causa.adapters.tools.registry import make_tool_registry
from causa.agents.base import BaseAgent
from causa.agents.causal import CausalPlanningAgent
from causa.agents.chain_of_thought import ChainOfThoughtAgent
from causa.agents.no_memory import NoMemoryAgent
from causa.agents.react import ReActAgent
from causa.config.settings import CausaSettings
from causa.core.scm import SCM
from causa.domain.scm_debugging import debugging_observation_schema
from causa.domain.tasks import DebuggingTask
from causa.evaluation.swebench import load_swebench_tasks
from causa.planning.dowhy_scorer import DoWhyActionScorer
from causa.planning.hybrid_scorer import HybridActionScorer
from causa.planning.llm_scorer import LLMActionScorer
from causa.planning.warm_start import SyntheticWarmStart
from causa.ports.history import ObservationHistory
from causa.ports.llm import LLMClient
from causa.reflection.counterfactual import CounterfactualReflectionModule
from causa.reflection.threshold import StaticThreshold


# ─── arm enum ────────────────────────────────────────────────────────────────


class AgentArm(str, Enum):
    """The §J3 ablation arms, plus the proposed system."""

    CAUSAL = "causal"
    """The proposed system: do-calculus + reflection."""

    CAUSAL_NO_REFLECTION = "causal_no_reflection"
    """Component 2 only; tests the contribution of Component 3 (§J3#3)."""

    LLM_SCORER = "llm_scorer"
    """LLM correlational scorer + the rest of the loop (§J3#2)."""

    REACT = "react"
    """Yao et al. 2023 baseline."""

    COT = "cot"
    """Wei et al. 2022 chain-of-thought baseline."""

    NO_MEMORY = "no_memory"
    """Floor baseline — memoryless LLM (§J3#5)."""


# ─── builders ────────────────────────────────────────────────────────────────


def build_llm(settings: CausaSettings) -> LLMClient:
    """Build the configured LLM client (delegates to the adapter factory)."""
    from causa.adapters.llm.factory import make_llm_client  # noqa: PLC0415
    return make_llm_client(settings)


def build_estimator(settings: CausaSettings):  # noqa: ANN201
    """Build the configured causal estimator."""
    from causa.adapters.estimators.factory import make_estimator
    return make_estimator(settings.dowhy_estimator)


def make_history(scm: SCM) -> ObservationHistory:
    """Construct a fresh, schema-checked history seeded with warm-start rows."""
    history = PandasObservationHistory(schema=debugging_observation_schema())
    return history


def seed_warm_start(
    *,
    history: ObservationHistory,
    scm: SCM,
    settings: CausaSettings,
) -> None:
    """Populate the history with the configured warm-start prior size."""
    if settings.warm_start_prior_size <= 0:
        return
    generator = SyntheticWarmStart(seed=settings.random_seed)
    rows = generator.generate(scm, n=settings.warm_start_prior_size)
    history.append_many(rows)


def build_agent(
    arm: AgentArm,
    *,
    settings: CausaSettings,
    scm: SCM,
    history: ObservationHistory,
) -> BaseAgent:
    """Materialise one of the :class:`AgentArm` configurations."""
    tools = make_tool_registry()
    action_var = scm.action_variable.name
    outcome_var = scm.outcome_variable.name
    llm = build_llm(settings)

    # All arms operate on a history that has been warm-started — the
    # canonical configuration the dissertation reports.  The §J3#4 ablation
    # against this is one config flip away (set warm_start_prior_size=0).
    seed_warm_start(history=history, scm=scm, settings=settings)

    threshold = settings.success_threshold

    match arm:
        case AgentArm.CAUSAL | AgentArm.CAUSAL_NO_REFLECTION:
            estimator = build_estimator(settings)
            dowhy = DoWhyActionScorer(scm=scm, estimator=estimator)
            scorer = HybridActionScorer(
                cold_start=LLMActionScorer(llm=llm),
                steady_state=dowhy,
                min_history=settings.dowhy_min_history,
            )
            reflection = None if arm is AgentArm.CAUSAL_NO_REFLECTION else (
                CounterfactualReflectionModule(
                    llm=llm,
                    threshold=StaticThreshold(value=settings.reflection_threshold),
                    samples=settings.reflection_samples,
                    action_variable=action_var,
                    outcome_variable=outcome_var,
                )
            )
            return CausalPlanningAgent(
                scorer=scorer,
                reflection=reflection,
                history=history,
                tools=tools,
                step_budget=settings.step_budget,
                action_variable=action_var,
                outcome_variable=outcome_var,
                success_threshold=threshold,
                name=f"causa.{arm.value}",
            )

        case AgentArm.LLM_SCORER:
            estimator = build_estimator(settings)
            scorer = LLMActionScorer(llm=llm)
            return CausalPlanningAgent(
                scorer=scorer,
                reflection=None,
                history=history,
                tools=tools,
                step_budget=settings.step_budget,
                action_variable=action_var,
                outcome_variable=outcome_var,
                success_threshold=threshold,
                name="causa.llm_scorer",
            )

        case AgentArm.REACT:
            return ReActAgent(
                llm=llm,
                tools=tools,
                step_budget=settings.step_budget,
                action_variable=action_var,
                outcome_variable=outcome_var,
                success_threshold=threshold,
            )

        case AgentArm.COT:
            return ChainOfThoughtAgent(
                llm=llm,
                tools=tools,
                step_budget=settings.step_budget,
                action_variable=action_var,
                outcome_variable=outcome_var,
                success_threshold=threshold,
            )

        case AgentArm.NO_MEMORY:
            return NoMemoryAgent(
                llm=llm,
                tools=tools,
                step_budget=settings.step_budget,
                action_variable=action_var,
                outcome_variable=outcome_var,
                success_threshold=threshold,
            )


def load_tasks(path: Path, *, limit: int | None = None) -> list[DebuggingTask]:
    """Load a JSON Lines task file using the SWE-bench loader."""
    return load_swebench_tasks(path, limit=limit)
