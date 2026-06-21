"""Centralised settings for Causa.

Twelve-factor: every knob is an environment variable.  The :class:`CausaSettings`
class is loaded once at process start, then passed by dependency injection.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    MOCK = "mock"


class EstimatorKind(str, Enum):
    LINEAR_REGRESSION = "linear_regression"
    PROPENSITY_SCORE = "propensity_score"
    DOUBLY_ROBUST = "doubly_robust"


class CausaSettings(BaseSettings):
    """Causa runtime settings — loaded from env vars and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CAUSA_",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ────────────────────────────────────────────────────────────────
    llm_provider: LLMProvider = Field(default=LLMProvider.MOCK)
    llm_model: str = Field(default="claude-3-5-sonnet-latest")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=2048, ge=64, le=64_000)

    # ── Planning layer ─────────────────────────────────────────────────────
    warm_start_prior_size: int = Field(default=20, ge=0, le=10_000)
    """Number of synthetic prior observations seeded by the warm-start generator."""
    dowhy_min_history: int = Field(default=10, ge=1)
    """Below this many real observations, the agent falls back to LLM scoring."""
    dowhy_estimator: EstimatorKind = Field(default=EstimatorKind.LINEAR_REGRESSION)

    # ── Reflection ─────────────────────────────────────────────────────────
    reflection_threshold: float = Field(default=0.15, ge=0.0, le=1.0)
    """θ in §D1 of the dissertation."""
    reflection_samples: int = Field(default=3, ge=1, le=10)
    """Per-alternative samples; median over samples reduces LLM noise (§D3)."""

    # ── Decision loop ──────────────────────────────────────────────────────
    step_budget: int = Field(default=12, ge=1)
    """Per-task tool-call budget; matches the SWE-bench protocol."""
    success_threshold: float = Field(default=0.9, ge=0.0, le=1.0)
    """Outcome score at or above which a task is considered solved."""

    # ── Telemetry ──────────────────────────────────────────────────────────
    trace_dir: Path = Field(default=Path("traces"))

    # ── Evaluation ─────────────────────────────────────────────────────────
    swe_bench_cache: Path = Field(default=Path("data/cache/swe-bench"))
    bootstrap_iterations: int = Field(default=1000, ge=100, le=100_000)
    random_seed: int = Field(default=42)


def load_settings(**overrides: object) -> CausaSettings:
    """Load settings from env + .env, optionally overriding selected fields."""
    base = CausaSettings()
    if not overrides:
        return base
    return base.model_copy(update=dict(overrides))
