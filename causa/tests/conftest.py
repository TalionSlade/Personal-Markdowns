"""Shared pytest fixtures for Causa.

Fixtures here are the *only* place test files build adapters or SCMs.
Keeps tests focused on what they assert rather than how they wire.
"""

from __future__ import annotations

import pandas as pd
import pytest

from causa.adapters.history.pandas_history import PandasObservationHistory
from causa.adapters.llm.mock_client import MockLLMClient
from causa.adapters.tools.registry import make_tool_registry
from causa.core.scm import SCM
from causa.domain.scm_debugging import (
    build_debugging_scm,
    debugging_observation_schema,
)
from causa.planning.warm_start import SyntheticWarmStart
from causa.reflection.threshold import StaticThreshold


@pytest.fixture
def debugging_scm() -> SCM:
    """The canonical 9-node debugging SCM."""
    return build_debugging_scm()


@pytest.fixture
def mock_llm() -> MockLLMClient:
    """A deterministic LLM client — no API cost, no network."""
    return MockLLMClient()


@pytest.fixture
def empty_history() -> PandasObservationHistory:
    """An empty pandas-backed history with the debugging SCM's schema."""
    return PandasObservationHistory(schema=debugging_observation_schema())


@pytest.fixture
def warm_started_history(debugging_scm: SCM) -> PandasObservationHistory:
    """A history pre-populated with 40 synthetic prior rows.

    40 chosen empirically: enough that the back-door estimator has signal
    on every action and outcome level, small enough that test runtime
    stays under a few hundred ms per case.
    """
    h = PandasObservationHistory(schema=debugging_observation_schema())
    rows = SyntheticWarmStart(seed=7).generate(debugging_scm, n=40)
    h.append_many(rows)
    return h


@pytest.fixture
def tool_registry():
    return make_tool_registry()


@pytest.fixture
def static_threshold() -> StaticThreshold:
    return StaticThreshold(value=0.05)
