"""Tests for the synthetic warm-start prior generator."""

from __future__ import annotations

from causa.domain.scm_debugging import build_debugging_scm
from causa.planning.warm_start import SyntheticWarmStart


def test_generate_zero_rows_returns_empty():
    scm = build_debugging_scm()
    rows = SyntheticWarmStart(seed=0).generate(scm, n=0)
    assert rows == []


def test_generated_rows_cover_all_scm_variables():
    scm = build_debugging_scm()
    rows = SyntheticWarmStart(seed=0).generate(scm, n=5)
    var_names = set(scm.variables)
    for row in rows:
        assert set(row) == var_names


def test_categorical_values_drawn_from_levels():
    scm = build_debugging_scm()
    rows = SyntheticWarmStart(seed=0).generate(scm, n=50)
    error_levels = set(scm.variables["error_message_type"].levels)
    for row in rows:
        assert row["error_message_type"] in error_levels


def test_continuous_values_in_unit_interval():
    scm = build_debugging_scm()
    rows = SyntheticWarmStart(seed=0).generate(scm, n=50)
    for row in rows:
        for var in ("information_gained", "patch_quality"):
            assert 0.0 <= row[var] <= 1.0


def test_seed_reproducibility():
    scm = build_debugging_scm()
    a = SyntheticWarmStart(seed=42).generate(scm, n=10)
    b = SyntheticWarmStart(seed=42).generate(scm, n=10)
    assert a == b
