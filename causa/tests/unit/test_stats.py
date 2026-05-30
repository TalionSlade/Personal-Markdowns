"""Tests for the BCa bootstrap and paired permutation test."""

from __future__ import annotations

import math

from causa.evaluation.stats import (
    bca_bootstrap_mean,
    paired_permutation_test,
)


def test_paired_permutation_exact_for_small_n():
    a = [1.0, 1.0, 1.0, 1.0, 1.0]
    b = [0.0, 0.0, 0.0, 0.0, 0.0]
    result = paired_permutation_test(a, b)
    assert result.method == "exact"
    assert math.isclose(result.mean_diff, 1.0)
    # All 2^5 = 32 sign flips; the only sign-flip vector producing a
    # mean-difference as extreme as +1.0 is all-positive (and as extreme
    # as -1.0 is all-negative).  So p = 2/32.
    assert math.isclose(result.p_value, 2.0 / 32.0, rel_tol=1e-9)


def test_paired_permutation_zero_difference_unrejected():
    a = [0.3, 0.5, 0.7]
    result = paired_permutation_test(a, a)
    assert math.isclose(result.mean_diff, 0.0)
    assert result.p_value == 1.0


def test_paired_permutation_monte_carlo_above_threshold():
    a = [0.7] * 20
    b = [0.3] * 20
    result = paired_permutation_test(a, b, n_permutations=2000, seed=0)
    assert result.method == "monte_carlo"
    # 20-way matched separation → extreme p-value (≪ 0.001)
    assert result.p_value < 0.01


def test_bca_returns_valid_interval():
    sample = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 0.5, 0.7, 0.3, 0.4]
    ci = bca_bootstrap_mean(sample, confidence=0.95, n_bootstrap=2000, seed=0)
    assert ci.lower < ci.point < ci.upper
    assert ci.confidence == 0.95
    assert ci.method == "bca"


def test_bca_requires_at_least_two_observations():
    import pytest
    with pytest.raises(ValueError):
        bca_bootstrap_mean([0.5], n_bootstrap=1000)


def test_bca_interval_contains_true_mean_for_high_n():
    rng_sample = [i / 100 for i in range(100)]
    ci = bca_bootstrap_mean(rng_sample, confidence=0.95, n_bootstrap=4000, seed=0)
    true_mean = sum(rng_sample) / len(rng_sample)
    assert ci.lower <= true_mean <= ci.upper
