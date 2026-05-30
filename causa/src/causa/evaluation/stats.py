"""Statistical tests for paired evaluation outcomes.

§F3 of the dissertation: every headline comparison between Causa and a
baseline must report (a) a paired non-parametric test with an exact
p-value, and (b) a bias-corrected and accelerated (BCa) bootstrap
confidence interval over the mean difference.

Why these specific tests:

- **Paired permutation**: matched-task comparisons; exact under the null
  of zero per-task difference, makes no normality assumption, and behaves
  sensibly at the small task counts the eval splits produce.
- **BCa bootstrap**: handles skew in the per-task outcome distribution
  (success rate over small subsets is skewed near 0 or 1) better than the
  percentile method.  Two correction factors:
  ``z₀`` for bias and ``a`` for skewness.

The implementations use NumPy.  They make no assumption about outcome
distribution shape and are deterministic given the supplied seed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt
from typing import Sequence

import numpy as np


# ─── permutation test ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class PairedTestResult:
    """Result of a paired permutation test on per-task outcome differences.

    Attributes
    ----------
    mean_diff:
        Mean of ``a - b`` across paired observations.
    p_value:
        Two-sided p-value under the null of zero mean difference,
        estimated by symmetric sign-flip permutation.
    n_permutations:
        Permutations sampled (may equal ``2**n`` when exact).
    n_pairs:
        Sample size (length of either input).
    method:
        Either ``"exact"`` (small n) or ``"monte_carlo"``.
    """

    mean_diff: float
    p_value: float
    n_permutations: int
    n_pairs: int
    method: str


def paired_permutation_test(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_permutations: int = 10_000,
    exact_threshold: int = 14,
    seed: int = 42,
) -> PairedTestResult:
    """Run a paired permutation (sign-flip) test on ``a - b``.

    For ``n ≤ exact_threshold`` the test enumerates all ``2ⁿ`` sign
    flips exactly.  Above the threshold it samples ``n_permutations``
    random sign-flip vectors (Monte Carlo); reproducible given ``seed``.
    """
    if len(a) != len(b):
        raise ValueError(f"length mismatch: |a|={len(a)}, |b|={len(b)}")
    if len(a) == 0:
        raise ValueError("cannot test on empty samples")

    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    observed_mean = float(diffs.mean())
    n = len(diffs)

    if n <= exact_threshold:
        # Enumerate the 2^n sign-flip vectors as the rows of a {-1,+1} matrix.
        # n ≤ 14 → ≤ 16384 rows of length n; cheap.
        bits = np.arange(2 ** n, dtype=np.int64)
        signs = (((bits[:, None] >> np.arange(n)) & 1) * 2 - 1).astype(float)
        means = (signs * diffs).mean(axis=1)
        as_extreme = np.abs(means) >= abs(observed_mean) - 1e-12
        p_value = float(as_extreme.mean())
        return PairedTestResult(
            mean_diff=observed_mean,
            p_value=p_value,
            n_permutations=2 ** n,
            n_pairs=n,
            method="exact",
        )

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_permutations, n))
    means = (signs * diffs).mean(axis=1)
    # +1 in numerator/denominator: standard +1 correction so the p-value is
    # never zero (Phipson & Smyth 2010).
    count_extreme = int(np.sum(np.abs(means) >= abs(observed_mean) - 1e-12))
    p_value = (count_extreme + 1) / (n_permutations + 1)
    return PairedTestResult(
        mean_diff=observed_mean,
        p_value=p_value,
        n_permutations=n_permutations,
        n_pairs=n,
        method="monte_carlo",
    )


# ─── BCa bootstrap ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BootstrapInterval:
    """A bias-corrected and accelerated (BCa) bootstrap CI.

    Attributes
    ----------
    point:
        The plug-in estimate (sample mean).
    lower:
        Lower bound of the CI.
    upper:
        Upper bound of the CI.
    confidence:
        Nominal coverage (e.g. 0.95).
    method:
        Always ``"bca"`` here; the field is kept so other bootstrap
        variants (percentile, basic) can coexist later.
    n_bootstrap:
        Number of resamples drawn.
    """

    point: float
    lower: float
    upper: float
    confidence: float
    method: str
    n_bootstrap: int


def bca_bootstrap_mean(
    sample: Sequence[float],
    *,
    confidence: float = 0.95,
    n_bootstrap: int = 5000,
    seed: int = 42,
) -> BootstrapInterval:
    """BCa CI for the mean of ``sample``.

    Uses the standard BCa formulation:
      1. Draw ``n_bootstrap`` resamples-with-replacement, take their means.
      2. Estimate bias correction ``z₀`` from the proportion of resamples
         below the plug-in mean.
      3. Estimate acceleration ``a`` from jackknife deviates.
      4. Adjust the percentile endpoints accordingly.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if n_bootstrap < 100:
        raise ValueError(f"n_bootstrap must be ≥ 100, got {n_bootstrap}")
    if len(sample) < 2:
        raise ValueError(f"need ≥ 2 observations, got {len(sample)}")

    arr = np.asarray(sample, dtype=float)
    n = arr.size
    point = float(arr.mean())
    rng = np.random.default_rng(seed)

    boot_means = arr[rng.integers(0, n, size=(n_bootstrap, n))].mean(axis=1)

    # Bias correction z₀
    proportion_below = float(np.mean(boot_means < point))
    proportion_below = min(max(proportion_below, 1e-6), 1.0 - 1e-6)
    z0 = _ndtri(proportion_below)

    # Acceleration via jackknife
    jackknife = np.array([np.delete(arr, i).mean() for i in range(n)])
    jack_mean = jackknife.mean()
    num = float(np.sum((jack_mean - jackknife) ** 3))
    den = 6.0 * (float(np.sum((jack_mean - jackknife) ** 2)) ** 1.5)
    accel = num / den if den > 0 else 0.0

    alpha = 1.0 - confidence
    z_low = _ndtri(alpha / 2.0)
    z_high = _ndtri(1.0 - alpha / 2.0)
    alpha_low = _ndtr(z0 + (z0 + z_low) / (1.0 - accel * (z0 + z_low)))
    alpha_high = _ndtr(z0 + (z0 + z_high) / (1.0 - accel * (z0 + z_high)))
    lower = float(np.quantile(boot_means, alpha_low))
    upper = float(np.quantile(boot_means, alpha_high))

    return BootstrapInterval(
        point=point,
        lower=lower,
        upper=upper,
        confidence=confidence,
        method="bca",
        n_bootstrap=n_bootstrap,
    )


# ─── normal CDF / inverse helpers ────────────────────────────────────────────
# We keep these here so the module has no scipy dependency.  Accuracy is
# more than sufficient for confidence-interval endpoints.


def _ndtr(x: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _ndtri(p: float) -> float:
    """Inverse standard normal CDF — Beasley-Springer-Moro approximation.

    Accurate to ~10⁻⁹ over (0, 1) — comfortably tight for BCa endpoints.
    """
    if not 0.0 < p < 1.0:
        raise ValueError(f"p must be in (0, 1), got {p}")

    # Coefficients (Wichura 1988 / Acklam 2003)
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]

    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = sqrt(-2.0 * _ln(p))
        return (
            ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
        ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
        ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    q = sqrt(-2.0 * _ln(1.0 - p))
    return -(
        ((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]
    ) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)


def _ln(x: float) -> float:
    # Tiny shim so the float-only API doesn't import math twice.
    import math

    return math.log(x)
