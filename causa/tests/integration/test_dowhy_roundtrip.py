"""DoWhy round-trip integration test.

A minimal 3-node SCM with a *known* interventional effect, fitted on
synthetic data, must produce an estimate close to the truth.  This is the
end-to-end sanity check on the do-calculus pipeline: identification →
estimator → ActionScorer.

3-node SCM
==========
    Z (binary confounder)  →  X (binary treatment)
                           →  Y (continuous outcome)
    X                      →  Y

True structural equations:
    Z ~ Bernoulli(0.5)
    X = Z (deterministic mapping for clarity)
    Y = 0.5 * X + 0.3 * Z + ε,  ε ~ N(0, 0.05)

True interventional effect:
    E[Y | do(X=1)] - E[Y | do(X=0)] = 0.5

Adjusting for Z (the back-door variable) should recover ~0.5 from the
linear regression estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from causa.adapters.estimators.linear_regression import LinearRegressionEstimator
from causa.core.identifiability import identify_effect
from causa.core.scm import SCMBuilder, VariableDomain
from causa.planning.dowhy_scorer import DoWhyActionScorer
from causa.ports.scorer import ActionCandidate


def _synth_3node_data(n: int = 800, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    z = rng.integers(0, 2, size=n)
    # X is stochastically influenced by Z (not deterministic) to avoid
    # perfect multicollinearity in the back-door OLS adjustment.
    p_x = np.where(z == 1, 0.8, 0.2)
    x = (rng.random(size=n) < p_x).astype(int)
    y = 0.5 * x + 0.3 * z + rng.normal(0, 0.05, size=n)
    return pd.DataFrame({"Z": z.astype(str), "X": x.astype(str), "Y": y})


def _three_node_scm():
    b = SCMBuilder("3-node-scm")
    b.observational("Z", VariableDomain.BINARY, levels=("0", "1"))
    b.action("X", VariableDomain.BINARY, levels=("0", "1"))
    b.outcome("Y", VariableDomain.CONTINUOUS)
    b.edge("Z", "X")
    b.edge("Z", "Y")
    b.edge("X", "Y")
    return b.build()


def test_identification_picks_confounder_as_backdoor():
    scm = _three_node_scm()
    result = identify_effect(scm.graph, "X", "Y")
    assert result.identifiable
    assert result.adjustment_set == frozenset({"Z"})


def test_linear_regression_recovers_intervention_effect():
    """Within tolerance, adjusting for Z recovers the +0.5 treatment effect."""
    scm = _three_node_scm()
    data = _synth_3node_data(n=800, seed=0)

    estimator = LinearRegressionEstimator()
    e1 = estimator.estimate(
        data=data, treatment="X", outcome="Y",
        treatment_value="1", adjustment_set=frozenset({"Z"}),
    )
    e0 = estimator.estimate(
        data=data, treatment="X", outcome="Y",
        treatment_value="0", adjustment_set=frozenset({"Z"}),
    )
    effect = e1.value - e0.value
    # True effect is +0.5; back-door-adjusted estimator should be close.
    assert effect == pytest.approx(0.5, abs=0.05)


def test_dowhy_scorer_emits_backdoor_provenance():
    """The ActionScorer must surface the back-door adjustment set in its
    rationale so the trace is auditable."""
    scm = _three_node_scm()
    data = _synth_3node_data(n=400, seed=1)
    scorer = DoWhyActionScorer(scm=scm, estimator=LinearRegressionEstimator())
    candidates = [ActionCandidate(name="0"), ActionCandidate(name="1")]
    scores = scorer.score(
        candidates,
        state={"Z": "1"},
        history=data,
    )
    assert len(scores) == 2
    # X=1 should outperform X=0 on the canonical synthetic data.
    score_by_name = {s.action.name: s for s in scores}
    assert score_by_name["1"].score > score_by_name["0"].score
    # Provenance includes the back-door set.
    for s in scores:
        assert "back-door" in s.rationale
        assert s.adjustment_set == frozenset({"Z"})
