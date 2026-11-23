import numpy as np
import pandas as pd
import pytest

from conftest import random_cov
from qpm.optimize import (Constraints, black_litterman, bl_posterior, check_feasibility, efficient_frontier, erc,
                          erc_weights, explain, factor_neutral, hrp, hrp_weights, inverse_vol, kelly,
                          linkage_matrix, liquidity_caps, max_sharpe, mean_variance, min_variance,
                          quasi_diagonal_order, risk_contributions, vol_target)

FREE = Constraints(lower=None, upper=None)   # budget only


@pytest.fixture
def inputs(rng):
    n = 8
    s = random_cov(rng, n)
    mu = rng.normal(0.06, 0.03, n)
    names = [f"a{i}" for i in range(n)]
    return pd.Series(mu, names), pd.DataFrame(s, names, names)


def test_min_variance_closed_form(inputs):
    mu, cov = inputs
    p = min_variance(cov, FREE)
    w = np.linalg.solve(cov.values, np.ones(len(mu)))
    assert np.allclose(p.weights.values, w / w.sum(), atol=1e-9)


def test_mean_variance_closed_form(inputs):
    mu, cov = inputs
    lam = 3.0
    p = mean_variance(mu, cov, lam, FREE)
    inv = np.linalg.inv(cov.values)
    ones = np.ones(len(mu))
    gamma = (ones @ inv @ mu.values - lam) / (ones @ inv @ ones)
    w = inv @ (mu.values - gamma) / lam
    assert np.allclose(p.weights.values, w, atol=1e-9)


def test_max_sharpe_is_tangency(inputs):
    mu, cov = inputs
    p = max_sharpe(mu, cov, FREE)
    w = np.linalg.solve(cov.values, mu.values)
    w /= w.sum()
    assert np.allclose(p.weights.values, w, atol=1e-7)


def test_max_sharpe_beats_every_frontier_point(inputs):
    mu, cov = inputs
    cons = Constraints(lower=0, upper=0.3)
    p = max_sharpe(mu, cov, cons)
    fr = efficient_frontier(mu, cov, cons, 40)
    assert p.sharpe >= fr["sharpe"].max() - 1e-8
    assert p.weights.min() >= -1e-12 and p.weights.max() <= 0.3 + 1e-12


def test_erc_equalises_risk(inputs):
    _, cov = inputs
    p = erc(cov)
    assert np.allclose(p.risk_contrib.values, 1 / len(cov), atol=1e-9)
    b = np.arange(1, len(cov) + 1, dtype=float)
    w = erc_weights(cov.values, b)
    rc = risk_contributions(w, cov.values)
    assert np.allclose(rc, b / b.sum(), atol=1e-9)


def test_inverse_vol_ignores_correlation(inputs):
    _, cov = inputs
    p = inverse_vol(cov)
    inv = 1 / np.sqrt(np.diag(cov.values))
    assert np.allclose(p.weights.values, inv / inv.sum())


def test_linkage_matches_scipy(rng):
    sp = pytest.importorskip("scipy.cluster.hierarchy")
    for method in ("single", "average", "complete"):
        for _ in range(10):
            c = random_cov(rng, 9)
            sd = np.sqrt(np.diag(c))
            d = np.sqrt(np.clip((1 - c / np.outer(sd, sd)) / 2, 0, None))
            dd = np.sqrt(((d[:, None, :] - d[None, :, :]) ** 2).sum(2))
            z_sp = sp.linkage(d, method)   # scipy treats the rows of d as observations, as the HRP paper does
            z_me = linkage_matrix(dd, method)
            assert np.allclose(z_sp, z_me, atol=1e-10)
            assert list(sp.leaves_list(z_sp)) == quasi_diagonal_order(z_me)


def test_hrp_block_structure():
    # two uncorrelated blocks: within-block correlation 0.9; block B has 4x the variance of block A
    n = 4
    corr = np.eye(n)
    corr[0, 1] = corr[1, 0] = 0.9
    corr[2, 3] = corr[3, 2] = 0.9
    sd = np.array([0.1, 0.1, 0.2, 0.2])
    cov = corr * np.outer(sd, sd)
    w, info = hrp_weights(cov)
    assert np.isclose(w.sum(), 1) and np.all(w > 0)
    assert np.isclose(w[0], w[1]) and np.isclose(w[2], w[3])
    var_a = 0.25 * (0.01 + 0.01 + 2 * 0.9 * 0.01)   # inverse-variance (equal) weights inside block A
    var_b = 4 * var_a
    assert np.isclose(w[:2].sum(), 1 - var_a / (var_a + var_b))


def test_black_litterman_limits(inputs):
    mu, cov = inputs
    w_mkt = np.full(len(mu), 1 / len(mu))
    prior = bl_posterior(cov.values, w_mkt)
    assert np.allclose(prior["mu"], prior["pi"])
    P = np.eye(len(mu))[[2]]
    exact = bl_posterior(cov.values, w_mkt, P, [0.10], confidences=[1.0])
    assert np.isclose(exact["posterior_view"][0], 0.10, atol=1e-10)
    half = bl_posterior(cov.values, w_mkt, P, [0.10], confidences=[0.5])
    lo, hi = sorted([prior["pi"][2], 0.10])
    assert lo < half["mu"][2] < hi
    p = black_litterman(cov, w_mkt, risk_aversion=2.5, cons=Constraints(lower=None, upper=None, budget=None))
    # no views, no constraints: BL returns the market portfolio scaled by 1/(1+tau)
    assert np.allclose(p.weights.values, w_mkt / 1.05, atol=1e-8)


def test_kelly_unconstrained_and_fractional(inputs):
    mu, cov = inputs
    loose = Constraints(lower=None, upper=None, budget=None)
    full = kelly(mu, cov, 1.0, loose)
    half = kelly(mu, cov, 0.5, loose)
    target = np.linalg.solve(cov.values, mu.values)
    assert np.allclose(full.weights.values, target, atol=1e-8)
    assert np.allclose(half.weights.values, 0.5 * target, atol=1e-8)


def test_vol_target_and_leverage_cap(inputs):
    mu, cov = inputs
    base = min_variance(cov, Constraints(lower=0))
    p = vol_target(base, cov, 0.05, max_leverage=10)
    assert np.isclose(p.volatility, 0.05)
    capped = vol_target(base, cov, 5.0, max_leverage=1.5)
    assert np.isclose(capped.gross, 1.5)


def test_factor_neutral_exposures(inputs, rng):
    mu, cov = inputs
    betas = rng.normal(1, 0.3, (len(mu), 2))
    p = factor_neutral(mu, cov, betas, ["mkt", "size"])
    assert p.ok
    assert np.allclose(betas.T @ p.weights.values, 0, atol=1e-9)
    assert abs(p.weights.sum()) < 1e-9 and p.gross <= 2 + 1e-9


def test_explanation_reproduces_weights(inputs):
    mu, cov = inputs
    names = list(mu.index)
    cons = Constraints(lower=0, upper=0.25, groups={"g": names[:4]}, group_max={"g": 0.4},
                       turnover_max=0.5, w_prev=np.full(len(names), 1 / len(names)))
    p = mean_variance(mu, cov, 4.0, cons)
    e = explain(p, cov, mu)
    t = e["table"]
    assert np.allclose(t["weight_from_formula"], t["weight"], atol=1e-8)
    assert np.allclose(4.0 * cov.values @ p.weights.values, p.mu_eff.values, atol=1e-8)
    assert len(e["sentences"]) == len(names)


def test_max_vol_cap(inputs):
    mu, cov = inputs
    p = mean_variance(mu, cov, 0.5, Constraints(lower=0, max_vol=0.12))
    assert p.ok and p.volatility <= 0.12 + 1e-9


def test_infeasible_constraints_reported(inputs):
    mu, cov = inputs
    cons = Constraints(lower=0, upper=0.1)   # 8 assets x 10% < 100%
    msgs = check_feasibility(cons, list(mu.index))
    assert any("caps add up" in m for m in msgs)
    p = mean_variance(mu, cov, 4.0, cons)
    assert not p.ok and p.status == "primal_infeasible"


def test_liquidity_caps_bound_positions(inputs):
    mu, cov = inputs
    adv = np.full(len(mu), 1e8)      # 10% of ADV for 5 days = 50% of a $100m book
    adv[0] = 1e5                     # ... but only 0.05% in the illiquid asset
    caps = liquidity_caps(adv, aum=1e8, participation=0.1, days=5)
    p = mean_variance(mu, cov, 2.0, Constraints(lower=0, upper=np.minimum(caps, 0.5)))
    assert p.ok and p.weights.iloc[0] <= caps[0] + 1e-12


def test_frontier_is_monotone(inputs):
    mu, cov = inputs
    fr = efficient_frontier(mu, cov, Constraints(lower=0), 15)
    assert (np.diff(fr["volatility"]) >= -1e-10).all()
    assert (np.diff(fr["expected_return"]) > 0).all()
