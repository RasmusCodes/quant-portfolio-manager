import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from qpm.risk import (Scenario, component_es, concentration, conditional_drawdown_at_risk, contributions,
                      cornish_fisher_var, correlation_breakdown, drawdown_episodes, factor_decomposition,
                      historical_es, historical_var, kupiec_pof, liquidity_profile, load_scenarios,
                      monte_carlo_var, parametric_es, parametric_var, rolling_var_backtest, run_scenario,
                      stressed_cov)

ROOT = Path(__file__).resolve().parents[1]


def test_var_es_basics(rng):
    r = rng.normal(0.0005, 0.01, 200_000)
    v, es = historical_var(r, 0.99), historical_es(r, 0.99)
    assert abs(v - parametric_var(0.0005, 0.01, 0.99)) < 2e-4
    assert abs(es - parametric_es(0.0005, 0.01, 0.99)) < 3e-4
    assert es > v
    assert abs(cornish_fisher_var(r, 0.99) - v) < 3e-4


def test_t_es_exceeds_normal_es():
    assert parametric_es(0, 0.01, 0.99, "t", 4) > parametric_es(0, 0.01, 0.99)
    assert parametric_var(0, 0.01, 0.99, "t", 4) > parametric_var(0, 0.01, 0.95, "t", 4)


def test_monte_carlo_matches_parametric():
    cov = np.array([[0.0004, 0.0001], [0.0001, 0.0009]])
    w = np.array([0.6, 0.4])
    v, es = monte_carlo_var(w, np.zeros(2), cov, 0.99, 400_000, dist="normal", seed=1)
    sig = np.sqrt(w @ cov @ w)
    assert abs(v - parametric_var(0, sig, 0.99)) / v < 0.02


def test_kupiec_and_rolling_backtest(rng):
    k = kupiec_pof(10, 1000, 0.99)
    assert k["p_value"] > 0.99
    assert kupiec_pof(40, 1000, 0.99)["p_value"] < 0.001
    r = rng.standard_t(4, 3000) * 0.01
    bt = rolling_var_backtest(r, 500, 0.99, "normal")
    assert bt["kupiec"]["exceptions"] > bt["kupiec"]["expected"]   # normal VaR understates fat tails


def test_contributions_sum_to_vol(rng):
    from conftest import random_cov
    cov = random_cov(rng, 6)
    w = pd.Series(rng.dirichlet(np.ones(6)), [f"a{i}" for i in range(6)])
    c = contributions(w, cov)
    assert np.isclose(c["component"].sum(), c.attrs["volatility"])
    x = rng.multivariate_normal(np.zeros(6), cov / 12, 5000)
    ces = component_es(w, x, 0.975)
    assert np.isclose(ces.sum(), historical_es(x @ w.values, 0.975), rtol=1e-6)


def test_factor_decomposition_shares(small_market):
    returns, _ = small_market
    f = pd.DataFrame({"f1": returns.mean(1), "f2": returns.iloc[:, 0] - returns.iloc[:, 1]})
    w = pd.Series(1 / returns.shape[1], returns.columns)
    d = factor_decomposition(w, returns, f)
    assert np.isclose(d["factor_contrib"].sum() + d["idiosyncratic_share"], 1.0)


def test_concentration_measures():
    c = concentration(np.full(10, 0.1), np.eye(10) * 0.04)
    assert np.isclose(c["effective_n"], 10) and np.isclose(c["effective_bets"], 10)
    one = concentration(np.array([1.0, 0, 0]), np.eye(3) * 0.04)
    assert np.isclose(one["diversification_ratio"], 1.0)


def test_scenarios_folder_loads_and_runs(small_market):
    returns, _ = small_market
    scs = load_scenarios(ROOT / "scenarios")
    assert len(scs) >= 8
    kinds = {s.kind for s in scs}
    assert {"historical", "factor", "volatility"} <= kinds
    for path in (ROOT / "scenarios").glob("*.json"):
        json.loads(path.read_text(encoding="utf-8"))


def test_historical_factor_and_vol_scenarios(small_market):
    returns, _ = small_market
    w = pd.Series(1 / returns.shape[1], returns.columns)
    hist = Scenario("h", "historical", start="2005-01-01", end="2005-06-30")
    out = run_scenario(w, hist, returns=returns)
    expected = ((1 + returns.loc["2005-01":"2005-06"]).prod() - 1) @ w
    assert np.isclose(out["pnl"], expected)
    betas = pd.DataFrame({"Mkt-RF": np.linspace(0.5, 1.5, len(w))}, index=w.index)
    fac = Scenario("f", "factor", factor_shocks={"Mkt-RF": -0.2}, asset_shocks={"A0": -0.5})
    out = run_scenario(w, fac, betas=betas)
    moves = -0.2 * betas["Mkt-RF"]
    moves["A0"] = -0.5
    assert np.isclose(out["pnl"], (w * moves).sum())
    cov = returns.cov() * 12
    vol = Scenario("v", "volatility", vol_multiplier=2.0, correlation_target=0.9)
    out = run_scenario(w, vol, cov=cov)
    assert out["stressed_vol"] > 2.0 * out["base_vol"] * 0.99


def test_stressed_cov_psd(rng):
    from conftest import random_cov
    s = stressed_cov(random_cov(rng, 7), 1.5, 0.95, 1.0)
    assert np.linalg.eigvalsh(s).min() > 0


def test_drawdown_tools():
    r = pd.Series([0.1, -0.2, -0.1, 0.05, 0.3, -0.05], index=pd.date_range("2020-01-31", periods=6, freq="ME"))
    ep = drawdown_episodes(r)
    assert np.isclose(ep.loc[0, "depth"], 1.1 * 0.8 * 0.9 / 1.1 - 1)
    assert conditional_drawdown_at_risk(r, 0.5) > 0


def test_correlation_breakdown(small_market):
    returns, _ = small_market
    out = correlation_breakdown(returns, returns.mean(1))
    assert -1 <= out["calm_avg_corr"] <= 1 and 0 < out["absorption_ratio"] <= 1


def test_liquidity_profile():
    w = pd.Series([0.5, 0.5], ["a", "b"])
    adv = pd.Series([1e6, 1e9], ["a", "b"])
    prof = liquidity_profile(w, 1e8, adv, 0.1, (1, 5))
    assert prof["1d"] < 1 and prof["max_days"] > 400
