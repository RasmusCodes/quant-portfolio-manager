import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpm.backtest import (BacktestConfig, CostModel, EqualWeight, FixedWeights, OptimizerStrategy, run_backtest,
                          sharpe_diff_bootstrap, sharpe_diff_jkm, deflated_sharpe, performance_stats)
from qpm.execution import CostModelExecution, SmartOrderRouterAdapter
from qpm.optimize import Constraints

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("strategy", [OptimizerStrategy("min_variance", window=60, constraints=Constraints(lower=0, upper=0.5)),
                                      OptimizerStrategy("mean_variance", window=60, mean_method="bayes_stein",
                                                        constraints=Constraints(lower=0, upper=0.5)),
                                      OptimizerStrategy("hrp", window=60)])
def test_no_look_ahead_future_changes_do_not_alter_past(small_market, strategy):
    returns, rf = small_market
    cfg = BacktestConfig(window=60, costs=CostModel(bps=10))
    base = run_backtest(strategy, returns, rf, cfg)
    cut = returns.index[150]
    shocked = returns.copy()
    shocked.loc[shocked.index > cut] *= -3.0          # rewrite the future
    alt = run_backtest(strategy, shocked, rf, cfg)
    w0 = base.weights.loc[:cut]
    w1 = alt.weights.loc[:cut]
    assert np.allclose(w0.values, w1.values, atol=1e-12)
    assert np.allclose(base.returns.loc[:cut].values, alt.returns.loc[:cut].values, atol=1e-14)


def test_equal_weight_matches_hand_computation(small_market):
    returns, rf = small_market
    cfg = BacktestConfig(window=12, costs=CostModel(bps=0), borrow_bps=0)
    res = run_backtest(EqualWeight(), returns, rf, cfg)
    # rebalanced monthly to 1/N with no costs: each month's return is the cross-sectional mean
    expected = returns.iloc[13:].mean(axis=1)
    assert np.allclose(res.returns.values, expected.values, atol=1e-14)


def test_costs_equal_turnover_times_bps(small_market):
    returns, rf = small_market
    cfg = BacktestConfig(window=24, costs=CostModel(bps=25))
    res = run_backtest(OptimizerStrategy("inverse_vol", window=24), returns, rf, cfg)
    assert np.allclose(res.costs.values, res.turnover.values * 25e-4)
    net = (1 + res.returns).prod()
    gross = (1 + res.gross_returns).prod()
    assert net < gross


def test_all_cash_earns_the_bill_rate(small_market):
    returns, rf = small_market
    res = run_backtest(FixedWeights({}), returns, rf, BacktestConfig(window=12))
    assert np.allclose(res.returns.values, rf.iloc[13:].values)


def test_executor_path_equals_vectorised_costs(small_market):
    returns, rf = small_market
    adv = pd.DataFrame(5e8, index=returns.index, columns=returns.columns)
    costs = CostModel(bps=8, impact_eta=0.5, aum=1e8, adv=adv)
    strat = OptimizerStrategy("erc", window=36)
    a = run_backtest(strat, returns, rf, BacktestConfig(window=36, costs=costs))
    ex = CostModelExecution(commission_bps=0.0, impact_eta=0.5)
    b = run_backtest(strat, returns, rf, BacktestConfig(window=36, costs=costs, executor=ex))
    assert np.allclose(a.costs.values, b.costs.values, rtol=1e-9, atol=1e-15)


def test_router_cli_integration(small_market):
    returns, rf = small_market
    router = SmartOrderRouterAdapter("cli", [sys.executable, str(ROOT / "examples" / "stub_router.py")])
    adv = pd.DataFrame(5e8, index=returns.index, columns=returns.columns)
    cfg = BacktestConfig(window=36, costs=CostModel(bps=5, aum=1e8, adv=adv), executor=router)
    res = run_backtest(OptimizerStrategy("inverse_vol", window=36), returns.iloc[:60], rf.iloc[:60], cfg)
    assert (res.costs > 0).all() and res.log == []


def test_vol_target_overlay_respects_leverage(small_market):
    returns, rf = small_market
    cfg = BacktestConfig(window=36, vol_target=0.50, max_leverage=1.2)
    res = run_backtest(EqualWeight(), returns, rf, cfg)
    assert (res.exposure["gross"] <= 1.2 + 1e-12).all()


def test_drawdown_governor_cuts_exposure(small_market):
    returns, rf = small_market
    crash = returns.copy()
    crash.iloc[60:70] = -0.08
    res = run_backtest(EqualWeight(), crash, rf, BacktestConfig(window=12, drawdown_limit=0.2))
    assert res.weights.sum(axis=1).min() < 0.5


def test_sharpe_tests_agree_on_identical_series(rng):
    a = pd.Series(rng.normal(0.01, 0.04, 300))
    t = sharpe_diff_jkm(a, a + rng.normal(0, 1e-6, 300))
    assert t["p_value"] > 0.5
    b = pd.Series(rng.normal(0.0, 0.04, 300))
    boot = sharpe_diff_bootstrap(a, b, n_boot=300)
    assert boot["ci_low"] < boot["diff"] < boot["ci_high"]


def test_deflated_sharpe_penalises_many_trials():
    one = deflated_sharpe(0.8, 1, 300)
    many = deflated_sharpe(0.8, 100, 300, sr_trials_std_annual=0.4)
    assert many["dsr"] < one["dsr"]


def test_performance_stats_values():
    r = pd.Series([0.01] * 12)
    s = performance_stats(r, pd.Series([0.0] * 12))
    assert np.isclose(s["ann_return"], 1.01 ** 12 - 1)
    assert s["max_drawdown"] == 0
