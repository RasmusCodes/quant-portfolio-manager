"""Run the pre-registered experiments (reports/PREREGISTRATION.md) and write results/<id>/.

    python scripts/run_experiments.py                 # everything (about 30-40 minutes)
    python scripts/run_experiments.py --only E1 E4    # a subset

Every experiment is deterministic (fixed seeds) and reads only data/processed/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpm.backtest import (BacktestConfig, CostModel, EqualWeight, OptimizerStrategy, SignalLongShort,  # noqa: E402
                          run_backtest, sharpe_diff_bootstrap, sharpe_diff_jkm)
from qpm.data import load_universe  # noqa: E402
from qpm.estimation import ledoit_wolf_constant_correlation  # noqa: E402
from qpm.execution import SmartOrderRouterAdapter  # noqa: E402
from qpm.optimize import (Constraints, erc_weights, max_sharpe, mean_variance, min_variance)  # noqa: E402
from qpm.risk import (contributions, concentration, conditional_drawdown_at_risk, correlation_breakdown,  # noqa: E402
                      cornish_fisher_var, factor_decomposition, historical_es, historical_var, liquidity_profile,
                      load_scenarios, monte_carlo_var, parametric_es, parametric_var, rolling_var_backtest,
                      run_scenario, scenario_betas, drawdown_episodes)
from qpm.signals import combine, cross_sectional_ic, forward_returns, newey_west_t, panel_zscore, reference  # noqa: E402
from qpm.signals import synthetic  # noqa: E402

RESULTS = ROOT / "results"
BASE = {"default": 10.0, "UST10": 2.0, "UST5": 2.0}
HIGH = {"default": 30.0, "UST10": 5.0, "UST5": 5.0}
C25 = Constraints(lower=0.0, upper=0.25)
C10 = Constraints(lower=0.0, upper=0.10)
C40 = Constraints(lower=0.0, upper=0.40)
TRADE_COST = {"default": 0.001, "UST10": 0.0002, "UST5": 0.0002}


def out_dir(exp: str) -> Path:
    d = RESULTS / exp
    d.mkdir(parents=True, exist_ok=True)
    return d


def cfg(costs=BASE, **kw) -> BacktestConfig:
    kw.setdefault("financing_spread_bps", 50.0)
    return BacktestConfig(window=kw.pop("window", 120), costs=CostModel(bps=costs), **kw)


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        adj = min(1.0, (m - i) * p)
        running = max(running, adj)
        out[k] = running
    return out


def nw_mean_t(x: pd.Series) -> float:
    return newey_west_t(x.dropna())


class SixtyForty:
    """60% cap-weighted industries, 20% each Treasury, rebalanced monthly."""
    name = "60/40"

    def target(self, ctx):
        return pd.Series(sixty_forty_weights(ctx, list(ctx.returns.columns)), index=ctx.returns.columns)


def sixty_forty_weights(ctx, assets):
    bonds = [a for a in assets if a.startswith("UST")]
    eq = [a for a in assets if a not in bonds]
    caps = ctx.universe.mcap.iloc[-1].reindex(eq)
    caps = caps.fillna(caps.median())
    w = pd.Series(0.0, index=assets)
    w[eq] = 0.6 * caps / caps.sum()
    for b in bonds:
        w[b] = 0.4 / len(bonds)
    return w.values


def run_set(universe, strategies: dict, signals=None, base_cfg=None) -> dict:
    out = {}
    for name, spec in strategies.items():
        strat, conf = spec if isinstance(spec, tuple) else (spec, base_cfg or cfg())
        conf.name = name
        t0 = time.time()
        res = run_backtest(strat, universe.returns, universe.rf, conf, universe=universe, signals=signals or {})
        out[name] = res
        s = res.stats()
        print(f"  {name:28s} {time.time() - t0:6.1f}s  SR {s['sharpe']:5.2f}  ret {s['ann_return']:6.2%}  vol "
              f"{s['volatility']:6.2%}  mdd {s['max_drawdown']:6.1%}  TO {s['turnover']:5.2f}  errors {len(res.log)}",
              flush=True)
    return out


def save_set(exp: str, tag: str, results: dict) -> pd.DataFrame:
    d = out_dir(exp)
    pd.DataFrame({k: r.returns for k, r in results.items()}).to_csv(d / f"returns_{tag}.csv")
    pd.DataFrame({k: r.gross_returns for k, r in results.items()}).to_csv(d / f"gross_{tag}.csv")
    pd.DataFrame({k: r.turnover for k, r in results.items()}).to_csv(d / f"turnover_{tag}.csv")
    weights = {k: {"dates": [str(x.date()) for x in r.weights.index],
                   "assets": list(r.weights.columns), "w": np.round(r.weights.values, 5).tolist()}
               for k, r in results.items()}
    (d / f"weights_{tag}.json").write_text(json.dumps(weights), encoding="utf-8")
    stats = pd.DataFrame({k: r.stats() for k, r in results.items()}).T
    stats.to_csv(d / f"stats_{tag}.csv")
    errors = {k: [(str(a.date()), b) for a, b in r.log] for k, r in results.items() if r.log}
    if errors:
        (d / f"errors_{tag}.json").write_text(json.dumps(errors, indent=1), encoding="utf-8")
    return stats


def compare(results: dict, bench: str, n_boot: int = 5000, family=None) -> pd.DataFrame:
    rows = {}
    for name, res in results.items():
        if name == bench or (family is not None and name not in family):
            continue
        a, b = res.excess, results[bench].excess
        j = sharpe_diff_jkm(a, b)
        bt = sharpe_diff_bootstrap(a, b, n_boot=n_boot, seed=7)
        rows[name] = {"sr": j["sr_a"], "sr_bench": j["sr_b"], "diff": j["diff"], "p_jkm": j["p_value"],
                      "p_boot": bt["p_value"], "ci_low": bt["ci_low"], "ci_high": bt["ci_high"], "rho": j["rho"]}
    t = pd.DataFrame(rows).T
    if len(t):
        t["p_boot_holm"] = pd.Series(holm(t["p_boot"].to_dict()))
        t["p_jkm_holm"] = pd.Series(holm(t["p_jkm"].to_dict()))
    return t


# ------------------------------------------------------------------------------------------------ E1

def e1():
    print("E1 horse race (multi_asset)")
    u = load_universe("multi_asset")
    mom = panel_zscore(reference.momentum_12_1(u.returns))
    kelly_cons = Constraints(lower=0.0, upper=0.25, budget=None, net_min=0.0, net_max=1.5)

    def strategies(costs):
        return {
            "1/N": (EqualWeight(), cfg(costs)),
            "60/40": (SixtyForty(), cfg(costs)),
            "Inverse vol": (OptimizerStrategy("inverse_vol"), cfg(costs)),
            "Min variance": (OptimizerStrategy("min_variance", constraints=C25), cfg(costs)),
            "ERC": (OptimizerStrategy("erc"), cfg(costs)),
            "HRP": (OptimizerStrategy("hrp"), cfg(costs)),
            "MV sample": (OptimizerStrategy("mean_variance", constraints=C25), cfg(costs)),
            "MV Bayes-Stein": (OptimizerStrategy("mean_variance", mean_method="bayes_stein", constraints=C25), cfg(costs)),
            "Max Sharpe sample": (OptimizerStrategy("max_sharpe", constraints=C25), cfg(costs)),
            "Max Sharpe Bayes-Stein": (OptimizerStrategy("max_sharpe", mean_method="bayes_stein", constraints=C25), cfg(costs)),
            "Black-Litterman": (OptimizerStrategy("black_litterman", mean_method="zero", signal="momentum", constraints=C25,
                                                  risk_aversion=2.5, market_weights=sixty_forty_weights), cfg(costs)),
            "Half Kelly": (OptimizerStrategy("kelly_half", constraints=kelly_cons), cfg(costs)),
            "Cost-aware MV": (OptimizerStrategy("mean_variance", mean_method="bayes_stein", constraints=C25, cost_aware=True,
                                                trade_cost=TRADE_COST), cfg(costs)),
            "1/N vol target 10%": (EqualWeight(), cfg(costs, vol_target=0.10, max_leverage=1.0)),
        }

    res_base = run_set(u, strategies(BASE), {"momentum": mom})
    save_set("E1", "base", res_base)
    print("  high costs")
    res_high = run_set(u, strategies(HIGH), {"momentum": mom})
    save_set("E1", "high", res_high)
    d = out_dir("E1")
    family = [k for k in res_base if k != "1/N"]
    compare(res_base, "1/N", family=family).to_csv(d / "tests_base.csv")
    compare(res_high, "1/N", family=family).to_csv(d / "tests_high.csv")
    h3 = {}
    for a, b in (("MV Bayes-Stein", "MV sample"), ("Max Sharpe Bayes-Stein", "Max Sharpe sample")):
        j = sharpe_diff_jkm(res_base[a].excess, res_base[b].excess)
        bt = sharpe_diff_bootstrap(res_base[a].excess, res_base[b].excess, n_boot=5000, seed=7)
        h3[f"{a} vs {b}"] = {"diff": j["diff"], "p_jkm": j["p_value"], "p_boot": bt["p_value"],
                             "ci_low": bt["ci_low"], "ci_high": bt["ci_high"]}
    pd.DataFrame(h3).T.to_csv(d / "tests_h3.csv")
    # drawdown episodes and sub-period Sharpe ratios for the report
    eps = {k: drawdown_episodes(r.returns, 3).assign(strategy=k) for k, r in res_base.items()}
    pd.concat(eps.values()).to_csv(d / "drawdowns_base.csv", index=False)
    rets = pd.DataFrame({k: r.excess for k, r in res_base.items()})
    decades = rets.groupby((rets.index.year // 10) * 10).apply(lambda x: x.mean() / x.std() * np.sqrt(12))
    decades.to_csv(d / "sharpe_by_decade.csv")


# ------------------------------------------------------------------------------------------------ E2

def e2(draws: int = 300):
    print("E2 estimation error Monte Carlo")
    u = load_universe("multi_asset")
    ex = u.returns.sub(u.rf, axis=0)
    mu_t, cov_t = ex.mean().values, ex.cov().values      # "truth" (monthly)
    names = list(ex.columns)
    n = len(names)
    rng = np.random.default_rng(2026)
    chol = np.linalg.cholesky(cov_t)
    true_sr = lambda w: float(np.sqrt(12) * (w @ mu_t) / np.sqrt(w @ cov_t @ w))  # noqa: E731
    true_growth = lambda w: float(12 * (w @ mu_t + u.rf.mean()) - 6 * (w @ cov_t @ w))  # noqa: E731

    def build(method, mu, cov):
        mu_a, cov_a = mu * 12, cov * 12
        if method == "max_sharpe":
            p = max_sharpe(mu_a, cov_a, C25, names)
        elif method == "mean_variance":
            p = mean_variance(mu_a, cov_a, 4.0, C25, names)
        elif method == "min_variance":
            p = min_variance(cov_a, C25, names)
        elif method == "erc":
            return erc_weights(cov_a)
        elif method == "kelly_full":
            p = mean_variance(mu_a, cov_a, 1.0, Constraints(lower=0, upper=None, budget=None, net_min=0, net_max=2.0), names)
        elif method == "equal":
            return np.full(n, 1 / n)
        else:
            raise KeyError(method)
        return p.weights.values if p.ok else np.full(n, np.nan)

    methods = ["max_sharpe", "mean_variance", "min_variance", "erc", "equal", "kelly_full"]
    rows = []
    for T in (60, 120, 240):
        for d in range(draws):
            x = mu_t + rng.standard_normal((T, n)) @ chol.T
            mu_e = x.mean(axis=0)
            cov_e = ledoit_wolf_constant_correlation(x)[0]
            for case, (m_in, c_in) in {"estimated": (mu_e, cov_e), "true_mean": (mu_t, cov_e),
                                       "true_cov": (mu_e, cov_t), "truth": (mu_t, cov_t)}.items():
                if case == "truth" and d > 0:
                    continue
                for m in methods:
                    if m in ("min_variance", "erc", "equal") and case in ("true_mean",):
                        continue
                    w = build(m, m_in, c_in)
                    if not np.all(np.isfinite(w)):
                        continue
                    est_sr = float(np.sqrt(12) * (w @ m_in) / np.sqrt(w @ c_in @ w))
                    rows.append({"T": T, "draw": d, "case": case, "method": m, "true_sr": true_sr(w),
                                 "estimated_sr": est_sr, "true_growth": true_growth(w), "gross": float(np.abs(w).sum()),
                                 "max_w": float(w.max()), "hhi": float(np.sum(w ** 2) / np.sum(np.abs(w)) ** 2)})
        print(f"  T={T} done", flush=True)
    df = pd.DataFrame(rows)
    d = out_dir("E2")
    df.to_csv(d / "draws.csv", index=False)
    summ = df.groupby(["T", "method", "case"]).agg(true_sr=("true_sr", "mean"), true_sr_sd=("true_sr", "std"),
                                                   estimated_sr=("estimated_sr", "mean"),
                                                   true_growth=("true_growth", "mean"), max_w=("max_w", "mean"),
                                                   n=("true_sr", "size")).reset_index()
    summ.to_csv(d / "summary.csv", index=False)


# ------------------------------------------------------------------------------------------------ E3

def e3():
    print("E3 sensitivity: covariance estimators (minimum variance)")
    rows, rets = [], {}
    for uname, cons in (("multi_asset", C25), ("industries49", C10)):
        u = load_universe(uname)
        for window in (60, 120):
            res = {}
            for est in ("sample", "lw_identity", "lw_cc", "ewma", "factor", "pca"):
                strat = OptimizerStrategy("min_variance", constraints=cons, cov_method=est, window=window)
                conf = cfg(BASE, window=120)   # common first decision date for both windows
                conf.name = est
                r = run_backtest(strat, u.returns, u.rf, conf, universe=u)
                res[est] = r
                rets[f"{uname}|{window}|{est}"] = r.returns
                s = r.stats()
                rows.append({"universe": uname, "window": window, "estimator": est, "volatility": s["volatility"],
                             "sharpe": s["sharpe"], "turnover": s["turnover"], "max_drawdown": s["max_drawdown"],
                             "errors": len(r.log)})
                print(f"  {uname:13s} w{window:3d} {est:12s} vol {s['volatility']:.2%} TO {s['turnover']:.2f}", flush=True)
            # H6: lower realised variance than the sample covariance (paired NW test on squared deviations)
            base = res["sample"].returns
            for est, r in res.items():
                if est == "sample":
                    continue
                dvar = (r.returns - r.returns.mean()) ** 2 - (base - base.mean()) ** 2
                rows.append({"universe": uname, "window": window, "estimator": f"{est} - sample (var diff)",
                             "volatility": float(dvar.mean() * 12), "sharpe": nw_mean_t(dvar), "turnover": np.nan,
                             "max_drawdown": np.nan, "errors": 0})
    d = out_dir("E3")
    pd.DataFrame(rows).to_csv(d / "covariance.csv", index=False)
    pd.DataFrame(rets).to_csv(d / "covariance_returns.csv")

    print("E3 sensitivity: expected-return inputs (mean-variance, multi_asset)")
    u = load_universe("multi_asset")
    mom = panel_zscore(reference.momentum_12_1(u.returns))
    strategies = {}
    for mm in ("sample", "bayes_stein", "james_stein", "equilibrium", "zero", "signal"):
        strategies[f"MV {mm}"] = (OptimizerStrategy("mean_variance", mean_method=mm, constraints=C25, signal="momentum",
                                                    market_weights=sixty_forty_weights), cfg())
    res = run_set(u, strategies, {"momentum": mom})
    save_set("E3", "means", res)

    print("E3 sensitivity: window length (common sample from the 240-month start)")
    strategies = {}
    for w in (60, 120, 240):
        strategies[f"MV sample w{w}"] = (OptimizerStrategy("mean_variance", constraints=C25, window=w), cfg(window=240))
        strategies[f"Min variance w{w}"] = (OptimizerStrategy("min_variance", constraints=C25, window=w), cfg(window=240))
        strategies[f"Max Sharpe BS w{w}"] = (OptimizerStrategy("max_sharpe", mean_method="bayes_stein", constraints=C25, window=w),
                                             cfg(window=240))
    strategies["1/N"] = (EqualWeight(), cfg(window=240))
    res = run_set(u, strategies)
    save_set("E3", "windows", res)


# ------------------------------------------------------------------------------------------------ E4

def regime_series(u) -> pd.Series:
    mkt = u.factors["Mkt-RF"] if "Mkt-RF" in u.factors else u.returns.mean(axis=1)
    vol = mkt.rolling(12, min_periods=12).std()
    med = vol.expanding(min_periods=60).median()
    return (vol > med).astype(float).where(med.notna())


def e4():
    print("E4 signal combination")
    d = out_dir("E4")
    all_ic, all_stats, all_tests, weights_out = [], [], [], {}
    for uname, start in (("industries49", "1975-01-31"), ("industries12", "1965-01-31")):
        u = load_universe(uname)
        raw = reference.build(u)
        zs = {k: panel_zscore(v) for k, v in raw.items()}
        ex = u.returns.sub(u.rf, axis=0)
        fwd = forward_returns(ex)
        regimes = regime_series(u)
        combos = {}
        for m in ("equal", "ic_weighted", "shrunk", "mean_var", "bma", "regime", "ridge"):
            t0 = time.time()
            combos[m] = combine(zs, fwd, m, min_history=60, window=120, regimes=regimes)
            print(f"  {uname} {m:12s} {time.time() - t0:5.1f}s", flush=True)
        # single signals too, for context
        ics = {f"signal:{k}": cross_sectional_ic(z, fwd) for k, z in zs.items()}
        ics.update({m: cross_sectional_ic(c.scores, fwd) for m, c in combos.items()})
        ic = pd.DataFrame(ics).loc[start:]
        ic.to_csv(d / f"ic_{uname}.csv")
        for k in ic.columns:
            s = ic[k].dropna()
            all_ic.append({"universe": uname, "method": k, "ic_mean": s.mean(), "ic_t": nw_mean_t(s),
                           "hit": (s > 0).mean(), "months": len(s)})
        pvals = {}
        for m in ("ic_weighted", "shrunk", "mean_var", "bma", "regime", "ridge"):
            diff = (ic[m] - ic["equal"]).dropna()
            t = nw_mean_t(diff)
            from scipy import stats as st
            p = float(2 * (1 - st.norm.cdf(abs(t))))
            pvals[m] = p
            all_tests.append({"universe": uname, "method": m, "ic_diff": diff.mean(), "t": t, "p": p})
        adj = holm(pvals)
        for row in all_tests:
            if row["universe"] == uname:
                row["p_holm"] = adj[row["method"]]
        # portfolio check: dollar-neutral book proportional to the combined score
        res = {}
        for m, c in combos.items():
            strat = SignalLongShort(signal=m, gross=1.0, name=m)
            conf = cfg(BASE, window=1, start=start)
            conf.name = m
            res[m] = run_backtest(strat, u.returns, u.rf, conf, universe=u, signals={m: c.scores})
        for m, r in res.items():
            s = r.stats()
            all_stats.append({"universe": uname, "method": m, "sharpe": s["sharpe"], "ann_excess": s["ann_excess"],
                              "volatility": s["volatility"], "max_drawdown": s["max_drawdown"], "turnover": s["turnover"],
                              "cost_drag": s["cost_drag"]})
        pd.DataFrame({m: r.excess for m, r in res.items()}).to_csv(d / f"ls_returns_{uname}.csv")
        weights_out[uname] = {m: c.weights.loc[start:].round(4).reset_index().assign(
            index=lambda x: x["index"].dt.strftime("%Y-%m-%d")).values.tolist() for m, c in combos.items()}
        incl = combos["bma"].info.get("inclusion", {})
        if incl:
            pd.DataFrame(incl, index=list(zs)).T.loc[start:].to_csv(d / f"bma_inclusion_{uname}.csv")
    pd.DataFrame(all_ic).to_csv(d / "ic_summary.csv", index=False)
    pd.DataFrame(all_tests).to_csv(d / "tests.csv", index=False)
    pd.DataFrame(all_stats).to_csv(d / "portfolio_stats.csv", index=False)
    (d / "signal_weights.json").write_text(json.dumps(weights_out), encoding="utf-8")


# ------------------------------------------------------------------------------------------------ E5

def e5(seeds: int = 4):
    print("E5 synthetic signal lab")
    cases = {
        "strong, stable": dict(ics=(0.08, 0.04, 0.0, 0.0)),
        "weak, stable": dict(ics=(0.03, 0.02, 0.01, 0.0)),
        "time-varying": dict(ics=(0.04, 0.04, 0.0, 0.0), ic_noise=0.05),
        "regime flips": dict(ics=(0.05, 0.03, 0.03, 0.0), regime_flip=(1, 2)),
        "many junk signals": dict(ics=(0.05, 0.0, 0.0, 0.0, 0.0, 0.0)),
    }
    rows = []
    for case, kw in cases.items():
        for seed in range(seeds):
            lab = synthetic.simulate(n_assets=49, n_periods=480, seed=100 + seed, **kw)
            fwd = lab["fwd"]
            oracle = cross_sectional_ic(lab["oracle"], fwd).iloc[120:].mean()
            for m in ("equal", "ic_weighted", "shrunk", "mean_var", "bma", "regime", "ridge"):
                c = combine(lab["zs"], fwd, m, min_history=60, window=120, regimes=lab["regime"])
                ic = cross_sectional_ic(c.scores, fwd).iloc[120:].mean()
                rows.append({"case": case, "seed": seed, "method": m, "ic": ic, "oracle_ic": oracle})
        print(f"  {case} done", flush=True)
    df = pd.DataFrame(rows)
    df["share_of_oracle"] = df["ic"] / df["oracle_ic"]
    d = out_dir("E5")
    df.to_csv(d / "lab.csv", index=False)
    df.groupby(["case", "method"])[["ic", "oracle_ic", "share_of_oracle"]].mean().to_csv(d / "summary.csv")


# ------------------------------------------------------------------------------------------------ E6-E8

def e6():
    print("E6 strategy sleeves")
    u = load_universe("sleeves")
    strategies = {
        "1/N": (EqualWeight(), cfg()),
        "Inverse vol": (OptimizerStrategy("inverse_vol"), cfg()),
        "Min variance": (OptimizerStrategy("min_variance", constraints=C40), cfg()),
        "ERC": (OptimizerStrategy("erc"), cfg()),
        "HRP": (OptimizerStrategy("hrp"), cfg()),
        "MV Bayes-Stein": (OptimizerStrategy("mean_variance", mean_method="bayes_stein", constraints=C40), cfg()),
        "Max Sharpe Bayes-Stein": (OptimizerStrategy("max_sharpe", mean_method="bayes_stein", constraints=C40), cfg()),
    }
    res = run_set(u, strategies)
    save_set("E6", "base", res)
    compare(res, "1/N").to_csv(out_dir("E6") / "tests.csv")
    corr = u.returns.sub(u.rf, axis=0).corr()
    corr.to_csv(out_dir("E6") / "sleeve_correlations.csv")


def e7():
    print("E7 risk overlays")
    for uname in ("multi_asset", "sleeves"):
        u = load_universe(uname)
        strategies = {}
        for label, base in (("1/N", EqualWeight()), ("ERC", OptimizerStrategy("erc"))):
            strategies[label] = (base, cfg())
            strategies[f"{label} + vol target 10%"] = (base, cfg(vol_target=0.10, max_leverage=1.0))
            strategies[f"{label} + drawdown governor"] = (base, cfg(drawdown_limit=0.20, drawdown_floor=0.25))
        res = run_set(u, strategies)
        save_set("E7", uname, res)


def e8():
    print("E8 Kelly sizing (multi_asset)")
    u = load_universe("multi_asset")
    lev = Constraints(lower=0.0, upper=None, budget=None, net_min=0.0, net_max=2.0)
    strategies = {"1/N": (EqualWeight(), cfg())}
    for f in (1.0, 0.5, 0.25):
        s = OptimizerStrategy("kelly" if f == 1 else "kelly_half", constraints=lev, name=f"Kelly {f:g}")
        if f == 0.25:
            s = OptimizerStrategy("mean_variance", constraints=lev, risk_aversion=4.0, name="Kelly 0.25")
        strategies[f"Kelly {f:g}"] = (s, cfg())
    res = run_set(u, strategies)
    stats = save_set("E8", "base", res)
    growth = {k: float(np.log1p(r.returns).mean() * 12) for k, r in res.items()}
    pd.Series(growth, name="log_growth").to_csv(out_dir("E8") / "log_growth.csv")


# ------------------------------------------------------------------------------------------------ E9

def adv_proxy(u) -> pd.DataFrame:
    """Average daily traded value: 0.4% of industry market cap a day; $50bn for each Treasury."""
    adv = u.mcap.reindex(columns=u.assets) * 1e6 * 0.004
    for b in [a for a in u.assets if a.startswith("UST")]:
        adv[b] = 5e10
    return adv.ffill()


def e9():
    print("E9 capacity (square-root impact)")
    u = load_universe("multi_asset")
    adv = adv_proxy(u)
    rows = []
    for aum in (1e8, 1e9, 1e10):
        costs = CostModel(bps=BASE, impact_eta=0.5, aum=aum, adv=adv)
        for name, strat in (("1/N", EqualWeight()), ("ERC", OptimizerStrategy("erc")),
                            ("Cost-aware MV", OptimizerStrategy("mean_variance", mean_method="bayes_stein", constraints=C25,
                                                                cost_aware=True, trade_cost=TRADE_COST))):
            conf = BacktestConfig(window=120, costs=costs, name=name)
            r = run_backtest(strat, u.returns, u.rf, conf, universe=u)
            s = r.stats()
            rows.append({"aum": aum, "strategy": name, "sharpe": s["sharpe"], "cost_drag": s["cost_drag"],
                         "turnover": s["turnover"]})
            print(f"  ${aum / 1e9:5.1f}bn {name:14s} SR {s['sharpe']:.2f} cost {s['cost_drag']:.3%}", flush=True)
    d = out_dir("E9")
    pd.DataFrame(rows).to_csv(d / "capacity.csv", index=False)
    # integration check: the same 1/N book executed through the router adapter (stub router, CLI transport)
    router = SmartOrderRouterAdapter("cli", [sys.executable, str(ROOT / "examples" / "stub_router.py")])
    window = u.window("2014-01-31", None)
    rows = []
    for label, ex in (("built-in cost model", None), ("router adapter (stub)", router)):
        conf = BacktestConfig(window=12, costs=CostModel(bps=BASE, impact_eta=0.5, aum=1e9, adv=adv), executor=ex, name=label)
        r = run_backtest(EqualWeight(), window.returns, window.rf, conf, universe=window)
        rows.append({"executor": label, "cost_drag": r.stats()["cost_drag"], "rebalances": len(r.costs),
                     "errors": len(r.log)})
    pd.DataFrame(rows).to_csv(d / "router_check.csv", index=False)


# ------------------------------------------------------------------------------------------------ E10

def e10():
    print("E10 risk model checks")
    d = out_dir("E10")
    u = load_universe("multi_asset", daily=True)
    daily = u.daily
    ew = daily.mean(axis=1)
    out = {}
    for method in ("historical", "normal", "cornish_fisher"):
        bt = rolling_var_backtest(ew.values, 500, 0.99, method)
        out[method] = {"kupiec": bt["kupiec"], "independence": bt["independence"]}
        print(f"  VaR99 {method:15s} exceptions {bt['kupiec']['exceptions']} vs {bt['kupiec']['expected']:.0f} "
              f"(p {bt['kupiec']['p_value']:.3f}); clustering p {bt['independence']['p_value']:.3f}", flush=True)
    out["sample"] = {"first": str(daily.index[0].date()), "last": str(daily.index[-1].date()), "days": int(len(daily))}
    (d / "var_backtest.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")

    # risk report for portfolios as of the last decision date
    ex = u.returns.sub(u.rf, axis=0)
    hist = ex.iloc[-120:]
    cov = pd.DataFrame(ledoit_wolf_constant_correlation(hist.values)[0] * 12, ex.columns, ex.columns)
    mu = hist.mean() * 12
    ports = {"1/N": pd.Series(1 / len(ex.columns), ex.columns),
             "ERC": pd.Series(erc_weights(cov.values), ex.columns),
             "Min variance": min_variance(cov, C25).weights,
             "Max Sharpe sample": max_sharpe(mu, cov, C25).weights}
    factors = u.factors.copy()
    factors["Rates"] = ex["UST10"]
    fac = factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "Rates"]]
    betas = scenario_betas(ex.drop(columns=[]), fac, 120)
    scen = load_scenarios(ROOT / "scenarios")
    last2y = daily.iloc[-504:]
    adv = adv_proxy(u).iloc[-1]
    report = {}
    for name, w in ports.items():
        port_d = last2y.values @ w.values
        c = contributions(w, cov)
        fd = factor_decomposition(w, ex.iloc[-120:], fac.iloc[-120:].dropna())
        stress = []
        for sc in scen:
            r = run_scenario(w, sc, betas=betas, returns=u.returns, cov=cov)
            stress.append({"scenario": sc.name, "kind": sc.kind, "pnl": r["pnl"], "stressed_var": r.get("stressed_var")})
        vol = float(np.sqrt(w.values @ cov.values @ w.values))
        report[name] = {
            "weights": w.round(4).to_dict(),
            "volatility": vol,
            "var99_1d": {"historical": historical_var(port_d, 0.99), "normal": parametric_var(port_d.mean(), port_d.std(), 0.99),
                         "cornish_fisher": cornish_fisher_var(port_d, 0.99),
                         "monte_carlo_t5": monte_carlo_var(w.values, np.zeros(len(w)), last2y.cov().values, 0.99, 50000)[0]},
            "es975_1d": {"historical": historical_es(port_d, 0.975), "normal": parametric_es(port_d.mean(), port_d.std(), 0.975)},
            "risk_share": c["share"].round(4).to_dict(),
            "factor_exposures": fd["exposures"].round(3).to_dict(),
            "factor_share": fd["factor_contrib"].round(4).to_dict(),
            "idiosyncratic_share": fd["idiosyncratic_share"],
            "concentration": concentration(w.values, cov.values),
            "liquidity_1bn": liquidity_profile(w, 1e9, adv, 0.1, (1, 5, 20)),
            "stress": stress,
        }
    (d / "risk_report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    cb = correlation_breakdown(ex, factors["Mkt-RF"], bonds=["UST10", "UST5"])
    mkt = factors["Mkt-RF"].reindex(ex.index)
    sb = {int(dec): round(float(g.corr(mkt.loc[g.index])), 3) for dec, g in ex["UST10"].groupby((ex.index.year // 10) * 10)}
    (d / "correlation.json").write_text(json.dumps({"industries": cb, "stock_bond_corr_by_decade": sb},
                                                   indent=1, default=float), encoding="utf-8")


def x1():
    """EXPLORATORY (not pre-registered; added after E4's result): does learning beat equal weights once
    the residual-reversal signal's sign problem is handled by hand?  Industries trend at one month
    (Moskowitz and Grinblatt 1999), so a careful analyst might have dropped or flipped that signal."""
    print("X1 exploratory: sign-aware equal weights (E4 follow-up)")
    rows = []
    for uname, start in (("industries49", "1975-01-31"), ("industries12", "1965-01-31")):
        u = load_universe(uname)
        zs = {k: panel_zscore(v) for k, v in reference.build(u).items()}
        fwd = forward_returns(u.returns.sub(u.rf, axis=0))
        variants = {"equal (pre-registered)": zs,
                    "equal, reversal dropped": {k: v for k, v in zs.items() if k != "residual_reversal"},
                    "equal, reversal flipped": {k: (-v if k == "residual_reversal" else v) for k, v in zs.items()}}
        ics = {name: cross_sectional_ic(combine(z, fwd, "equal").scores, fwd).loc[start:] for name, z in variants.items()}
        ics["shrunk (learned)"] = cross_sectional_ic(combine(zs, fwd, "shrunk", min_history=60, window=120).scores, fwd).loc[start:]
        for name, ic in ics.items():
            diff = (ics["shrunk (learned)"] - ic).dropna()
            rows.append({"universe": uname, "variant": name, "ic_mean": ic.mean(), "ic_t": nw_mean_t(ic),
                         "shrunk_minus_this": diff.mean(), "t_diff": nw_mean_t(diff) if name != "shrunk (learned)" else np.nan})
    pd.DataFrame(rows).to_csv(out_dir("E4") / "exploratory_sign_aware.csv", index=False)
    print(pd.DataFrame(rows).round(4).to_string())


def x2():
    """EXPLORATORY: a conditional (EWMA) VaR against the pre-registered unconditional ones (E10)."""
    print("X2 exploratory: EWMA VaR backtest")
    u = load_universe("multi_asset", daily=True)
    ew = u.daily.mean(axis=1)
    out = {}
    for method in ("ewma", "fhs"):
        bt = rolling_var_backtest(ew.values, 500, 0.99, method)
        out[method] = {"kupiec": bt["kupiec"], "independence": bt["independence"]}
        print(f"  VaR99 {method:5s} exceptions {bt['kupiec']['exceptions']} vs {bt['kupiec']['expected']:.0f} "
              f"(p {bt['kupiec']['p_value']:.3f}); clustering p {bt['independence']['p_value']:.3f}")
    (out_dir("E10") / "var_backtest_conditional.json").write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")


def x3():
    """EXPLORATORY: execution through the separate smart order router (multi-venue-smart-router).

    (a) router shortfall per routing strategy on its standard parent order (12 seeds x buy/sell);
    (b) the last rebalance of E1's cost-aware mean-variance book sent through the adapter, order by order;
    (c) what smart instead of naive routing is worth per year for each E1 strategy, given its turnover.
    """
    import os
    from qpm.execution import ParentOrder, validate_report
    from qpm.execution import smartrouter_bridge as sb
    os.environ.setdefault("QPM_SMARTROUTER_SRC", str(ROOT.parent / "multi-venue-smart-router" / "src"))
    print("X3 exploratory: smart order router integration")
    d = out_dir("X3")
    table = sb.cost_table(("nbbo", "sync", "smart"), range(1, 13))
    pd.DataFrame(table).T.to_csv(d / "router_costs.csv")
    print(pd.DataFrame(table).T.round(3).to_string())
    wts = json.loads((RESULTS / "E1" / "weights_base.json").read_text())["Cost-aware MV"]
    assets, last, prev, date = wts["assets"], np.array(wts["w"][-1]), np.array(wts["w"][-2]), wts["dates"][-1]
    aum, orders = 1e9, []
    for a, t in zip(assets, last - prev):
        if abs(t) * aum < 1e5:
            continue
        orders.append(ParentOrder(f"{date}-{a}", a, "buy" if t > 0 else "sell", abs(t) * aum / 100.0, 100.0,
                                  f"{date}T16:00:00", urgency="medium"))
    router = SmartOrderRouterAdapter("python", callable_path="qpm.execution.smartrouter_bridge:route")
    reps = router.execute(orders)
    rows = [{"order": o.order_id, "side": o.side, "notional": o.notional, "filled_share": r.filled_quantity / o.quantity,
             "shortfall_bps": r.shortfall_bps, "venues": len(r.fills), "contract_problems": len(validate_report(o, r))}
            for o, r in zip(orders, reps)]
    pd.DataFrame(rows).to_csv(d / "last_rebalance.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string())
    stats = pd.read_csv(RESULTS / "E1" / "stats_base.csv", index_col=0)
    saving = table["nbbo"]["is_bps"] - table["smart"]["is_bps"]
    value = (stats["turnover"] * saving / 1e4).rename("smart_vs_naive_per_year")
    value.to_csv(d / "routing_value.csv")
    print(f"  smart saves {saving:.2f} bp per unit traded; per year: {value.min():.5%} to {value.max():.5%}")


EXPERIMENTS = {"E1": e1, "E2": e2, "E3": e3, "E4": e4, "E5": e5, "E6": e6, "E7": e7, "E8": e8, "E9": e9, "E10": e10,
               "X1": x1, "X2": x2, "X3": x3}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=[k for k in EXPERIMENTS if k.startswith("E")] + ["X1", "X2"],
                    help="X3 needs the multi-venue-smart-router repository next to this one (or QPM_SMARTROUTER_SRC)")
    args = ap.parse_args(argv)
    for key in args.only:
        t0 = time.time()
        EXPERIMENTS[key]()
        print(f"{key} finished in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
