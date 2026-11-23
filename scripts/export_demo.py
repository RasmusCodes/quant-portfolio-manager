"""Write the browser demo's data files and the parity-test fixtures.

    python scripts/export_demo.py            # demo/data/*.json + demo/tests/fixtures.json

Data: monthly returns (rounded to 0.001%) of the two demo universes, factor returns for betas, the
built-in signals as of the last month with their walk-forward track records, the stress scenarios, and
(when results/ exists) the study's headline results.  Sources are credited on the page.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpm.data import load_universe  # noqa: E402
from qpm.estimation import bayes_stein, ledoit_wolf_constant_correlation  # noqa: E402
from qpm.optimize import (Constraints, bl_posterior, efficient_frontier, erc_weights, explain, hrp_weights,  # noqa: E402
                          max_sharpe, mean_variance, min_variance)
from qpm.risk import cornish_fisher_var, historical_es, historical_var, parametric_var  # noqa: E402
from qpm.signals import cross_sectional_ic, forward_returns, ic_summary, panel_zscore, reference  # noqa: E402

OUT = ROOT / "demo" / "data"
RESULTS = ROOT / "results"


def r5(x):
    return np.round(np.asarray(x, dtype=float), 6).tolist()


def universe_payload(name: str) -> dict:
    u = load_universe(name)
    ex = u.returns.sub(u.rf, axis=0)
    fac = u.factors.copy() if u.factors is not None else pd.DataFrame(index=u.returns.index)
    if name == "multi_asset":
        fac["Rates"] = ex["UST10"]
    else:
        bonds = pd.read_csv(ROOT / "data" / "processed" / "ust_monthly.csv", index_col=0, parse_dates=True)
        fac["Rates"] = bonds["UST10"].reindex(fac.index) - u.rf
    fac = fac[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "Rates"]]
    mcap = None
    if u.mcap is not None:
        last = u.mcap.iloc[-1].reindex(u.assets)
        mcap = {a: (None if pd.isna(v) else float(v)) for a, v in last.items()}
    return {
        "name": name,
        "assets": u.assets,
        "labels": u.labels,
        "groups": u.group,
        "dates": [d.strftime("%Y-%m") for d in u.returns.index],
        "returns": r5(u.returns.values),
        "rf": r5(u.rf.values),
        "factors": {"names": list(fac.columns), "values": r5(fac.fillna(0.0).values),
                    "available_from": {c: fac[c].first_valid_index().strftime("%Y-%m") for c in fac.columns}},
        "mcap_last": mcap,
        "source": "Kenneth R. French Data Library (industry portfolios, factors, T-bill); FRED H.15 yields rebuilt "
                  "into constant-maturity Treasury returns",
    }


def signals_payload() -> dict:
    """Built-in signals as of the last month, with track records computed walk-forward."""
    u = load_universe("industries12")
    raw = reference.build(u)
    ex = u.returns.sub(u.rf, axis=0)
    fwd = forward_returns(ex)
    last = u.returns.index[-1]
    out = []
    for key, panel in raw.items():
        z = panel_zscore(panel)
        ic = cross_sectional_ic(z, fwd).loc["1965-01-31":]
        known = ic.iloc[:-1]                       # the last IC needs next month's return
        summ = ic_summary(known)
        spec = reference.SPECS[key]
        out.append({"source": f"reference.{key}", "label": spec["label"], "family": spec["family"],
                    "kind": "score", "horizon_days": spec["horizon_days"], "model_version": "1.0",
                    "timestamp": last.strftime("%Y-%m-%d"), "provenance": f"public data: {spec['inputs']}",
                    "values": {a: (None if pd.isna(v) else round(float(v), 4)) for a, v in z.loc[last].items()},
                    "raw": {a: (None if pd.isna(v) else round(float(v), 5)) for a, v in panel.loc[last].items()},
                    "track_record": {k: (None if pd.isna(v) else round(float(v), 4)) for k, v in summ.items()},
                    "ic_history": r5(known.iloc[-240:].values), "illustrative": False})
    # Treasuries: time-series signals (only two bonds, so "IC" is the time-series correlation with the
    # next month's excess return, pooled over both bonds)
    m = load_universe("multi_asset")
    exm = m.returns.sub(m.rf, axis=0)
    ylds = pd.read_csv(ROOT / "data" / "processed" / "yields_monthly.csv", index_col=0, parse_dates=True)
    carry = reference.term_carry(ylds, m.rf, {"UST10": "DGS10", "UST5": "DGS5"}).reindex(exm.index)
    tsm = reference.tsmom(exm[["UST10", "UST5"]])
    for key, panel, label in (("bond_carry", carry, "Bond carry"), ("bond_tsmom", tsm, "Bond trend (TSMOM)")):
        zts = (panel - panel.rolling(120, min_periods=60).mean()) / panel.rolling(120, min_periods=60).std()
        nxt = exm[["UST10", "UST5"]].shift(-1)
        both = pd.concat([zts.stack(), nxt.stack()], axis=1).dropna()
        both.columns = ["z", "r"]
        by_month = both.groupby(level=0).apply(lambda g: np.sign(g["z"]).mul(g["r"]).mean())
        corr = float(both["z"].corr(both["r"]))
        hit = float((np.sign(both["z"]) == np.sign(both["r"])).mean())
        out.append({"source": f"reference.{key}", "label": label, "family": "macro_fixed_income", "kind": "score",
                    "horizon_days": 21, "model_version": "1.0", "timestamp": last.strftime("%Y-%m-%d"),
                    "provenance": "public data: H.15 yields, T-bill; z-score vs own trailing 120 months",
                    "values": {a: round(float(zts[a].loc[last]), 4) for a in ("UST10", "UST5")},
                    "raw": {a: round(float(panel[a].loc[last]), 5) for a in ("UST10", "UST5")},
                    "track_record": {"ic_mean": round(corr, 4), "hit_rate": round(hit, 4),
                                     "n_obs": int(both.index.get_level_values(0).nunique()),
                                     "ic_t": round(float(by_month.mean() / by_month.std() * np.sqrt(len(by_month))), 2),
                                     "note": "time-series correlation pooled over both bonds; t from the monthly sign-weighted return"},
                    "ic_history": r5(by_month.iloc[-240:].values / max(by_month.std(), 1e-9) * 0.05), "illustrative": False})
    examples = [  # illustrative rows showing the other research families; not model outputs
        ("example.vol_skew", "Implied-volatility skew", "options_volatility", {"BusEq": -0.5, "Utils": 0.3, "Hlth": 0.2}, "score"),
        ("example.altdata_inventory", "Inventory-to-sales screen", "alternative_data", {"BusEq": 0.6, "Manuf": 0.3, "Durbl": -0.4}, "score"),
        ("example.earnings_revisions", "Earnings revisions", "earnings_event", {"BusEq": 0.8, "Hlth": 0.2, "Enrgy": -0.6}, "score"),
        ("example.credit_spread", "Credit spread momentum", "credit", {"Money": -0.7, "UST10": 0.5, "UST5": 0.4}, "score"),
        ("example.order_flow", "Order-flow imbalance", "microstructure", {"Shops": 0.3, "Telcm": -0.2}, "score"),
    ]
    for src, label, fam, vals, kind in examples:
        vals = {k: v for k, v in vals.items() if v is not None}
        if not vals:
            continue
        out.append({"source": src, "label": label, "family": fam, "kind": kind, "horizon_days": 21, "model_version": "example",
                    "timestamp": last.strftime("%Y-%m-%d"), "provenance": "illustrative example, not a model output",
                    "values": vals, "raw": vals, "track_record": None, "ic_history": None, "illustrative": True})
    return {"as_of": last.strftime("%Y-%m-%d"), "signals": out}


def study_payload() -> dict:
    out: dict = {}
    def read(path, **kw):
        p = RESULTS / path
        return pd.read_csv(p, **kw) if p.exists() else None
    for exp, tags in (("E1", ("base", "high")), ("E6", ("base",)), ("E7", ("multi_asset", "sleeves")), ("E8", ("base",))):
        for tag in tags:
            stats = read(f"{exp}/stats_{tag}.csv", index_col=0)
            rets = read(f"{exp}/returns_{tag}.csv", index_col=0, parse_dates=True)
            if stats is None or rets is None:
                continue
            keep = ["ann_return", "volatility", "sharpe", "max_drawdown", "turnover", "cost_drag", "ce_return"]
            out[f"{exp}_{tag}"] = {
                "stats": stats[[c for c in keep if c in stats]].round(4).to_dict(orient="index"),
                "dates": [d.strftime("%Y-%m") for d in rets.index],
                "growth": {c: r5(np.log((1 + rets[c].fillna(0)).cumprod())) for c in rets.columns},
            }
            tests = read(f"{exp}/tests_{tag}.csv", index_col=0) if exp == "E1" else read(f"{exp}/tests.csv", index_col=0)
            if tests is not None:
                out[f"{exp}_{tag}"]["tests"] = tests.round(4).to_dict(orient="index")
    e2 = read("E2/summary.csv")
    if e2 is not None:
        out["E2"] = e2.round(4).to_dict(orient="records")
    for f in ("ic_summary", "tests", "portfolio_stats"):
        t = read(f"E4/{f}.csv")
        if t is not None:
            out.setdefault("E4", {})[f] = t.round(4).to_dict(orient="records")
    e5 = read("E5/summary.csv")
    if e5 is not None:
        out["E5"] = e5.round(4).to_dict(orient="records")
    e3 = read("E3/covariance.csv")
    if e3 is not None:
        out["E3_cov"] = e3.round(5).to_dict(orient="records")
    e9 = read("E9/capacity.csv")
    if e9 is not None:
        out["E9"] = e9.round(5).to_dict(orient="records")
    for f in ("var_backtest.json", "correlation.json"):
        p = RESULTS / "E10" / f
        if p.exists():
            out.setdefault("E10", {})[f.split(".")[0]] = json.loads(p.read_text(encoding="utf-8"))
    return out


def fixtures() -> dict:
    """Inputs and Python outputs the JavaScript engine must reproduce (demo/tests.html)."""
    rng = np.random.default_rng(2026)
    cases = []
    for k in range(6):
        n = int(rng.integers(5, 12))
        g = rng.normal(size=(n, n))
        cov = g @ g.T / n * 0.04 + np.diag(rng.uniform(0.002, 0.02, n))
        mu = rng.normal(0.06, 0.04, n)
        names = [f"a{i}" for i in range(n)]
        cons = Constraints(lower=0.0, upper=0.35, groups={"g": names[: n // 2]}, group_max={"g": 0.5},
                           turnover_max=0.6 if k % 2 else None, w_prev=np.full(n, 1 / n) if k % 2 else None)
        mv = mean_variance(mu, cov, 4.0, cons, names)
        e = explain(mv, cov, mu)
        w_mkt = rng.dirichlet(np.ones(n))
        P = np.eye(n)[[0, 2]]
        post = bl_posterior(cov, w_mkt, P, [0.08, 0.02], confidences=[0.5, 0.8])
        fr = efficient_frontier(mu, cov, Constraints(lower=0.0, upper=0.35), 5)
        cases.append({
            "n": n, "names": names, "mu": mu.tolist(), "cov": cov.tolist(),
            "cons": {"lower": 0.0, "upper": 0.35, "groups": [{"name": "g", "members": list(range(n // 2)), "min": None, "max": 0.5}],
                     "turnoverMax": 0.6 if k % 2 else None, "wPrev": (np.full(n, 1 / n).tolist() if k % 2 else None)},
            "mean_variance": mv.weights.values.tolist(),
            "weight_from_formula": e["table"]["weight_from_formula"].values.tolist(),
            "min_variance": min_variance(cov, Constraints(lower=0.0, upper=0.35), names).weights.values.tolist(),
            "max_sharpe": max_sharpe(mu, cov, Constraints(lower=0.0, upper=0.35), names).weights.values.tolist(),
            "erc": erc_weights(cov).tolist(),
            "hrp": hrp_weights(cov)[0].tolist(),
            "bl": {"w_mkt": w_mkt.tolist(), "P": P.tolist(), "Q": [0.08, 0.02], "conf": [0.5, 0.8], "mu": post["mu"].tolist()},
            "frontier_vol": fr["volatility"].tolist(),
        })
    x = rng.standard_t(5, (150, 6)) * 0.04
    port = x[:, 0]
    return {
        "cases": cases,
        "estimators": {"returns": x.tolist(), "lw_cc": ledoit_wolf_constant_correlation(x)[0].tolist(),
                       "bayes_stein": bayes_stein(x)[0].tolist()},
        "var": {"series": port.tolist(), "historical": historical_var(port, 0.99), "es": historical_es(port, 0.99),
                "normal": parametric_var(port.mean(), port.std(ddof=1), 0.99), "cornish_fisher": cornish_fisher_var(port, 0.99)},
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    payloads = {f"universe_{name}": universe_payload(name) for name in ("multi_asset", "sleeves")}
    payloads["signals"] = signals_payload()
    payloads["scenarios"] = [json.loads(p.read_text(encoding="utf-8")) | {"key": p.stem}
                             for p in sorted((ROOT / "scenarios").glob("*.json"))]
    payloads["study"] = study_payload()
    for name, obj in payloads.items():
        (OUT / f"{name}.json").write_text(json.dumps(obj, separators=(",", ":"), default=float), encoding="utf-8")
    # one script bundle so the page needs no fetch() (works from a file and on any static host)
    bundle = "window.QPM_DATA = " + json.dumps(payloads, separators=(",", ":"), default=float) + ";\n"
    (OUT / "bundle.js").write_text(bundle, encoding="utf-8")
    tests = ROOT / "demo" / "tests"
    tests.mkdir(parents=True, exist_ok=True)
    fx = fixtures()
    (tests / "fixtures.json").write_text(json.dumps(fx), encoding="utf-8")
    (tests / "fixtures.js").write_text("window.QPM_FIXTURES = " + json.dumps(fx) + ";\n", encoding="utf-8")
    for p in sorted(OUT.glob("*")):
        print(f"{p.name:28s} {p.stat().st_size / 1024:8.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
