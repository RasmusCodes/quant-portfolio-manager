"""Figures and tables for reports/REPORT.md from results/.

    python scripts/make_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FIG = ROOT / "reports" / "figures"
TAB = ROOT / "reports" / "tables"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans"], "font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
    "axes.edgecolor": "#c3c2b7", "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "semibold",
    "axes.titlesize": 11, "axes.titlelocation": "left", "lines.linewidth": 2, "lines.solid_capstyle": "round",
    "text.parse_math": False, "legend.frameon": False,
})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / name, dpi=160, bbox_inches="tight")
    plt.close(fig)


def md_table(df: pd.DataFrame, fmt: dict, name: str) -> str:
    TAB.mkdir(parents=True, exist_ok=True)
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            f = fmt.get(c, "{}")
            cells.append("" if (isinstance(v, float) and np.isnan(v)) else (f.format(v) if not isinstance(v, str) else v))
        lines.append("| " + " | ".join([str(idx)] + cells) + " |")
    text = "\n".join(lines) + "\n"
    (TAB / f"{name}.md").write_text(text, encoding="utf-8")
    return text


def e1():
    s = pd.read_csv(RES / "E1/stats_base.csv", index_col=0)
    sh = pd.read_csv(RES / "E1/stats_high.csv", index_col=0)
    t = pd.read_csv(RES / "E1/tests_base.csv", index_col=0).reindex(s.index)
    ci = pd.Series([f"[{a:+.2f}, {b:+.2f}]" if pd.notna(a) else "" for a, b in zip(t["ci_low"], t["ci_high"])],
                   index=s.index)
    tab = pd.DataFrame({
        "Return": s["ann_return"], "Volatility": s["volatility"], "Sharpe": s["sharpe"],
        "Sharpe, high costs": sh["sharpe"].reindex(s.index), "Max drawdown": s["max_drawdown"],
        "Turnover/yr": s["turnover"], "vs 1/N": t["diff"], "95% CI": ci, "p (Holm)": t["p_boot_holm"]},
        index=s.index)   # keeps E1's family order: baselines, risk-based, return-based, overlays
    tab.index.name = "Method"
    t = t.dropna(subset=["diff"])
    md_table(tab, {"Return": "{:.1%}", "Volatility": "{:.1%}", "Sharpe": "{:.2f}", "Sharpe, high costs": "{:.2f}",
                   "Max drawdown": "{:.0%}", "Turnover/yr": "{:.2f}", "vs 1/N": "{:+.2f}", "p (Holm)": "{:.2f}"}, "e1")
    # figure 1: Sharpe difference vs 1/N with bootstrap CI
    order = t.sort_values("diff").index
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(order))
    ax.hlines(y, t.loc[order, "ci_low"], t.loc[order, "ci_high"], color=SERIES[0], linewidth=2)
    ax.plot(t.loc[order, "diff"], y, "o", color=SERIES[0], markersize=7, markeredgecolor=SURFACE, markeredgewidth=2)
    ax.axvline(0, color=INK2, linewidth=1)
    ax.set_yticks(y, order)
    ax.set_xlabel("Net Sharpe ratio minus 1/N's (1972-2026), with 95% bootstrap interval")
    ax.set_title("No optimizer beats equal weights by more than chance")
    ax.grid(axis="y", visible=False)
    save(fig, "fig1_e1_sharpe_vs_1n.png")
    # figure 2: growth of $1
    r = pd.read_csv(RES / "E1/returns_base.csv", index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for i, c in enumerate(["1/N", "Min variance", "Max Sharpe sample", "Black-Litterman"]):
        g = (1 + r[c]).cumprod()
        ax.plot(g.index, g.values, color=SERIES[i], label=c)
        ax.annotate(f"{c} {g.iloc[-1]:.0f}x", (g.index[-1], g.iloc[-1]), xytext=(4, 0), textcoords="offset points",
                    va="center", fontsize=8.5, color=INK2)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_title("Growth of $1 after costs, multi-asset universe (log scale)")
    ax.legend(loc="upper left", fontsize=8.5)
    ax.set_xlim(r.index[0], r.index[-1] + pd.Timedelta(days=3650))
    save(fig, "fig2_e1_growth.png")


def e2():
    e = pd.read_csv(RES / "E2/summary.csv")
    sub = e[e["T"] == 120]
    methods = [("max_sharpe", "Max Sharpe"), ("mean_variance", "Mean-variance"), ("kelly_full", "Full Kelly"),
               ("min_variance", "Min variance"), ("erc", "ERC"), ("equal", "1/N")]
    cases = [("estimated", "Both estimated"), ("true_cov", "True covariance"), ("true_mean", "True means"), ("truth", "Truth")]
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    width = 0.19
    for j, (case, lab) in enumerate(cases):
        vals = [sub[(sub.method == m) & (sub.case == case)]["true_sr"].mean() if ((sub.method == m) & (sub.case == case)).any()
                else np.nan for m, _ in methods]
        ax.bar(np.arange(len(methods)) + (j - 1.5) * (width + 0.01), vals, width, color=SERIES[j], label=lab)
    promised = [sub[(sub.method == m) & (sub.case == "estimated")]["estimated_sr"].mean() for m, _ in methods]
    ax.plot(np.arange(len(methods)) - 1.5 * (width + 0.01), promised, "_", color=INK, markersize=16, markeredgewidth=2,
            label="Promised (in-sample)")
    ax.set_xticks(np.arange(len(methods)), [m for _, m in methods])
    ax.set_ylabel("Sharpe ratio under the true parameters")
    ax.set_ylim(0.0, 0.95)
    ax.set_title("Errors in means cost far more than errors in covariances (10-year histories)")
    ax.legend(fontsize=8, ncol=3, loc="upper right")
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_e2_estimation_error.png")
    rows = []
    for T in (60, 120, 240):
        for m, lab in methods:
            g = e[(e["T"] == T) & (e.method == m)].set_index("case")
            rows.append({"T (months)": T, "Method": lab, "Promised": g.loc["estimated", "estimated_sr"],
                         "Delivered": g.loc["estimated", "true_sr"],
                         "With true means": g["true_sr"].get("true_mean", np.nan),
                         "With true covariance": g.loc["true_cov", "true_sr"], "Truth": g.loc["truth", "true_sr"]})
    tab = pd.DataFrame(rows).set_index("T (months)")
    md_table(tab, {c: "{:.2f}" for c in tab.columns if c != "Method"}, "e2")


def e4():
    ic = pd.read_csv(RES / "E4/ic_summary.csv")
    tests = pd.read_csv(RES / "E4/tests.csv")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 4.0), sharex=True)
    labels = {"signal:momentum_12_1": "Momentum", "signal:value_bm": "Value", "signal:low_beta": "Low beta",
              "signal:residual_reversal": "Residual reversal", "equal": "Equal weights", "ic_weighted": "Dynamic confidence",
              "shrunk": "Shrinkage", "mean_var": "Mean-variance on ICs", "bma": "Bayesian model averaging",
              "regime": "Regime-dependent", "ridge": "Ridge ensemble"}
    for ax, (u, title) in zip(axes, (("industries12", "12 industries, 1965-2026"), ("industries49", "49 industries, 1975-2026"))):
        d = ic[ic.universe == u].set_index("method").reindex(list(labels))
        y = np.arange(len(d))[::-1]
        se = d["ic_mean"] / d["ic_t"]
        colors = [MUTED if k.startswith("signal:") else (SERIES[1] if k == "equal" else SERIES[0]) for k in d.index]
        ax.hlines(y, d["ic_mean"] - 1.96 * se, d["ic_mean"] + 1.96 * se, color=colors, linewidth=2)
        ax.scatter(d["ic_mean"], y, color=colors, s=42, edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.set_yticks(y, [labels[k] for k in d.index] if ax is axes[0] else [])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
    axes[0].set_xlabel("Mean monthly rank IC, 95% interval (Newey-West)")
    axes[1].set_xlabel("Mean monthly rank IC, 95% interval (Newey-West)")
    fig.suptitle("Single signals (gray), equal weights (orange) and learned combinations (blue)", x=0.01, ha="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    save(fig, "fig4_e4_ic.png")
    t = tests.copy()
    t["Method"] = t["method"].map(labels)
    t = t.set_index("universe")[["Method", "ic_diff", "t", "p_holm"]]
    t.index.name = "Universe"
    md_table(t, {"ic_diff": "{:+.3f}", "t": "{:.2f}", "p_holm": "{:.3f}"}, "e4_tests")
    p = pd.read_csv(RES / "E4/portfolio_stats.csv")
    p["Method"] = p["method"].map(labels)
    p = p.set_index("universe")[["Method", "sharpe", "ann_excess", "volatility", "max_drawdown", "turnover"]]
    p.index.name = "Universe"
    md_table(p, {"sharpe": "{:.2f}", "ann_excess": "{:+.1%}", "volatility": "{:.1%}", "max_drawdown": "{:.0%}",
                 "turnover": "{:.1f}"}, "e4_portfolios")
    x = pd.read_csv(RES / "E4/exploratory_sign_aware.csv").set_index("universe")
    x.index.name = "Universe"
    md_table(x, {"ic_mean": "{:.3f}", "ic_t": "{:.2f}", "shrunk_minus_this": "{:+.3f}", "t_diff": "{:.2f}"}, "x1")


def e5():
    s = pd.read_csv(RES / "E5/summary.csv")
    pv = s.pivot(index="case", columns="method", values="share_of_oracle")
    cols = ["equal", "ic_weighted", "shrunk", "mean_var", "bma", "regime", "ridge"]
    pv = pv[cols].reindex(["strong, stable", "weak, stable", "time-varying", "regime flips", "many junk signals"])
    pv.index.name = "Case"
    md_table(pv, {c: "{:.0%}" for c in cols}, "e5")


def others():
    c = pd.read_csv(RES / "E3/covariance.csv")
    base = c[~c.estimator.str.contains("var diff")].set_index(["universe", "window", "estimator"])
    diff = c[c.estimator.str.contains("var diff")].copy()
    diff["estimator"] = diff["estimator"].str.replace(" - sample (var diff)", "", regex=False)
    diff = diff.set_index(["universe", "window", "estimator"])["sharpe"].rename("t vs sample")
    tab = base[["volatility", "turnover"]].join(diff).reset_index().set_index("universe")
    tab.index.name = "Universe"
    md_table(tab, {"volatility": "{:.2%}", "turnover": "{:.2f}", "t vs sample": "{:.2f}", "window": "{:.0f}"}, "e3_cov")
    for f, name in (("E3/stats_means.csv", "e3_means"), ("E3/stats_windows.csv", "e3_windows"), ("E6/stats_base.csv", "e6"),
                    ("E7/stats_multi_asset.csv", "e7_multi"), ("E7/stats_sleeves.csv", "e7_sleeves"), ("E8/stats_base.csv", "e8")):
        s = pd.read_csv(RES / f, index_col=0)
        tab = s[["ann_return", "volatility", "sharpe", "max_drawdown", "turnover"]]
        tab.index.name = "Strategy"
        md_table(tab, {"ann_return": "{:.1%}", "volatility": "{:.1%}", "sharpe": "{:.2f}", "max_drawdown": "{:.0%}",
                       "turnover": "{:.2f}"}, name)
    t6 = pd.read_csv(RES / "E6/tests.csv", index_col=0)[["diff", "ci_low", "ci_high", "p_boot_holm"]]
    t6.index.name = "Method"
    md_table(t6, {"diff": "{:+.2f}", "ci_low": "{:+.2f}", "ci_high": "{:+.2f}", "p_boot_holm": "{:.2f}"}, "e6_tests")
    cap = pd.read_csv(RES / "E9/capacity.csv")
    cap["aum"] = cap["aum"].map(lambda v: f"${v / 1e9:g}bn")
    cap = cap.set_index("aum")
    cap.index.name = "Capital"
    md_table(cap, {"sharpe": "{:.2f}", "cost_drag": "{:.2%}", "turnover": "{:.2f}"}, "e9")
    vb = json.loads((RES / "E10/var_backtest.json").read_text())
    cond = json.loads((RES / "E10/var_backtest_conditional.json").read_text())
    rows = []
    for k, v in list(vb.items()) + list(cond.items()):
        if k == "sample":
            continue
        rows.append({"Model": k, "Exceptions": v["kupiec"]["exceptions"], "Expected": v["kupiec"]["expected"],
                     "Kupiec p": v["kupiec"]["p_value"], "Clustering p": v["independence"]["p_value"]})
    tab = pd.DataFrame(rows).set_index("Model")
    md_table(tab, {"Exceptions": "{:.0f}", "Expected": "{:.0f}", "Kupiec p": "{:.3f}", "Clustering p": "{:.3f}"}, "e10_var")


def main():
    e1()
    e2()
    e4()
    e5()
    others()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))
    print("tables:", sorted(p.name for p in TAB.glob("*.md")))


if __name__ == "__main__":
    main()
