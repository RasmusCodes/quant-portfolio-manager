"""Why the optimizer chose each weight.

For any solution of a mean-variance-type problem (min 1/2 w'(lambda Cov)w - mu'w subject to linear
constraints) the KKT conditions give

    lambda Cov w = mu - A'y = mu_eff                                         (1)

so the weights are an unconstrained mean-variance solution for "effective" expected returns: the inputs
minus the shadow costs of the binding constraints (a position cap lowers that asset's effective return,
the budget subtracts a common hurdle, a sector cap lowers every member's).

Inverting (1) asset by asset (Stevens 1998, "On the inverse of the covariance matrix in portfolio
analysis", J. Finance): regress asset i on all the others, r_i = c + b_i'r_(-i) + e_i, with residual
variance s_i^2 = 1 / (Cov^-1)_ii.  Then

    w_i = (mu_eff_i - b_i'mu_eff_(-i)) / (lambda s_i^2)                      (2)

The optimizer holds an asset in proportion to the part of its effective return that its "hedge
portfolio" of other assets cannot replicate, divided by the risk that cannot be hedged away.  That is the
explanation the demo draws, and it is exact (tests check (2) reproduces the weights).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .methods import Portfolio, risk_contributions


def hedge_decomposition(mu_eff, cov, names=None, top: int = 3) -> pd.DataFrame:
    mu_eff = np.asarray(mu_eff, dtype=float)
    cov = np.asarray(cov, dtype=float)
    n = len(mu_eff)
    names = list(names) if names is not None else [f"a{i}" for i in range(n)]
    theta = np.linalg.inv(cov)
    rows = []
    for i in range(n):
        s2 = 1.0 / theta[i, i]
        others = [j for j in range(n) if j != i]
        beta = -theta[i, others] * s2
        hedge = float(beta @ mu_eff[others])
        order = np.argsort(-np.abs(beta))[:top]
        rows.append({
            "asset": names[i],
            "effective_return": mu_eff[i],
            "hedge_return": hedge,
            "unique_return": mu_eff[i] - hedge,
            "residual_vol": float(np.sqrt(s2)),
            "total_vol": float(np.sqrt(cov[i, i])),
            "r2": float(1.0 - s2 / cov[i, i]),
            "top_hedges": ", ".join(f"{names[others[k]]} {beta[k]:+.2f}" for k in order),
            "precision_term": (mu_eff[i] - hedge) / s2,
        })
    return pd.DataFrame(rows).set_index("asset")


def constraint_attribution(portfolio: Portfolio) -> pd.DataFrame:
    """Columns: objective's linear term, then one column per constraint family, then mu_eff (their sum)."""
    problem = portfolio.info.get("problem")
    sol = portfolio.info.get("solution")
    if problem is None or sol is None:
        raise ValueError("constraint attribution needs a QP-based portfolio")
    n = problem.n_assets
    names = problem.assets
    cols: dict[str, np.ndarray] = {"input_return": -problem.q[:n].copy()}
    for k, label in enumerate(problem.labels):
        y = sol.y[k]
        if abs(y) < 1e-12:
            continue
        row = problem.A[k, :n]
        if not np.any(row):
            continue
        family = label.split(":")[0]
        family = {"bound": "position limits", "budget": "budget", "net": "net exposure", "trade+": "trading",
                  "trade-": "trading", "gross+": "gross exposure", "gross-": "gross exposure"}.get(family, label)
        cols.setdefault(family, np.zeros(n))
        cols[family] = cols[family] - row * y
    frame = pd.DataFrame(cols, index=names)
    frame["effective_return"] = frame.sum(axis=1)
    return frame


def implied_returns(weights, cov, risk_aversion: float = 2.5) -> np.ndarray:
    """Reverse optimization: the expected excess returns that make these weights mean-variance optimal."""
    return risk_aversion * np.asarray(cov, dtype=float) @ np.asarray(weights, dtype=float)


def explain(portfolio: Portfolio, cov, mu=None, labels: dict | None = None) -> dict:
    """Per-asset table plus one sentence per asset, for reports and the demo."""
    names = list(portfolio.weights.index)
    cov_a = cov.loc[names, names].values if isinstance(cov, pd.DataFrame) else np.asarray(cov, dtype=float)
    w = portfolio.weights.values
    table = pd.DataFrame({"weight": w, "risk_share": risk_contributions(w, cov_a)}, index=names)
    lam = portfolio.info.get("lambda")
    out: dict = {"method": portfolio.method, "table": table, "sentences": {}}
    labels = labels or {}
    if "problem" in portfolio.info and portfolio.ok:
        attr = constraint_attribution(portfolio)
        hedge = hedge_decomposition(attr["effective_return"].values, cov_a, names)
        table = table.join(attr).join(hedge.drop(columns=["effective_return"]))
        table["weight_from_formula"] = table["precision_term"] / lam
        out["table"] = table
        binding = {b["constraint"]: b for b in portfolio.binding}
        # the asset's own bound multiplier enters only its own effective return, so the hedged return
        # it would have without that bound is unique_return + y (y > 0 at a cap, y < 0 at a floor)
        sol, problem = portfolio.info["solution"], portfolio.info["problem"]
        own = {lab.split(":", 1)[1]: float(sol.y[k]) for k, lab in enumerate(problem.labels) if lab.startswith("bound:")}
        table["unique_return_unbounded"] = table["unique_return"] + pd.Series(own).reindex(names).fillna(0.0)
        for a in names:
            r = table.loc[a]
            name = labels.get(a, a)
            cap = binding.get(f"bound:{a}")
            if cap is not None and cap["side"] == "upper":
                why = (f"at its {cap['bound']:.0%} cap. Net of what the other holdings already deliver it would "
                       f"earn {r.unique_return_unbounded:+.2%} a year, so the optimizer wants more; the cap costs "
                       f"{cap['shadow']:.2%} of expected return at the margin")
            elif cap is not None:
                floor = "zero" if abs(cap["bound"]) < 1e-12 else f"its {cap['bound']:.0%} floor"
                why = (f"held at {floor}: net of what the other holdings already deliver its expected return is "
                       f"{r.unique_return_unbounded:+.2%} a year, so the optimizer would go lower if allowed")
            elif abs(r.weight) < 1e-9:
                why = "no position"
            else:
                why = (f"its return not explained by the other assets is {r.unique_return:+.2%} a year, against "
                       f"{r.residual_vol:.1%} of volatility the others can't hedge (R^2 {r.r2:.0%})")
            out["sentences"][a] = f"{name}: {r.weight:.1%}, {why}."
        if "budget" in attr.columns:
            out["budget_hurdle"] = float(attr["budget"].iloc[0])
    elif portfolio.method == "erc":
        for a in names:
            out["sentences"][a] = (f"{labels.get(a, a)}: {table.loc[a, 'weight']:.1%}, sized so that it contributes "
                                   f"{table.loc[a, 'risk_share']:.1%} of portfolio risk, like every other asset.")
    elif portfolio.method == "hrp":
        order = portfolio.info.get("order_names", names)
        out["cluster_order"] = order
        for a in names:
            out["sentences"][a] = (f"{labels.get(a, a)}: {table.loc[a, 'weight']:.1%}, from splitting capital between "
                                   f"clusters of similar assets in inverse proportion to cluster variance.")
    else:
        for a in names:
            out["sentences"][a] = f"{labels.get(a, a)}: {table.loc[a, 'weight']:.1%} ({portfolio.method})."
    out["implied_returns"] = pd.Series(implied_returns(w, cov_a, lam or 2.5), index=names)
    return out
