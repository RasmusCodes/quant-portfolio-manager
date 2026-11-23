"""Efficient frontiers: constrained (numerical), unconstrained (closed form), and the 'realised'
frontier of portfolios built from estimated inputs but evaluated under the true ones (estimation error)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .constraints import Constraints, build_qp
from .methods import _prep, min_variance
from .qp import INF


def efficient_frontier(mu, cov, cons: Constraints | None = None, n_points: int = 30, assets=None) -> pd.DataFrame:
    """Minimum-variance portfolios for evenly spaced return targets between the minimum-variance
    portfolio's return and the highest attainable return under ``cons``."""
    mu_a, cov_a, names = _prep(mu, cov, assets)
    cons = cons or Constraints()
    mv = min_variance(cov_a, cons, names, mu_a)
    if not mv.ok:
        raise ValueError(f"constraints are infeasible ({mv.status})")
    top = build_qp(1e-6 * cov_a, -mu_a, cons, names).solve()
    r_min, r_max = float(mv.weights.values @ mu_a), float(top.x[:len(names)] @ mu_a)
    rows = []
    x0 = None
    for target in np.linspace(r_min, r_max, n_points):
        prob = build_qp(cov_a, np.zeros(len(names)), cons, names, extra_rows=[("target", mu_a, target, INF)])
        sol = prob.solve(x0=x0)
        if not sol.ok:
            continue
        x0 = sol.x
        w = sol.x[:len(names)]
        vol = float(np.sqrt(max(w @ cov_a @ w, 0.0)))
        er = float(w @ mu_a)
        rows.append({"expected_return": er, "volatility": vol, "sharpe": er / vol if vol > 0 else np.nan,
                     **{f"w_{a}": wi for a, wi in zip(names, w)}})
    return pd.DataFrame(rows)


def analytic_frontier(mu, cov, vols=None, n_points: int = 50) -> pd.DataFrame:
    """Unconstrained frontier with weights summing to one (Merton 1972): sigma^2 = (A r^2 - 2 B r + C) / D."""
    mu = np.asarray(mu, dtype=float)
    inv = np.linalg.inv(np.asarray(cov, dtype=float))
    ones = np.ones_like(mu)
    a, b, c = ones @ inv @ ones, ones @ inv @ mu, mu @ inv @ mu
    d = a * c - b * b
    r_mv = b / a
    span = np.sqrt(max(d / a, 1e-12))
    targets = np.linspace(r_mv - 0.2 * span, r_mv + 2.0 * span, n_points)
    var = (a * targets ** 2 - 2 * b * targets + c) / d
    return pd.DataFrame({"expected_return": targets, "volatility": np.sqrt(var)})


def realized_frontier(mu_est, cov_est, mu_true, cov_true, cons: Constraints | None = None, n_points: int = 20) -> pd.DataFrame:
    """Frontier built from estimated inputs, re-evaluated with the true inputs (Broadie 1993):
    the estimated frontier promises more than the same portfolios deliver."""
    est = efficient_frontier(mu_est, cov_est, cons, n_points)
    mu_t = np.asarray(mu_true, dtype=float)
    cov_t = np.asarray(cov_true, dtype=float)
    wcols = [c for c in est.columns if c.startswith("w_")]
    W = est[wcols].values
    est["true_return"] = W @ mu_t
    est["true_volatility"] = np.sqrt(np.einsum("ij,jk,ik->i", W, cov_t, W))
    return est
