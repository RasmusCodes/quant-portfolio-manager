"""Portfolio construction methods.

Conventions: ``mu`` are expected EXCESS returns over cash and ``cov`` the covariance, both annualised.
Weights are fractions of capital; cash is 1 - sum(w) and earns the risk-free rate, so a portfolio's
expected excess return is w'mu and its Sharpe ratio w'mu / sqrt(w'Cov w).

Methods (key -> function):
  equal_weight, market_cap, inverse_vol (naive risk parity), min_variance, mean_variance, max_sharpe,
  erc (equal risk contribution / risk budgeting), hrp (hierarchical risk parity), black_litterman,
  kelly (full or fractional), vol_target (overlay), factor_neutral, constrained.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd

from .constraints import Constraints, QPProblem, binding_constraints, build_qp, check_feasibility
from .qp import INF, QPSolution, solve_qp


@dataclass
class Portfolio:
    weights: pd.Series
    method: str
    status: str = "solved"
    expected_return: float | None = None
    volatility: float | None = None
    sharpe: float | None = None
    risk_contrib: pd.Series | None = None
    binding: list = field(default_factory=list)
    mu_eff: pd.Series | None = None
    info: dict = field(default_factory=dict)
    messages: list = field(default_factory=list)

    @property
    def cash(self) -> float:
        return float(1.0 - self.weights.sum())

    @property
    def gross(self) -> float:
        return float(self.weights.abs().sum())

    @property
    def ok(self) -> bool:
        return self.status in ("solved", "solved_inaccurate", "closed_form")

    def summary(self) -> dict:
        return {"method": self.method, "status": self.status, "expected_return": self.expected_return,
                "volatility": self.volatility, "sharpe": self.sharpe, "gross": self.gross, "cash": self.cash}


# --------------------------------------------------------------------------------------------- helpers

def _prep(mu, cov, assets):
    if isinstance(cov, pd.DataFrame):
        names = list(cov.columns)
        cov_a = cov.values.astype(float)
    else:
        cov_a = np.asarray(cov, dtype=float)
        names = list(assets) if assets is not None else [f"a{i}" for i in range(cov_a.shape[0])]
    if mu is None:
        mu_a = None
    elif isinstance(mu, pd.Series):
        mu_a = mu.reindex(names).values.astype(float)
    else:
        mu_a = np.asarray(mu, dtype=float)
    return mu_a, cov_a, names


def risk_contributions(w, cov) -> np.ndarray:
    """Share of portfolio variance from each position: w_i (Cov w)_i / w'Cov w (sums to 1)."""
    w = np.asarray(w, dtype=float)
    cov = np.asarray(cov, dtype=float)
    var = float(w @ cov @ w)
    if var <= 0:
        return np.zeros_like(w)
    return w * (cov @ w) / var


def _finish(w, mu, cov, names, method, status="closed_form", **extra) -> Portfolio:
    w = np.asarray(w, dtype=float)
    vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
    er = None if mu is None else float(w @ mu)
    sharpe = None if (er is None or vol <= 0) else er / vol
    return Portfolio(pd.Series(w, index=names), method, status, er, vol, sharpe,
                     pd.Series(risk_contributions(w, cov), index=names), **extra)


def _solve(problem: QPProblem, x0=None) -> QPSolution:
    return problem.solve(x0=x0)


def _effective_mu(problem: QPProblem, sol: QPSolution, names) -> pd.Series:
    """-q_w - A_w'y: by stationarity P_w w equals this vector, so with P_w = lambda Cov these are the
    expected returns the solution is unconstrained mean-variance optimal for, after constraint costs."""
    n = problem.n_assets
    return pd.Series(-problem.q[:n] - problem.A[:, :n].T @ sol.y, index=names)


def _qp_portfolio(P_w, q_w, mu, cov, names, cons, method, lam, x0=None, info=None) -> Portfolio:
    msgs = check_feasibility(cons, names)
    problem = build_qp(P_w, q_w, cons, names)
    sol = _solve(problem, x0)
    if not sol.ok:
        w = np.full(len(names), np.nan)
        p = Portfolio(pd.Series(w, index=names), method, sol.status, messages=msgs or [f"solver: {sol.status}"])
        p.info = {"problem": problem, "solution": sol, **(info or {})}
        return p
    w = sol.x[:len(names)]
    w = np.where(np.abs(w) < 1e-12, 0.0, w)
    p = _finish(w, mu, cov, names, method, sol.status)
    p.binding = binding_constraints(problem, sol)
    p.mu_eff = _effective_mu(problem, sol, names)
    p.info = {"problem": problem, "solution": sol, "lambda": lam, "iterations": sol.iterations,
              "polished": sol.polished, "mu_objective": -np.asarray(q_w, float), **(info or {})}
    p.messages = msgs
    return p


# --------------------------------------------------------------------------------------------- methods

def equal_weight(cov, assets=None, mu=None) -> Portfolio:
    mu_a, cov_a, names = _prep(mu, cov, assets)
    n = len(names)
    return _finish(np.full(n, 1.0 / n), mu_a, cov_a, names, "equal_weight")


def market_cap(cov, market_caps, assets=None, mu=None) -> Portfolio:
    mu_a, cov_a, names = _prep(mu, cov, assets)
    caps = market_caps.reindex(names).values if isinstance(market_caps, pd.Series) else np.asarray(market_caps, float)
    caps = np.where(np.isfinite(caps) & (caps > 0), caps, 0.0)
    if caps.sum() <= 0:
        raise ValueError("market caps are all missing")
    return _finish(caps / caps.sum(), mu_a, cov_a, names, "market_cap")


def inverse_vol(cov, assets=None, mu=None) -> Portfolio:
    """Naive risk parity: weights proportional to 1 / volatility (ignores correlations)."""
    mu_a, cov_a, names = _prep(mu, cov, assets)
    inv = 1.0 / np.sqrt(np.diag(cov_a))
    return _finish(inv / inv.sum(), mu_a, cov_a, names, "inverse_vol")


def min_variance(cov, cons: Constraints | None = None, assets=None, mu=None, x0=None) -> Portfolio:
    mu_a, cov_a, names = _prep(mu, cov, assets)
    cons = cons or Constraints()
    return _qp_portfolio(cov_a, np.zeros(len(names)), mu_a, cov_a, names, cons, "min_variance", 1.0, x0)


def mean_variance(mu, cov, risk_aversion: float = 4.0, cons: Constraints | None = None, assets=None,
                  x0=None) -> Portfolio:
    """max mu'w - (risk_aversion / 2) w'Cov w  subject to ``cons``.

    With ``cons.max_vol`` set, risk aversion is raised (bisection) until volatility is within the cap.
    """
    mu_a, cov_a, names = _prep(mu, cov, assets)
    cons = cons or Constraints()
    lam = float(risk_aversion)
    p = _qp_portfolio(lam * cov_a, -mu_a, mu_a, cov_a, names, cons, "mean_variance", lam, x0)
    if cons.max_vol is None or not p.ok or p.volatility <= cons.max_vol * (1 + 1e-9):
        return p
    lo, hi = lam, lam
    best = None
    for _ in range(60):  # bracket
        hi *= 2.0
        q = _qp_portfolio(hi * cov_a, -mu_a, mu_a, cov_a, names, cons, "mean_variance", hi, p.weights.values)
        if not q.ok:
            break
        if q.volatility <= cons.max_vol:
            best = q
            break
        lo = hi
    if best is None:
        mv = min_variance(cov_a, cons, names, mu_a)
        mv.method = "mean_variance"
        mv.status = "vol_cap_infeasible" if mv.ok and mv.volatility > cons.max_vol else mv.status
        mv.messages.append(f"even the minimum-variance portfolio has volatility {mv.volatility:.2%}, above the "
                           f"{cons.max_vol:.2%} cap")
        return mv
    for _ in range(50):  # bisection on log lambda
        mid = float(np.sqrt(lo * hi))
        q = _qp_portfolio(mid * cov_a, -mu_a, mu_a, cov_a, names, cons, "mean_variance", mid, best.weights.values)
        if q.ok and q.volatility <= cons.max_vol:
            hi, best = mid, q
        else:
            lo = mid
        if hi / lo < 1 + 1e-9:
            break
    best.info["vol_cap_binding"] = True
    best.messages.append(f"risk aversion raised from {lam:g} to {best.info['lambda']:.3g} to meet the "
                         f"{cons.max_vol:.1%} volatility cap")
    return best


def _max_sharpe_homogenized(mu_a, cov_a, names, cons: Constraints) -> Portfolio | None:
    """Exact max Sharpe as ONE convex QP (Cornuejols and Tutuncu 2007, sec. 8.2): with y = k w, k > 0,

        min y'Cov y  s.t.  mu'y = 1,  every constraint row  l <= a x <= u  rewritten as  a x' - l k >= 0,
        a x' - u k <= 0  (x' = k x includes the auxiliary variables),  k >= 0;   then w = y / k.

    Works for any linear constraints.  Returns None when no feasible portfolio has mu'w > 0 or the solve
    fails, so the caller can fall back to the search."""
    base = build_qp(cov_a, np.zeros(len(names)), cons, names)
    n, nv = base.n_assets, base.P.shape[0]
    rows, lo, hi, labels = [], [], [], []
    for k in range(base.A.shape[0]):
        a, l, u = base.A[k], base.l[k], base.u[k]
        if l <= -INF / 2 and u >= INF / 2:
            continue
        if abs(u - l) < 1e-12:
            rows.append(np.append(a, -l)); lo.append(0.0); hi.append(0.0); labels.append(base.labels[k])
            continue
        if l > -INF / 2:
            rows.append(np.append(a, -l)); lo.append(0.0); hi.append(INF); labels.append(base.labels[k] + ":lo")
        if u < INF / 2:
            rows.append(np.append(a, -u)); lo.append(-INF); hi.append(0.0); labels.append(base.labels[k] + ":hi")
    r = np.zeros(nv + 1)
    r[:n] = mu_a
    rows.append(r); lo.append(1.0); hi.append(1.0); labels.append("return=1")
    r = np.zeros(nv + 1)
    r[-1] = 1.0
    rows.append(r); lo.append(0.0); hi.append(INF); labels.append("kappa>=0")
    P = np.zeros((nv + 1, nv + 1))
    P[:nv, :nv] = base.P
    P[nv, nv] = 1e-12
    sol = solve_qp(P, np.zeros(nv + 1), np.vstack(rows), np.array(lo), np.array(hi))
    if not sol.ok or sol.x[-1] <= 1e-12:
        return None
    w = sol.x[:n] / sol.x[-1]
    w = np.where(np.abs(w) < 1e-12, 0.0, w)
    # re-solve as mean-variance at the implied risk aversion to get named multipliers for the explanation
    vol2 = float(w @ cov_a @ w)
    lam = float(w @ mu_a) / vol2 if vol2 > 0 else 1.0
    p = _qp_portfolio(lam * cov_a, -mu_a, mu_a, cov_a, names, cons, "max_sharpe", lam, x0=None)
    if not p.ok or np.max(np.abs(p.weights.values - w)) > 1e-6:
        p = _finish(w, mu_a, cov_a, names, "max_sharpe", "solved")
    p.info["homogenized"] = True
    return p


def max_sharpe(mu, cov, cons: Constraints | None = None, assets=None) -> Portfolio:
    """Highest w'mu / vol under ``cons``: one homogenized QP; falls back to a golden-section search over
    risk aversion on the constrained frontier (needed only with linear trading costs in the objective)."""
    mu_a, cov_a, names = _prep(mu, cov, assets)
    cons = cons or Constraints()
    if cons.budget is None:
        raise ValueError("max Sharpe needs a budget (the ratio is scale-free otherwise)")
    if cons.trade_cost is None:
        p = _max_sharpe_homogenized(mu_a, cov_a, names, cons)
        if p is not None:
            return p
    cache: dict[float, Portfolio] = {}

    def at(log_lam: float) -> Portfolio:
        key = round(log_lam, 12)
        if key not in cache:
            lam = float(np.exp(log_lam))
            cache[key] = _qp_portfolio(lam * cov_a, -mu_a, mu_a, cov_a, names, cons, "max_sharpe", lam)
        return cache[key]

    def score(log_lam: float) -> float:
        p = at(log_lam)
        return -np.inf if not p.ok or p.sharpe is None else p.sharpe

    a, b = np.log(1e-3), np.log(1e5)
    g = (np.sqrt(5) - 1) / 2
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = score(c), score(d)
    for _ in range(80):
        if fc >= fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = score(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = score(d)
        if b - a < 1e-7:
            break
    best = max(cache.values(), key=lambda p: -np.inf if not p.ok or p.sharpe is None else p.sharpe)
    best.method = "max_sharpe"
    if best.sharpe is not None and best.sharpe <= 0:
        best.messages.append("no portfolio has a positive expected excess return under these inputs")
    return best


def erc_weights(cov, budgets=None, tol: float = 1e-12, max_iter: int = 10000) -> np.ndarray:
    """Risk budgeting by cyclical coordinate descent on  1/2 y'Cov y - sum b_i log y_i  (Spinu 2013;
    Griveau-Billion, Richard and Roncalli 2013); w = y / sum(y) has risk contributions proportional to b."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float) / np.sum(budgets)
    diag = np.diag(cov)
    y = 1.0 / np.sqrt(diag)
    y /= np.sqrt(y @ cov @ y)
    for _ in range(max_iter):
        for i in range(n):
            c = cov[i] @ y - diag[i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4.0 * diag[i] * b[i])) / (2.0 * diag[i])
        rc = y * (cov @ y)
        if np.max(np.abs(rc / rc.sum() - b)) < tol:
            break
    return y / y.sum()


def erc(cov, budgets=None, assets=None, mu=None) -> Portfolio:
    """Equal risk contribution (or risk budgeting with ``budgets``): long-only, fully invested."""
    mu_a, cov_a, names = _prep(mu, cov, assets)
    if isinstance(budgets, pd.Series):
        budgets = budgets.reindex(names).values
    w = erc_weights(cov_a, budgets)
    p = _finish(w, mu_a, cov_a, names, "erc")
    p.info = {"budgets": None if budgets is None else list(np.asarray(budgets, float) / np.sum(budgets))}
    return p


# --- hierarchical risk parity -----------------------------------------------------------------------

def linkage_matrix(dist: np.ndarray, method: str = "single") -> np.ndarray:
    """Agglomerative clustering (Lance-Williams updates). Rows: [a, b, distance, size], a < b, scipy layout."""
    d = np.asarray(dist, dtype=float)
    n = d.shape[0]
    big = 2 * n - 1
    M = np.full((big, big), np.inf)
    M[:n, :n] = d
    np.fill_diagonal(M, np.inf)
    size = np.zeros(big)
    size[:n] = 1
    active = list(range(n))
    Z = []
    for k in range(n - 1):
        ids = np.array(sorted(active))
        sub = M[np.ix_(ids, ids)]
        i, j = np.unravel_index(np.argmin(sub), sub.shape)
        a, b = sorted((int(ids[i]), int(ids[j])))
        dist_ab = float(sub[i, j])
        new = n + k
        size[new] = size[a] + size[b]
        Z.append([a, b, dist_ab, size[new]])
        for c in active:
            if c in (a, b):
                continue
            if method == "single":
                v = min(M[a, c], M[b, c])
            elif method == "complete":
                v = max(M[a, c], M[b, c])
            elif method == "average":
                v = (size[a] * M[a, c] + size[b] * M[b, c]) / (size[a] + size[b])
            else:
                raise KeyError(f"unknown linkage {method!r}")
            M[new, c] = M[c, new] = v
        active = [c for c in active if c not in (a, b)] + [new]
    return np.array(Z)


def quasi_diagonal_order(Z: np.ndarray) -> list[int]:
    n = Z.shape[0] + 1
    order = [int(Z[-1, 0]), int(Z[-1, 1])]
    while any(i >= n for i in order):
        expanded = []
        for i in order:
            if i >= n:
                expanded += [int(Z[i - n, 0]), int(Z[i - n, 1])]
            else:
                expanded.append(i)
        order = expanded
    return order


def _cluster_var(cov, idx) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hrp_weights(cov, linkage: str = "single", distance_of_distance: bool = True) -> tuple[np.ndarray, dict]:
    """Lopez de Prado (2016) hierarchical risk parity.

    Distance d_ij = sqrt((1 - rho_ij) / 2).  Like the paper's code, clustering runs on the Euclidean
    distance between columns of d ("distance of distances") unless ``distance_of_distance=False``.
    Recursive bisection splits the quasi-diagonal order in halves and allocates between halves in
    inverse proportion to their inverse-variance-portfolio variances.
    """
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(sd, sd), -1.0, 1.0)
    d = np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, None))
    if distance_of_distance:
        diff = d[:, None, :] - d[None, :, :]
        d = np.sqrt(np.sum(diff ** 2, axis=2))
    Z = linkage_matrix(d, linkage)
    order = quasi_diagonal_order(Z)
    w = np.ones(len(order))
    splits = []
    clusters = [order]
    while clusters:
        clusters = [c[s:e] for c in clusters for s, e in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v_l, v_r = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1.0 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            splits.append({"left": list(left), "right": list(right), "var_left": v_l, "var_right": v_r,
                           "alpha": alpha})
    return w, {"linkage": Z, "order": order, "splits": splits}


def hrp(cov, assets=None, mu=None, linkage: str = "single", distance_of_distance: bool = True) -> Portfolio:
    mu_a, cov_a, names = _prep(mu, cov, assets)
    w, info = hrp_weights(cov_a, linkage, distance_of_distance)
    p = _finish(w, mu_a, cov_a, names, "hrp")
    info["order_names"] = [names[i] for i in info["order"]]
    p.info = info
    return p


# --- Black-Litterman --------------------------------------------------------------------------------

def bl_posterior(cov, market_weights, P=None, Q=None, confidences=None, omega=None, tau: float = 0.05,
                 risk_aversion: float = 2.5) -> dict:
    """Black-Litterman posterior in the He-Litterman (1999) form.

    Prior: pi = risk_aversion x Cov x w_mkt.  Views: P mu = Q + e, e ~ N(0, Omega).  Without an explicit
    Omega, a view with confidence c in (0, 1] gets Omega_kk = (1 - c) / c x (P tau Cov P')_kk, so c = 0.5
    is He-Litterman's default and c = 1 imposes the view exactly.
    Returns pi, posterior mean mu, posterior covariance (Cov + uncertainty of the mean), Omega.
    """
    cov = np.asarray(cov, dtype=float)
    w = np.asarray(market_weights, dtype=float)
    w = w / w.sum()
    pi = risk_aversion * cov @ w
    ts = tau * cov
    if P is None or Q is None or len(np.atleast_1d(Q)) == 0:
        return {"pi": pi, "mu": pi.copy(), "cov": cov + ts, "omega": None, "P": None, "Q": None}
    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    base = P @ ts @ P.T
    if omega is None:
        if confidences is None:
            omega = np.diag(np.diag(base))
        else:
            c = np.clip(np.atleast_1d(np.asarray(confidences, dtype=float)), 1e-6, 1.0)
            omega = np.diag((1.0 - c) / c * np.diag(base))
    omega = np.atleast_2d(np.asarray(omega, dtype=float))
    gain = ts @ P.T @ np.linalg.pinv(base + omega)
    mu = pi + gain @ (Q - P @ pi)
    post = ts - gain @ P @ ts
    return {"pi": pi, "mu": mu, "cov": cov + post, "omega": omega, "P": P, "Q": Q,
            "prior_view": P @ pi, "posterior_view": P @ mu}


def black_litterman(cov, market_weights, P=None, Q=None, confidences=None, omega=None, tau: float = 0.05,
                    risk_aversion: float = 2.5, cons: Constraints | None = None, assets=None) -> Portfolio:
    _, cov_a, names = _prep(None, cov, assets)
    if isinstance(market_weights, pd.Series):
        market_weights = market_weights.reindex(names).fillna(0.0).values
    post = bl_posterior(cov_a, market_weights, P, Q, confidences, omega, tau, risk_aversion)
    cons = cons or Constraints()
    p = mean_variance(post["mu"], post["cov"], risk_aversion, cons, names)
    p.method = "black_litterman"
    # report risk under the plain covariance, expected return under the posterior
    p.volatility = float(np.sqrt(p.weights.values @ cov_a @ p.weights.values)) if p.ok else None
    if p.ok:
        p.sharpe = p.expected_return / p.volatility if p.volatility else None
        p.risk_contrib = pd.Series(risk_contributions(p.weights.values, cov_a), index=names)
    p.info.update({"bl": post, "market_weights": np.asarray(market_weights, float) / np.sum(market_weights)})
    return p


# --- Kelly, volatility targeting, factor neutrality ---------------------------------------------------

def kelly(mu, cov, fraction: float = 1.0, cons: Constraints | None = None, assets=None) -> Portfolio:
    """Growth-optimal (Kelly) weights under the Gaussian approximation E log(1+r) ~ w'mu - w'Cov w / 2.

    Full Kelly is mean-variance with risk aversion 1; fraction f scales it to risk aversion 1/f.  The
    budget is free: what is not invested sits in cash (negative cash = borrowing at the risk-free rate).
    Default constraints: long-only with total exposure at most 2x capital.
    """
    if fraction <= 0:
        raise ValueError("fraction must be positive")
    cons = cons or Constraints(lower=0.0, budget=None, net_min=0.0, net_max=2.0)
    if cons.budget is not None:
        cons = cons.with_(budget=None, net_min=0.0 if cons.net_min is None else cons.net_min)
    p = mean_variance(mu, cov, 1.0 / fraction, cons, assets)
    p.method = "kelly" if fraction == 1 else f"kelly_{fraction:g}"
    p.info["fraction"] = fraction
    if p.ok and p.expected_return is not None:
        p.info["expected_log_growth"] = p.expected_return - 0.5 * p.volatility ** 2
    return p


def vol_target(portfolio: Portfolio | pd.Series, cov, target_vol: float, max_leverage: float | None = 1.0,
               mu=None, assets=None) -> Portfolio:
    """Scale a portfolio to ``target_vol`` (cash makes up the rest); gross exposure stays <= ``max_leverage``."""
    w = portfolio.weights if isinstance(portfolio, Portfolio) else portfolio
    mu_a, cov_a, names = _prep(mu, cov, list(w.index) if assets is None else assets)
    wv = w.reindex(names).fillna(0.0).values
    vol = float(np.sqrt(wv @ cov_a @ wv))
    scale = target_vol / vol if vol > 0 else 0.0
    gross = np.abs(wv).sum()
    if max_leverage is not None and gross > 0:
        scale = min(scale, max_leverage / gross)
    p = _finish(scale * wv, mu_a, cov_a, names, "vol_target")
    if mu_a is None and isinstance(portfolio, Portfolio) and portfolio.expected_return is not None:
        p.expected_return = scale * portfolio.expected_return
        p.sharpe = portfolio.sharpe
    p.info = {"scale": scale, "base_method": getattr(portfolio, "method", "weights"), "target_vol": target_vol}
    return p


def factor_neutral(mu, cov, betas, factor_names: Sequence[str], neutral: Sequence[str] | None = None,
                   risk_aversion: float = 4.0, cons: Constraints | None = None, assets=None) -> Portfolio:
    """Mean-variance with zero exposure to the chosen factors.

    Default: dollar-neutral long-short (budget 0), positions within +/-25%, gross exposure <= 2.
    """
    _, cov_a, names = _prep(None, cov, assets)
    neutral = list(neutral if neutral is not None else factor_names)
    cons = cons or Constraints(lower=-0.25, upper=0.25, budget=0.0, gross_max=2.0)
    fmin = {f: 0.0 for f in neutral}
    cons = cons.with_(factor_betas=np.asarray(betas, float), factor_names=list(factor_names),
                      factor_min=fmin, factor_max=dict(fmin))
    p = mean_variance(mu, cov_a, risk_aversion, cons, names)
    p.method = "factor_neutral"
    if p.ok:
        p.info["exposures"] = dict(zip(factor_names, np.asarray(betas, float).T @ p.weights.values))
    return p


def constrained(mu, cov, cons: Constraints, risk_aversion: float = 4.0, objective: str = "mean_variance",
                assets=None) -> Portfolio:
    """Fully constrained optimization: any mix of the constraint types in ``Constraints``."""
    if objective == "min_variance":
        p = min_variance(cov, cons, assets, mu)
    else:
        p = mean_variance(mu, cov, risk_aversion, cons, assets)
    p.method = "constrained"
    return p


# ------------------------------------------------------------------------------------------- drawdown

def max_drawdown(returns) -> float:
    wealth = np.cumprod(1.0 + np.asarray(returns, dtype=float))
    peak = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
    return float(np.max(1.0 - wealth / peak)) if wealth.size else 0.0


def drawdown_cap(weights: pd.Series, history: pd.DataFrame, max_dd: float, rf: pd.Series | None = None) -> tuple[pd.Series, float]:
    """Scale exposure so the backfilled (in-sample) maximum drawdown is at most ``max_dd``.

    A heuristic: it bounds the drawdown the weights WOULD have had over ``history``, not future losses.
    Returns (scaled weights, scale).
    """
    w = weights.reindex(history.columns).fillna(0.0)
    cash = rf.reindex(history.index).fillna(0.0).values if rf is not None else np.zeros(len(history))

    def dd(scale):
        port = history.values @ (scale * w.values) + (1 - scale * w.values.sum()) * cash
        return max_drawdown(port)

    if dd(1.0) <= max_dd:
        return weights, 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if dd(mid) <= max_dd:
            lo = mid
        else:
            hi = mid
    return weights * lo, lo


# ------------------------------------------------------------------------------------------ registry

METHODS = {
    "equal_weight": {"label": "Equal weight (1/N)", "needs_mu": False},
    "market_cap": {"label": "Market-cap weight", "needs_mu": False},
    "inverse_vol": {"label": "Risk parity (inverse volatility)", "needs_mu": False},
    "min_variance": {"label": "Minimum variance", "needs_mu": False},
    "erc": {"label": "Equal risk contribution", "needs_mu": False},
    "hrp": {"label": "Hierarchical risk parity", "needs_mu": False},
    "mean_variance": {"label": "Mean-variance", "needs_mu": True},
    "max_sharpe": {"label": "Maximum Sharpe", "needs_mu": True},
    "black_litterman": {"label": "Black-Litterman", "needs_mu": False},
    "kelly": {"label": "Kelly (full)", "needs_mu": True},
    "kelly_half": {"label": "Kelly (half)", "needs_mu": True},
    "factor_neutral": {"label": "Factor-neutral mean-variance", "needs_mu": True},
    "constrained": {"label": "Constrained mean-variance", "needs_mu": True},
}


def optimize(method: str, mu=None, cov=None, cons: Constraints | None = None, assets=None, **kw) -> Portfolio:
    """Run a method by key; extra keywords go to the method (risk_aversion, market_weights, views, ...)."""
    if method == "equal_weight":
        return equal_weight(cov, assets, mu)
    if method == "market_cap":
        return market_cap(cov, kw["market_weights"], assets, mu)
    if method == "inverse_vol":
        return inverse_vol(cov, assets, mu)
    if method == "min_variance":
        return min_variance(cov, cons, assets, mu, kw.get("x0"))
    if method == "erc":
        return erc(cov, kw.get("budgets"), assets, mu)
    if method == "hrp":
        return hrp(cov, assets, mu, kw.get("linkage", "single"))
    if method == "mean_variance":
        return mean_variance(mu, cov, kw.get("risk_aversion", 4.0), cons, assets, kw.get("x0"))
    if method == "max_sharpe":
        return max_sharpe(mu, cov, cons, assets)
    if method == "black_litterman":
        return black_litterman(cov, kw["market_weights"], kw.get("P"), kw.get("Q"), kw.get("confidences"),
                               kw.get("omega"), kw.get("tau", 0.05), kw.get("risk_aversion", 2.5), cons, assets)
    if method == "kelly":
        return kelly(mu, cov, 1.0, cons, assets)
    if method == "kelly_half":
        return kelly(mu, cov, 0.5, cons, assets)
    if method == "factor_neutral":
        return factor_neutral(mu, cov, kw["betas"], kw["factor_names"], kw.get("neutral"),
                              kw.get("risk_aversion", 4.0), cons, assets)
    if method == "constrained":
        return constrained(mu, cov, cons or Constraints(), kw.get("risk_aversion", 4.0),
                           kw.get("objective", "mean_variance"), assets)
    raise KeyError(f"unknown method {method!r}; choose from {sorted(METHODS)}")
