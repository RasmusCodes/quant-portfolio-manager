"""Portfolio constraints and their translation into the QP form  l <= A x <= u.

Decision variables are the weights w (fractions of capital, cash = 1 - sum(w)).  Two kinds of auxiliary
variables are added only when needed:

* trade sizes t_i >= |w_i - w_prev_i| for a turnover limit or linear trading costs;
* absolute weights g_i >= |w_i| for a gross-exposure (leverage) limit when shorting is allowed.

Every row of A carries a label ("bound:NoDur", "budget", "group:Tech", "factor:Mkt-RF", "turnover",
...), so the solver's multipliers can be reported as shadow prices of named constraints.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Mapping, Sequence

import numpy as np

from .qp import INF, QPSolution, solve_qp


@dataclass
class Constraints:
    """Constraint set.  Scalars apply to every asset; sequences/dicts give per-asset or per-group values.

    lower, upper      position bounds (default long-only, no cap).  ``None`` means unbounded.
    budget            sum of weights fixed at this value (1 = fully invested; 0 = dollar neutral);
                      ``None`` leaves it free within [net_min, net_max].
    gross_max         sum |w| <= gross_max (leverage limit).
    groups            {group: [assets]} for sector / industry / asset-class limits, bounded by
                      group_min / group_max (scalar or {group: value}).
    factor_betas      n x k exposures with factor_names; factor_min / factor_max bound portfolio
                      exposure w'B (equal bounds = neutral to that factor).
    turnover_max      sum |w - w_prev| <= turnover_max (fraction of capital traded, both sides).
    trade_cost        linear cost per unit of weight traded (objective units), scalar or per asset.
    max_trade         per-asset cap on |w - w_prev| (e.g. from liquidity).
    max_vol           annualised volatility cap (mean-variance family: risk aversion is raised until
                      it holds).
    max_drawdown      cap on the in-sample (backfilled) maximum drawdown, applied by scaling exposure.
    """
    lower: float | Sequence[float] | Mapping[str, float] | None = 0.0
    upper: float | Sequence[float] | Mapping[str, float] | None = None
    budget: float | None = 1.0
    net_min: float | None = None
    net_max: float | None = None
    gross_max: float | None = None
    groups: Mapping[str, Sequence[str]] | None = None
    group_min: float | Mapping[str, float] | None = None
    group_max: float | Mapping[str, float] | None = None
    factor_betas: np.ndarray | None = None
    factor_names: Sequence[str] | None = None
    factor_min: float | Mapping[str, float] | None = None
    factor_max: float | Mapping[str, float] | None = None
    turnover_max: float | None = None
    w_prev: Sequence[float] | None = None
    trade_cost: float | Sequence[float] | None = None
    max_trade: float | Sequence[float] | None = None
    max_vol: float | None = None
    max_drawdown: float | None = None
    extra: dict = field(default_factory=dict)

    def with_(self, **kwargs) -> "Constraints":
        return replace(self, **kwargs)


def _vector(value, assets: Sequence[str], default: float) -> np.ndarray:
    n = len(assets)
    if value is None:
        return np.full(n, default)
    if isinstance(value, Mapping):
        return np.array([float(value.get(a, default)) for a in assets])
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr))
    if arr.size != n:
        raise ValueError(f"expected {n} values, got {arr.size}")
    return arr.astype(float)


def _group_value(value, name: str, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, Mapping):
        return float(value.get(name, default))
    return float(value)


def liquidity_caps(adv, aum: float, participation: float = 0.1, days: float = 5.0) -> np.ndarray:
    """Largest weight per asset that can be traded out within ``days`` at ``participation`` x ADV.

    ``adv`` is average daily traded value in the same currency as ``aum``.
    """
    adv = np.asarray(adv, dtype=float)
    return np.clip(participation * adv * days / float(aum), 0.0, None)


@dataclass
class QPProblem:
    P: np.ndarray
    q: np.ndarray
    A: np.ndarray
    l: np.ndarray
    u: np.ndarray
    labels: list[str]
    n_assets: int
    assets: list[str]
    layout: dict[str, slice]

    def solve(self, x0=None, **kwargs) -> QPSolution:
        return solve_qp(self.P, self.q, self.A, self.l, self.u, x0=x0, **kwargs)


def bounds(cons: Constraints, assets: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    lo = _vector(cons.lower, assets, -INF) if cons.lower is not None else np.full(len(assets), -INF)
    hi = _vector(cons.upper, assets, INF) if cons.upper is not None else np.full(len(assets), INF)
    if cons.max_trade is not None and cons.w_prev is not None:
        cap = _vector(cons.max_trade, assets, INF)
        prev = np.asarray(cons.w_prev, dtype=float)
        lo = np.maximum(lo, prev - cap)
        hi = np.minimum(hi, prev + cap)
    return lo, hi


def check_feasibility(cons: Constraints, assets: Sequence[str]) -> list[str]:
    """Cheap necessary conditions, worded for people (the solver also certifies infeasibility)."""
    msgs = []
    lo, hi = bounds(cons, assets)
    bad = np.flatnonzero(lo > hi + 1e-12)
    for i in bad:
        msgs.append(f"{assets[i]}: lower bound {lo[i]:.3f} is above upper bound {hi[i]:.3f}")
    if cons.budget is not None:
        lo_sum = np.sum(np.where(lo > -INF / 2, lo, -np.inf))
        hi_sum = np.sum(np.where(hi < INF / 2, hi, np.inf))
        if lo_sum > cons.budget + 1e-9:
            msgs.append(f"position floors add up to {lo_sum:.3f}, above the budget {cons.budget:.3f}")
        if hi_sum < cons.budget - 1e-9:
            msgs.append(f"position caps add up to {hi_sum:.3f}, below the budget {cons.budget:.3f}; "
                        "raise the caps or allow cash")
    if cons.groups and cons.budget is not None:
        caps = sum(_group_value(cons.group_max, g, INF) for g in cons.groups)
        covered = set(a for m in cons.groups.values() for a in m)
        if caps < cons.budget - 1e-9 and covered >= set(assets):
            msgs.append(f"group caps add up to {caps:.3f}, below the budget {cons.budget:.3f}")
    if cons.gross_max is not None and cons.budget is not None and cons.gross_max < abs(cons.budget) - 1e-12:
        msgs.append(f"gross limit {cons.gross_max:.2f} is below the budget {cons.budget:.2f}")
    return msgs


def build_qp(P_w, q_w, cons: Constraints, assets: Sequence[str], extra_rows=None) -> QPProblem:
    """Assemble the QP for objective 1/2 w'P_w w + q_w'w under ``cons``.

    ``extra_rows`` is an optional list of (label, row over w, lower, upper) appended as-is (used by the
    efficient frontier for its return target).
    """
    assets = list(assets)
    n = len(assets)
    P_w = np.asarray(P_w, dtype=float)
    q_w = np.asarray(q_w, dtype=float)
    lo, hi = bounds(cons, assets)
    long_only = bool(np.all(lo >= -1e-15))

    need_t = cons.turnover_max is not None or (cons.trade_cost is not None and np.any(_vector(cons.trade_cost, assets, 0.0) != 0))
    need_g = cons.gross_max is not None and not long_only
    w_prev = np.zeros(n) if cons.w_prev is None else np.asarray(cons.w_prev, dtype=float)
    nv = n + (n if need_t else 0) + (n if need_g else 0)
    layout = {"w": slice(0, n)}
    off = n
    if need_t:
        layout["t"] = slice(off, off + n)
        off += n
    if need_g:
        layout["g"] = slice(off, off + n)
        off += n

    rows, lows, ups, labels = [], [], [], []

    def add(row, lower, upper, label):
        rows.append(row)
        lows.append(lower)
        ups.append(upper)
        labels.append(label)

    for i, a in enumerate(assets):
        r = np.zeros(nv)
        r[i] = 1.0
        add(r, lo[i], hi[i], f"bound:{a}")

    r = np.zeros(nv)
    r[:n] = 1.0
    if cons.budget is not None:
        add(r, cons.budget, cons.budget, "budget")
    elif cons.net_min is not None or cons.net_max is not None:
        add(r, -INF if cons.net_min is None else cons.net_min, INF if cons.net_max is None else cons.net_max, "net")

    if cons.groups:
        index = {a: i for i, a in enumerate(assets)}
        for g, members in cons.groups.items():
            idx = [index[m] for m in members if m in index]
            if not idx:
                continue
            gmin = _group_value(cons.group_min, g, -INF)
            gmax = _group_value(cons.group_max, g, INF)
            if gmin <= -INF / 2 and gmax >= INF / 2:
                continue
            r = np.zeros(nv)
            r[idx] = 1.0
            add(r, gmin, gmax, f"group:{g}")

    if cons.factor_betas is not None:
        B = np.asarray(cons.factor_betas, dtype=float).reshape(n, -1)
        names = list(cons.factor_names or [f"f{j}" for j in range(B.shape[1])])
        for j, name in enumerate(names):
            fmin = _group_value(cons.factor_min, name, -INF)
            fmax = _group_value(cons.factor_max, name, INF)
            if fmin <= -INF / 2 and fmax >= INF / 2:
                continue
            r = np.zeros(nv)
            r[:n] = B[:, j]
            add(r, fmin, fmax, f"factor:{name}")

    q = np.zeros(nv)
    q[:n] = q_w
    if need_t:
        ts = layout["t"]
        for i, a in enumerate(assets):
            r = np.zeros(nv)
            r[i], r[ts.start + i] = 1.0, -1.0
            add(r, -INF, w_prev[i], f"trade+:{a}")
            r = np.zeros(nv)
            r[i], r[ts.start + i] = -1.0, -1.0
            add(r, -INF, -w_prev[i], f"trade-:{a}")
        if cons.turnover_max is not None:
            r = np.zeros(nv)
            r[ts] = 1.0
            add(r, -INF, cons.turnover_max, "turnover")
        if cons.trade_cost is not None:
            q[ts] = _vector(cons.trade_cost, assets, 0.0)
    if need_g:
        gs = layout["g"]
        for i, a in enumerate(assets):
            r = np.zeros(nv)
            r[i], r[gs.start + i] = 1.0, -1.0
            add(r, -INF, 0.0, f"gross+:{a}")
            r = np.zeros(nv)
            r[i], r[gs.start + i] = -1.0, -1.0
            add(r, -INF, 0.0, f"gross-:{a}")
        r = np.zeros(nv)
        r[gs] = 1.0
        add(r, -INF, cons.gross_max, "gross")
    elif cons.gross_max is not None:  # long-only: gross = net
        r = np.zeros(nv)
        r[:n] = 1.0
        add(r, -INF, cons.gross_max, "gross")

    for label, row_w, lower, upper in (extra_rows or []):
        r = np.zeros(nv)
        r[:n] = row_w
        add(r, lower, upper, label)

    P = np.zeros((nv, nv))
    P[:n, :n] = P_w
    if nv > n:  # tiny curvature on auxiliaries keeps the polish step's KKT system nonsingular
        P[n:, n:] = 1e-10 * np.eye(nv - n)
    A = np.vstack(rows) if rows else np.zeros((0, nv))
    return QPProblem(P, q, A, np.array(lows, dtype=float), np.array(ups, dtype=float), labels, n, assets, layout)


def binding_constraints(problem: QPProblem, sol: QPSolution, tol: float = 1e-7) -> list[dict]:
    """Constraints at a bound with nonzero multipliers, largest shadow price first.

    ``shadow`` is the multiplier in objective units: the objective improvement per unit of relaxation.
    """
    out = []
    for k, label in enumerate(problem.labels):
        y = float(sol.y[k])
        if abs(y) <= tol:
            continue
        side = "upper" if y > 0 else "lower"
        bound = problem.u[k] if y > 0 else problem.l[k]
        out.append({"constraint": label, "side": side, "bound": float(bound), "shadow": abs(y),
                    "value": float(sol.z[k])})
    out.sort(key=lambda d: -d["shadow"])
    return out
