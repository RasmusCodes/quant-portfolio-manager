"""Strategies for the backtester: an optimizer-driven strategy and fixed-rule baselines.

``OptimizerStrategy`` re-estimates inputs from the trailing window at every decision and runs one of the
construction methods.  Expected returns can come from history (sample, Bayes-Stein, James-Stein), from
equilibrium (reverse-optimized market weights), from a combined signal (Grinold: IC x vol x z), or be
switched off ("zero": risk-based methods ignore them anyway).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..estimation.covariance import estimate_cov
from ..estimation.returns import bayes_stein, equilibrium_returns, james_stein_grand_mean
from ..optimize.constraints import Constraints
from ..optimize.methods import optimize
from .engine import DecisionContext


@dataclass
class OptimizerStrategy:
    method: str
    name: str = ""
    window: int = 120
    cov_method: str = "lw_cc"
    mean_method: str = "sample"           # sample | bayes_stein | james_stein | equilibrium | signal | zero
    risk_aversion: float = 4.0
    constraints: Constraints = field(default_factory=lambda: Constraints(lower=0.0))
    signal: str | None = None             # key in ctx.signals for mean_method="signal" or BL views
    signal_ic: float = 0.05
    bl_tau: float = 0.05
    bl_confidence: float = 0.25
    cost_aware: bool = False              # pass current weights + trade costs into the optimizer
    trade_cost: float | dict = 0.001      # objective units per unit traded when cost_aware ({asset: cost} allowed)
    market_weights: object = None         # callable(ctx, assets) -> weights for equilibrium / BL / market_cap
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        self.name = self.name or self.method

    def _inputs(self, ctx: DecisionContext):
        hist = ctx.returns.iloc[-self.window:]
        ex = hist.sub(ctx.rf.reindex(hist.index), axis=0)
        ok = ex.columns[ex.notna().all()]
        ex = ex[ok]
        if self.cov_method == "factor":
            f = ctx.universe.factors.reindex(ex.index)
            keep = f.notna().all(axis=1)
            cov = estimate_cov(ex.values[keep.values], "factor", factors=f.values[keep.values]) * 12.0
        else:
            cov = estimate_cov(ex.values, self.cov_method) * 12.0
        cov = pd.DataFrame(cov, index=ok, columns=ok)
        if self.mean_method == "sample":
            mu = ex.mean() * 12.0
        elif self.mean_method == "bayes_stein":
            mu = pd.Series(bayes_stein(ex.values)[0] * 12.0, index=ok)
        elif self.mean_method == "james_stein":
            mu = pd.Series(james_stein_grand_mean(ex.values)[0] * 12.0, index=ok)
        elif self.mean_method == "equilibrium":
            mu = pd.Series(equilibrium_returns(cov.values, self._market_weights(ctx, ok), self.params.get("delta", 2.5)), index=ok)
        elif self.mean_method == "signal":
            z = ctx.signals[self.signal].iloc[-1].reindex(ok)
            vol = pd.Series(np.sqrt(np.diag(cov.values)), index=ok)
            mu = (self.signal_ic * vol * z).fillna(0.0)
        elif self.mean_method == "zero":
            mu = pd.Series(0.0, index=ok)
        else:
            raise KeyError(self.mean_method)
        return mu, cov

    def _market_weights(self, ctx: DecisionContext, assets) -> np.ndarray:
        if callable(self.market_weights):
            w = np.asarray(self.market_weights(ctx, list(assets)), dtype=float)
            return w / w.sum()
        u = ctx.universe
        if u is not None and u.mcap is not None and not u.mcap.empty:
            caps = u.mcap.iloc[-1].reindex(assets)
            if caps.notna().any():
                caps = caps.fillna(caps.median())
                return (caps / caps.sum()).values
        return np.full(len(assets), 1.0 / len(assets))

    def target(self, ctx: DecisionContext) -> pd.Series:
        mu, cov = self._inputs(ctx)
        assets = list(cov.columns)
        cons = self.constraints
        if self.cost_aware or cons.turnover_max is not None:
            prev = ctx.weights.reindex(assets).fillna(0.0).values
            tc = self.trade_cost
            if isinstance(tc, dict):
                tc = np.array([float(tc.get(a, tc.get("default", 0.001))) for a in assets])
            cons = cons.with_(w_prev=prev, trade_cost=tc if self.cost_aware else cons.trade_cost)
        kw = dict(self.params)
        kw.setdefault("risk_aversion", self.risk_aversion)
        if self.method in ("market_cap", "black_litterman"):
            kw["market_weights"] = pd.Series(self._market_weights(ctx, assets), index=assets)
        if self.method == "black_litterman" and self.signal is not None:
            z = ctx.signals[self.signal].iloc[-1].reindex(assets)
            have = z.notna() & (z.abs() > 1e-12)
            if have.any():
                vol = np.sqrt(np.diag(cov.values))
                pi = equilibrium_returns(cov.values, kw["market_weights"].values, kw["risk_aversion"])
                idx = np.flatnonzero(have.values)
                P = np.eye(len(assets))[idx]
                Q = pi[idx] + self.signal_ic * vol[idx] * z.values[idx]   # view: equilibrium plus the signal's alpha
                kw.update(P=P, Q=Q, confidences=np.full(len(idx), self.bl_confidence), tau=self.bl_tau)
        if self.method == "factor_neutral":
            kw.setdefault("betas", self.params.get("betas"))
        p = optimize(self.method, mu, cov, cons, assets, **kw)
        if not p.ok:
            raise RuntimeError(f"{self.method}: {p.status} {'; '.join(p.messages)}")
        ctx.state["last"] = p
        return p.weights


@dataclass
class FixedWeights:
    weights: dict
    name: str = "fixed"

    def target(self, ctx: DecisionContext) -> pd.Series:
        return pd.Series(self.weights)


@dataclass
class EqualWeight:
    name: str = "equal_weight"

    def target(self, ctx: DecisionContext) -> pd.Series:
        live = ctx.returns.columns[ctx.returns.iloc[-1].notna()]
        return pd.Series(1.0 / len(live), index=live)


@dataclass
class SignalLongShort:
    """Dollar-neutral book proportional to a combined score (z / sum|z|, gross = ``gross``)."""
    signal: str
    gross: float = 1.0
    name: str = "signal_long_short"

    def target(self, ctx: DecisionContext) -> pd.Series:
        z = ctx.signals[self.signal].iloc[-1].dropna()
        if z.abs().sum() == 0:
            return pd.Series(0.0, index=ctx.returns.columns)
        z = z - z.mean()
        return self.gross * z / z.abs().sum()
