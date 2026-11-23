"""Configurable stress scenarios.

A scenario is plain data (JSON in ``scenarios/``), so new ones need no code.  Kinds:

* ``historical``  replay the assets' actual returns between two dates (weights held fixed).
* ``factor``      shocks to factors, mapped to assets through estimated betas, optionally with direct
                  ``asset_shocks`` that replace the factor-implied move for named assets.
* ``asset``       direct shocks per asset (unnamed assets get ``default_shock``).
* ``volatility``  a covariance stress: volatilities x ``vol_multiplier`` and correlations pulled toward
                  ``correlation_target``; reports VaR/ES under the stressed covariance plus any
                  ``factor_shocks`` as the accompanying move.

Example (factor kind)::

    {"name": "AI infrastructure capex reversal", "kind": "factor",
     "factor_shocks": {"Mkt-RF": -0.15, "Mom": -0.20},
     "asset_shocks": {"BusEq": -0.40, "Utils": -0.15},
     "description": "..."}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..estimation.covariance import corr_to_cov, cov_to_corr, nearest_psd


@dataclass
class Scenario:
    name: str
    kind: str
    description: str = ""
    start: str | None = None
    end: str | None = None
    factor_shocks: dict = field(default_factory=dict)
    asset_shocks: dict = field(default_factory=dict)
    group_shocks: dict = field(default_factory=dict)
    default_shock: float = 0.0
    vol_multiplier: float = 1.0
    correlation_target: float | None = None
    correlation_weight: float = 0.5
    horizon_months: float = 1.0
    tags: list = field(default_factory=list)
    key: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "Scenario":
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        if "kind" not in known or "name" not in known:
            raise ValueError("a scenario needs at least 'name' and 'kind'")
        if known["kind"] not in ("historical", "factor", "asset", "volatility"):
            raise ValueError(f"unknown scenario kind {known['kind']!r}")
        return cls(**known)


def load_scenarios(path: str | Path) -> list[Scenario]:
    """Load one JSON file (a scenario or a list) or every *.json in a folder."""
    path = Path(path)
    files = sorted(path.glob("*.json")) if path.is_dir() else [path]
    out = []
    for f in files:
        data = json.loads(f.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        for item in items:
            sc = Scenario.from_dict(item)
            sc.key = sc.key or f.stem
            out.append(sc)
    return out


def stressed_cov(cov, vol_multiplier: float = 1.0, correlation_target: float | None = None,
                 correlation_weight: float = 0.5) -> np.ndarray:
    corr, sd = cov_to_corr(cov)
    if correlation_target is not None:
        corr = (1 - correlation_weight) * corr + correlation_weight * correlation_target
        np.fill_diagonal(corr, 1.0)
    return nearest_psd(corr_to_cov(corr, sd * vol_multiplier))


def asset_moves(sc: Scenario, assets: list[str], betas: pd.DataFrame | None = None,
                groups: dict | None = None, returns: pd.DataFrame | None = None) -> pd.Series:
    """Scenario move per asset (simple return over the scenario horizon)."""
    if sc.kind == "historical":
        if returns is None:
            raise ValueError("historical scenarios need asset returns")
        window = returns.loc[sc.start:sc.end, assets]
        if window.empty:
            raise ValueError(f"no data between {sc.start} and {sc.end}")
        return (1 + window).prod() - 1
    moves = pd.Series(sc.default_shock, index=assets, dtype=float)
    if sc.factor_shocks:
        if betas is None:
            raise ValueError(f"scenario {sc.name!r} has factor shocks but no betas were given")
        f = [k for k in sc.factor_shocks if k in betas.columns]
        shock = np.array([sc.factor_shocks[k] for k in f])
        moves = pd.Series(betas.reindex(assets)[f].fillna(0.0).values @ shock, index=assets)
    if sc.group_shocks and groups:
        for a in assets:
            g = groups.get(a)
            if g in sc.group_shocks:
                moves[a] = sc.group_shocks[g]
    for a, v in sc.asset_shocks.items():
        if a in moves.index:
            moves[a] = v
    return moves


def run_scenario(weights: pd.Series, sc: Scenario, betas: pd.DataFrame | None = None, groups: dict | None = None,
                 returns: pd.DataFrame | None = None, cov: pd.DataFrame | None = None, alpha: float = 0.99) -> dict:
    assets = list(weights.index)
    out = {"scenario": sc.name, "key": sc.key, "kind": sc.kind, "description": sc.description}
    if sc.kind == "volatility":
        if cov is None:
            raise ValueError("volatility scenarios need a covariance matrix")
        base = cov.loc[assets, assets].values
        st = stressed_cov(base, sc.vol_multiplier, sc.correlation_target, sc.correlation_weight)
        w = weights.values
        h = sc.horizon_months / 12.0
        vol_b, vol_s = np.sqrt(w @ base @ w * h), np.sqrt(w @ st @ w * h)
        z = stats.norm.ppf(alpha)
        out.update({"base_vol": float(vol_b), "stressed_vol": float(vol_s), "stressed_var": float(z * vol_s),
                    "stressed_es": float(vol_s * stats.norm.pdf(z) / (1 - alpha))})
        moves = asset_moves(Scenario(sc.name, "factor", factor_shocks=sc.factor_shocks, asset_shocks=sc.asset_shocks,
                                     group_shocks=sc.group_shocks, default_shock=sc.default_shock),
                            assets, betas, groups) if (sc.factor_shocks or sc.asset_shocks or sc.group_shocks) \
            else pd.Series(0.0, index=assets)
    else:
        moves = asset_moves(sc, assets, betas, groups, returns)
    contrib = weights * moves
    out.update({"pnl": float(contrib.sum()), "moves": moves, "contributions": contrib,
                "worst": contrib.sort_values().head(3).to_dict()})
    return out


def run_scenarios(weights: pd.Series, scenarios: list[Scenario], **context) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        try:
            r = run_scenario(weights, sc, **context)
            rows.append({"scenario": r["scenario"], "kind": r["kind"], "pnl": r["pnl"],
                         "stressed_var": r.get("stressed_var"),
                         "worst": ", ".join(f"{k} {v:+.1%}" for k, v in r["worst"].items())})
        except ValueError as exc:
            rows.append({"scenario": sc.name, "kind": sc.kind, "pnl": np.nan, "stressed_var": None, "worst": str(exc)})
    return pd.DataFrame(rows)


def scenario_betas(asset_returns: pd.DataFrame, factor_returns: pd.DataFrame, window: int | None = 120) -> pd.DataFrame:
    """OLS betas of each asset on the scenario factors over the last ``window`` months."""
    data = asset_returns.join(factor_returns, how="inner").dropna()
    if window:
        data = data.iloc[-window:]
    a = data[asset_returns.columns].values
    f = data[factor_returns.columns].values
    design = np.column_stack([np.ones(len(f)), f])
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    return pd.DataFrame(coef[1:].T, index=asset_returns.columns, columns=factor_returns.columns)
