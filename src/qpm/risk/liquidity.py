"""Liquidity risk: how long positions take to exit and what exiting costs.

Costs follow the square-root law of market impact (e.g. Almgren et al. 2005; Toth et al. 2011):
impact per unit traded ~ eta x daily volatility x sqrt(traded value / ADV).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def days_to_liquidate(weights: pd.Series, aum: float, adv: pd.Series, participation: float = 0.1) -> pd.Series:
    """Days needed to trade out of each position at ``participation`` x average daily traded value."""
    adv = adv.reindex(weights.index)
    return (weights.abs() * aum) / (participation * adv)


def liquidity_profile(weights: pd.Series, aum: float, adv: pd.Series, participation: float = 0.1,
                      horizons=(1, 5, 20)) -> dict:
    """Share of the portfolio (by gross weight) that can be exited within each horizon."""
    d = days_to_liquidate(weights, aum, adv, participation)
    gross = weights.abs().sum()
    out = {}
    for h in horizons:
        sellable = np.minimum(weights.abs(), (participation * adv.reindex(weights.index) * h) / aum)
        out[f"{h}d"] = float(sellable.sum() / gross) if gross > 0 else 1.0
    out["max_days"] = float(d.max())
    out["weighted_days"] = float((weights.abs() * d).sum() / gross) if gross > 0 else 0.0
    return out


def liquidation_cost(weights: pd.Series, aum: float, adv: pd.Series, daily_vol: pd.Series,
                     spread_bps: pd.Series | float = 5.0, eta: float = 0.7, participation: float = 0.1) -> pd.Series:
    """Cost (fraction of capital) of exiting each position at the given participation rate."""
    trade = weights.abs() * aum
    spread = spread_bps if np.isscalar(spread_bps) else spread_bps.reindex(weights.index)
    impact = eta * daily_vol.reindex(weights.index) * np.sqrt(trade / adv.reindex(weights.index))
    per_unit = spread / 1e4 / 2 + impact
    return weights.abs() * per_unit


def liquidity_adjusted_var(var: float, weights: pd.Series, spread_bps: pd.Series | float = 5.0,
                           spread_vol_bps: pd.Series | float = 0.0, z: float = 2.33) -> float:
    """Bangia, Diebold, Schuermann and Stroughair (1999): add half the (stressed) spread to VaR."""
    s = spread_bps if np.isscalar(spread_bps) else spread_bps.reindex(weights.index)
    sv = spread_vol_bps if np.isscalar(spread_vol_bps) else spread_vol_bps.reindex(weights.index)
    cost = 0.5 * (weights.abs() * (s + z * sv) / 1e4).sum()
    return float(var + cost)
