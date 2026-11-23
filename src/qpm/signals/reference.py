"""Reference research systems built only from public data.

Each function returns a raw score panel indexed by the date the score becomes known (a month end), so
it can go straight into a ``SignalStore``.  They are deliberately simple, published ideas: they stand in
for the user's own research systems in the public study, and they are what the demo's built-in signals
are computed from.

    momentum_12_1      equity factor        return over months t-11..t-1 (Jegadeesh-Titman; for
                                            industries, Moskowitz-Grinblatt 1999)
    value_bm           equity factor        log book-to-market, French's June convention
    low_beta           equity factor        minus the 60-month market beta (Frazzini-Pedersen 2014)
    residual_reversal  statistical arb.     minus this month's market-model residual (short-term
                                            reversal of idiosyncratic moves)
    tsmom              macro / fixed income 12-month excess return over its volatility (Moskowitz,
                                            Ooi and Pedersen 2012), a time-series signal
    term_carry         macro / fixed income Treasury yield minus the bill rate (bond carry)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SPECS = {
    "momentum_12_1": {"family": "equity_factor", "horizon_days": 21, "label": "Momentum 12-1",
                      "inputs": "monthly returns through t-1"},
    "value_bm": {"family": "equity_factor", "horizon_days": 252, "label": "Value (book-to-market)",
                 "inputs": "French BE/ME, known from June of year Y"},
    "low_beta": {"family": "equity_factor", "horizon_days": 21, "label": "Low beta",
                 "inputs": "60 monthly returns through t"},
    "residual_reversal": {"family": "statistical_arbitrage", "horizon_days": 21, "label": "Residual reversal",
                          "inputs": "36 monthly returns through t"},
    "tsmom": {"family": "macro_fixed_income", "horizon_days": 21, "label": "Time-series momentum",
              "inputs": "12 monthly excess returns through t"},
    "term_carry": {"family": "macro_fixed_income", "horizon_days": 21, "label": "Bond carry",
                   "inputs": "month-end Treasury yields and the T-bill rate"},
}


def momentum_12_1(returns: pd.DataFrame) -> pd.DataFrame:
    logr = np.log1p(returns)
    return np.expm1(logr.shift(1).rolling(11, min_periods=11).sum())


def _rolling_beta(returns: pd.DataFrame, market: pd.Series, window: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = market.reindex(returns.index)
    mean_m = m.rolling(window, min_periods=window).mean()
    var_m = m.rolling(window, min_periods=window).var()
    beta = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    alpha = pd.DataFrame(index=returns.index, columns=returns.columns, dtype=float)
    for c in returns.columns:
        cov = returns[c].rolling(window, min_periods=window).cov(m)
        b = cov / var_m
        beta[c] = b
        alpha[c] = returns[c].rolling(window, min_periods=window).mean() - b * mean_m
    return beta, alpha


def low_beta(excess_returns: pd.DataFrame, market_excess: pd.Series, window: int = 60) -> pd.DataFrame:
    beta, _ = _rolling_beta(excess_returns, market_excess, window)
    return -beta


def residual_reversal(excess_returns: pd.DataFrame, market_excess: pd.Series, window: int = 36) -> pd.DataFrame:
    beta, alpha = _rolling_beta(excess_returns, market_excess, window)
    resid = excess_returns - alpha - beta.mul(market_excess.reindex(excess_returns.index), axis=0)
    return -resid


def value_bm(bm_pit: pd.DataFrame) -> pd.DataFrame:
    bm = bm_pit.where(bm_pit > 0)
    return np.log(bm)


def tsmom(excess_returns: pd.DataFrame, lookback: int = 12, vol_window: int = 36) -> pd.DataFrame:
    logr = np.log1p(excess_returns)
    trail = logr.rolling(lookback, min_periods=lookback).sum()
    vol = excess_returns.rolling(vol_window, min_periods=24).std() * np.sqrt(lookback)
    return trail / vol


def term_carry(yields_pct: pd.DataFrame, rf_monthly: pd.Series, mapping: dict[str, str]) -> pd.DataFrame:
    """Carry of each bond = its yield minus the annualised bill rate, in decimals a year.
    ``mapping``: asset name -> yield column (e.g. {"UST10": "DGS10"})."""
    bill = (1 + rf_monthly) ** 12 - 1
    out = pd.DataFrame({a: yields_pct[col] / 100.0 for a, col in mapping.items()})
    return out.sub(bill.reindex(out.index), axis=0)


def build(universe, include=("momentum_12_1", "value_bm", "low_beta", "residual_reversal")) -> dict[str, pd.DataFrame]:
    """Raw score panels for a ``qpm.data.Universe`` (equity assets only for the cross-sectional signals)."""
    ex = universe.returns.sub(universe.rf, axis=0)
    eq = [a for a in universe.assets if universe.group.get(a) not in ("Treasury",)]
    mkt = universe.factors["Mkt-RF"] if universe.factors is not None else ex[eq].mean(axis=1)
    out = {}
    for name in include:
        if name == "momentum_12_1":
            out[name] = momentum_12_1(universe.returns[eq])
        elif name == "value_bm":
            if universe.bm is None:
                continue
            out[name] = value_bm(universe.bm[eq])
        elif name == "low_beta":
            out[name] = low_beta(ex[eq], mkt)
        elif name == "residual_reversal":
            out[name] = residual_reversal(ex[eq], mkt)
        elif name == "tsmom":
            out[name] = tsmom(ex)
        else:
            raise KeyError(name)
    return out
