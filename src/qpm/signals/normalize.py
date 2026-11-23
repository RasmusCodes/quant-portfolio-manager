"""Putting signals from different systems on one scale.

Raw signals arrive in incompatible units (a z-score, a percentage spread, an expected return over five
days).  The steps here map each to a cross-sectional z-score, then to an expected return over the
portfolio's rebalancing horizon with the Grinold rule alpha = IC x volatility x z.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def zscore(x: pd.Series, clip: float | None = 3.0) -> pd.Series:
    """Cross-sectional z-score; outliers clipped at +/- ``clip`` then re-standardised."""
    x = pd.to_numeric(x, errors="coerce")
    valid = x.dropna()
    if len(valid) < 2 or valid.std(ddof=0) == 0:
        return x * 0.0
    z = (x - valid.mean()) / valid.std(ddof=0)
    if clip is not None:
        z = z.clip(-clip, clip)
        v = z.dropna()
        if v.std(ddof=0) > 0:
            z = (z - v.mean()) / v.std(ddof=0)
    return z


def rank_gauss(x: pd.Series) -> pd.Series:
    """Map ranks to normal quantiles: robust to outliers, keeps order only."""
    x = pd.to_numeric(x, errors="coerce")
    r = x.rank()
    n = r.notna().sum()
    return pd.Series(stats.norm.ppf((r - 0.5) / n), index=x.index) if n else x * 0.0


def panel_zscore(panel: pd.DataFrame, clip: float | None = 3.0, method: str = "zscore") -> pd.DataFrame:
    fn = zscore if method == "zscore" else (lambda s, clip=None: rank_gauss(s))
    return panel.apply(lambda row: fn(row, clip=clip) if method == "zscore" else fn(row), axis=1)


def grinold_expected_return(z: pd.Series, volatility: pd.Series, ic: float) -> pd.Series:
    return ic * volatility.reindex(z.index) * z


def horizon_scale(expected_return: float | pd.Series, from_days: float, to_days: float):
    """Expected returns scale linearly with time (an expected 1% over 5 days is 4.2% over 21)."""
    return expected_return * (to_days / from_days)


def age_decay(age_days: float | np.ndarray, horizon_days: float | np.ndarray) -> np.ndarray:
    """Weight for a signal ``age_days`` old: exp(-age / horizon); zero past three horizons."""
    a = np.asarray(age_days, dtype=float)
    h = np.asarray(horizon_days, dtype=float)
    w = np.exp(-a / h)
    return np.where(a > 3 * h, 0.0, w)


def store_to_scores(snapshot: pd.DataFrame, instruments, rebalance_days: float = 21.0) -> pd.DataFrame:
    """One ``SignalStore.as_of`` snapshot -> instruments x sources table of comparable z-scores.

    expected_return signals are first converted to the rebalancing horizon and divided by volatility
    (a Sharpe-like score); direction signals are used as +/-1; everything is then z-scored per source.
    """
    out = {}
    for source, g in snapshot.groupby("source"):
        g = g.set_index("instrument")
        val = g["value"].astype(float)
        if (g["kind"] == "expected_return").all():
            per = horizon_scale(val, g["horizon_days"].astype(float), rebalance_days)
            vol = g["volatility"].astype(float)
            val = per / vol.where(vol > 0) if vol.notna().all() else per
        out[source] = zscore(val.reindex(list(instruments)))
    return pd.DataFrame(out, index=list(instruments))
