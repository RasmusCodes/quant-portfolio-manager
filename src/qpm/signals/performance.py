"""Signal track records: information coefficients, measured point in time.

Convention: ``fwd`` has, on row t, the return from t to t+1, so it is only KNOWN at t+1.  An IC stamped
t therefore enters a track record from t+1 on; ``track_record_asof`` enforces that.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(returns: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Row t: compounded return over t+1..t+periods (NaN where not yet realised)."""
    growth = np.log1p(returns)
    fwd = growth.rolling(periods).sum().shift(-periods)
    return np.expm1(fwd)


def cross_sectional_ic(scores: pd.DataFrame, fwd: pd.DataFrame, method: str = "spearman", min_obs: int = 5) -> pd.Series:
    """IC per date: correlation across instruments between the score at t and the return over t..t+1."""
    s, r = scores.align(fwd, join="inner")
    out = {}
    for date in s.index:
        a, b = s.loc[date], r.loc[date]
        ok = a.notna() & b.notna()
        if ok.sum() < min_obs:
            continue
        a, b = a[ok], b[ok]
        if method == "spearman":
            a, b = a.rank(), b.rank()
        if a.std() == 0 or b.std() == 0:
            continue
        out[date] = float(np.corrcoef(a, b)[0, 1])
    return pd.Series(out, dtype=float)


def newey_west_t(x: pd.Series, lags: int | None = None) -> float:
    x = pd.Series(x).dropna().values
    n = x.size
    if n < 3:
        return np.nan
    lags = int(np.floor(4 * (n / 100) ** (2 / 9))) if lags is None else lags
    e = x - x.mean()
    gamma0 = e @ e / n
    s = gamma0
    for k in range(1, lags + 1):
        g = e[k:] @ e[:-k] / n
        s += 2 * (1 - k / (lags + 1)) * g
    se = np.sqrt(s / n)
    return float(x.mean() / se) if se > 0 else np.nan


def ic_summary(ic: pd.Series) -> dict:
    ic = ic.dropna()
    if ic.empty:
        return {"ic_mean": np.nan, "ic_std": np.nan, "ic_t": np.nan, "hit_rate": np.nan, "n_obs": 0, "icir": np.nan}
    return {"ic_mean": float(ic.mean()), "ic_std": float(ic.std()), "ic_t": newey_west_t(ic),
            "hit_rate": float((ic > 0).mean()), "n_obs": int(ic.size),
            "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else np.nan}


def known_ics(ic: pd.Series, as_of, horizon_periods: int = 1, index: pd.Index | None = None) -> pd.Series:
    """ICs whose forward returns are realised by ``as_of`` (stamp t realised at t + horizon)."""
    if index is None:
        index = ic.index
    as_of = pd.Timestamp(as_of)
    pos = index.searchsorted(as_of, side="right") - 1  # position of as_of (or the last date before it)
    cutoff_pos = pos - horizon_periods
    if cutoff_pos < 0:
        return ic.iloc[0:0]
    cutoff = index[cutoff_pos]
    return ic[ic.index <= cutoff]


def track_record_asof(ic: pd.Series, as_of, horizon_periods: int = 1, index: pd.Index | None = None) -> dict:
    return ic_summary(known_ics(ic, as_of, horizon_periods, index))


def ewma_ic(ic: pd.Series, halflife: float) -> pd.Series:
    """EWMA of the IC series; value on row t uses ICs up to and including t (shift before use)."""
    return ic.ewm(halflife=halflife, min_periods=1).mean()


def ic_decay(scores: pd.DataFrame, returns: pd.DataFrame, max_lag: int = 12, method: str = "spearman") -> pd.Series:
    """Mean IC of the score at t against the single-period return in t+lag (lag = 1..max_lag)."""
    out = {}
    for lag in range(1, max_lag + 1):
        fwd = returns.shift(-lag)
        out[lag] = cross_sectional_ic(scores, fwd, method).mean()
    return pd.Series(out)
