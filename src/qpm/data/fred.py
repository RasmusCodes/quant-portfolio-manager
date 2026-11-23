"""FRED series and constant-maturity Treasury total returns rebuilt from H.15 yields.

FRED has no Treasury return index, so returns are rebuilt from constant-maturity par yields the usual
way (e.g. Swinkels 2019, "Treasury bond return data starting in 1962"): at each date buy a par bond with
coupon equal to that day's yield and maturity M; at the next date reprice it at the new yield with
maturity M - dt and add the accrued coupon.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def read_fred_csv(path: str | Path) -> pd.Series:
    """A FRED ``fredgraph.csv`` download -> float Series indexed by date (blank or '.' -> NaN)."""
    frame = pd.read_csv(path, na_values=[".", ""], keep_default_na=True)
    date_col = frame.columns[0]
    series = pd.Series(pd.to_numeric(frame.iloc[:, 1], errors="coerce").values,
                       index=pd.to_datetime(frame[date_col]), name=frame.columns[1])
    return series.sort_index()


def par_bond_price(y: np.ndarray | float, coupon: np.ndarray | float, maturity: np.ndarray | float) -> np.ndarray:
    """Clean price per unit face of a semiannual-coupon bond (yields and coupons as decimals).

    Uses the closed form with a fractional number of periods n = 2 * maturity, which is exact at coupon
    dates and a standard approximation between them.
    """
    y = np.asarray(y, dtype=float)
    coupon = np.asarray(coupon, dtype=float)
    maturity = np.asarray(maturity, dtype=float)
    disc = (1.0 + y / 2.0) ** (-2.0 * maturity)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = np.where(np.abs(y) > 1e-10, (1.0 - disc) / y, maturity)  # limit y -> 0
    return coupon * annuity + disc


def constant_maturity_returns(yields_pct: pd.Series, maturity: float, frequency: str = "M") -> pd.Series:
    """Total returns of a constant-maturity Treasury position from a daily yield series in percent.

    ``frequency``: "M" uses the last available yield of each month (dt = 1/12); "D" uses consecutive
    observations (dt = calendar days / 365.25).  The first period has no return and is dropped.
    """
    y = yields_pct.dropna() / 100.0
    if frequency.upper().startswith("M"):
        y = month_end_level(y)
        dt = np.full(len(y) - 1, 1.0 / 12.0)
    else:
        days = np.diff(y.index.values).astype("timedelta64[D]").astype(float)
        dt = days / 365.25
    y0, y1 = y.values[:-1], y.values[1:]
    price = par_bond_price(y1, y0, maturity - dt)
    ret = price + y0 * dt - 1.0
    return pd.Series(ret, index=y.index[1:], name=f"UST{maturity:g}")


def month_end_level(series: pd.Series, tolerance_days: int = 7) -> pd.Series:
    """Last observation of each calendar month, indexed by the month-end date.

    The final month is dropped when its last observation is more than ``tolerance_days`` before the
    month end (a month still in progress), so a partial month is never labelled as complete.
    """
    s = series.dropna()
    last_obs = s.index.to_series().groupby(s.index.to_period("M")).max()
    out = s.groupby(s.index.to_period("M")).last()
    out.index = out.index.to_timestamp(how="end").normalize()
    if len(out) and (out.index[-1] - last_obs.iloc[-1]).days > tolerance_days:
        out = out.iloc[:-1]
    return out
