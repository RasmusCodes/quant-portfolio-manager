"""Drawdowns and correlation breakdown."""
from __future__ import annotations

import numpy as np
import pandas as pd


def drawdown_series(returns: pd.Series) -> pd.DataFrame:
    wealth = (1 + returns.fillna(0.0)).cumprod()
    peak = wealth.cummax().clip(lower=1.0)
    return pd.DataFrame({"wealth": wealth, "peak": peak, "drawdown": wealth / peak - 1.0})


def drawdown_episodes(returns: pd.Series, top: int = 5) -> pd.DataFrame:
    """Largest peak-to-trough episodes with start, trough, recovery date (NaT if not recovered)."""
    dd = drawdown_series(returns)["drawdown"]
    episodes, in_dd, start = [], False, None
    for date, value in dd.items():
        if value < 0 and not in_dd:
            in_dd, start = True, date
        elif value >= 0 and in_dd:
            seg = dd.loc[start:date]
            episodes.append((start, seg.idxmin(), date, seg.min()))
            in_dd = False
    if in_dd:
        seg = dd.loc[start:]
        episodes.append((start, seg.idxmin(), pd.NaT, seg.min()))
    frame = pd.DataFrame(episodes, columns=["start", "trough", "recovered", "depth"])
    if frame.empty:
        return frame
    frame["months_to_trough"] = [len(dd.loc[s:t]) for s, t in zip(frame.start, frame.trough)]
    frame["months_to_recover"] = [len(dd.loc[t:r]) - 1 if pd.notna(r) else np.nan for t, r in zip(frame.trough, frame.recovered)]
    return frame.sort_values("depth").head(top).reset_index(drop=True)


def conditional_drawdown_at_risk(returns: pd.Series, alpha: float = 0.95) -> float:
    """Average of the worst (1 - alpha) share of drawdown observations (Chekhlov, Uryasev, Zabarankin 2005)."""
    dd = -drawdown_series(returns)["drawdown"].values
    cut = np.quantile(dd, alpha)
    tail = dd[dd >= cut]
    return float(tail.mean()) if tail.size else 0.0


def average_correlation(returns: pd.DataFrame) -> float:
    c = returns.corr().values
    n = c.shape[0]
    return float((c.sum() - n) / (n * (n - 1))) if n > 1 else np.nan


def stress_regime(market: pd.Series, window: int = 12, quantile: float = 0.8) -> pd.Series:
    """High-volatility months: trailing ``window``-month volatility of the market above its ``quantile``.

    Months are NOT picked by their own market return: conditioning on the tail of the common factor
    shrinks its variance inside the subsample and biases measured correlations down (the sample-selection
    problem in Forbes and Rigobon 2002).  This regime is descriptive (full-sample cut-off), not a signal.
    """
    vol = market.rolling(window, min_periods=window).std()
    return (vol > vol.quantile(quantile)).where(vol.notna())


def correlation_breakdown(returns: pd.DataFrame, market: pd.Series, window: int = 12, quantile: float = 0.8,
                          bonds: list[str] | None = None) -> dict:
    """Average pairwise correlation among risky assets in high-volatility vs other months, the stock-bond
    correlation in each regime, and the absorption ratio of the top fifth of eigenvectors.

    Higher correlation in volatile regimes is partly mechanical (a common factor's variance rises, Forbes
    and Rigobon 2002), which is exactly why diversification measured in calm times overstates it."""
    bonds = [b for b in (bonds or []) if b in returns.columns]
    risky = [c for c in returns.columns if c not in bonds]
    reg = stress_regime(market, window, quantile).reindex(returns.index)
    hi, lo = returns[reg == 1], returns[reg == 0]
    out = {"stress_avg_corr": average_correlation(hi[risky]), "calm_avg_corr": average_correlation(lo[risky]),
           "stress_months": int(len(hi)), "calm_months": int(len(lo)), "absorption_ratio": absorption_ratio(returns[risky]),
           "regime": f"trailing {window}-month market volatility above its {quantile:.0%} quantile"}
    if bonds:
        eq = returns[risky].mean(axis=1)
        for name, part in (("stress", reg == 1), ("calm", reg == 0)):
            out[f"{name}_stock_bond_corr"] = float(eq[part].corr(returns.loc[part, bonds].mean(axis=1)))
    return out


def absorption_ratio(returns: pd.DataFrame, fraction: float = 0.2) -> float:
    """Kritzman, Li, Page and Rigobon (2011): variance share of the top ``fraction`` of eigenvectors."""
    vals = np.sort(np.linalg.eigvalsh(np.cov(returns.dropna().values, rowvar=False)))[::-1]
    k = max(1, int(round(fraction * len(vals))))
    return float(vals[:k].sum() / vals.sum())


def rolling_average_correlation(returns: pd.DataFrame, window: int = 36) -> pd.Series:
    out = {}
    for end in range(window, len(returns) + 1):
        chunk = returns.iloc[end - window:end]
        out[returns.index[end - 1]] = average_correlation(chunk)
    return pd.Series(out)
