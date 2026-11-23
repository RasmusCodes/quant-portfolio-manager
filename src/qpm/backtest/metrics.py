"""Performance statistics and tests for comparing strategies.

Sharpe-ratio comparisons use the Jobson-Korkie test with Memmel's (2003) correction and a paired
stationary bootstrap (Politis-Romano 1994) that keeps both strategies' months together, which allows for
fat tails and autocorrelation.  With many strategies compared, the deflated Sharpe ratio (Bailey and
Lopez de Prado 2014) says how high the best one had to be to beat luck.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def performance_stats(returns: pd.Series, rf: pd.Series | None = None, periods_per_year: int = 12,
                      gamma: float = 5.0) -> dict:
    r = returns.dropna()
    ex = r - (rf.reindex(r.index).fillna(0.0) if rf is not None else 0.0)
    n = len(r)
    if n < 2:
        return {}
    years = n / periods_per_year
    growth = float((1 + r).prod())
    vol = float(r.std() * np.sqrt(periods_per_year))
    ex_mean = float(ex.mean() * periods_per_year)
    ex_vol = float(ex.std() * np.sqrt(periods_per_year))
    downside = ex[ex < 0]
    dd = 1 - (1 + r).cumprod() / (1 + r).cumprod().cummax().clip(lower=1.0)
    max_dd = float(dd.max())
    ann = growth ** (1 / years) - 1
    return {
        "ann_return": ann,
        "ann_excess": ex_mean,
        "volatility": vol,
        "sharpe": ex_mean / ex_vol if ex_vol > 0 else np.nan,
        "sortino": ex_mean / (np.sqrt((downside ** 2).sum() / n) * np.sqrt(periods_per_year)) if len(downside) else np.nan,
        "max_drawdown": max_dd,
        "calmar": ann / max_dd if max_dd > 0 else np.nan,
        "ce_return": ex_mean - 0.5 * gamma * ex_vol ** 2,
        "skew": float(stats.skew(ex)),
        "kurtosis": float(stats.kurtosis(ex)),
        "worst_month": float(r.min()),
        "hit_rate": float((ex > 0).mean()),
        "months": n,
    }


def sharpe_diff_jkm(a: pd.Series, b: pd.Series, periods_per_year: int = 12) -> dict:
    """Jobson-Korkie test with Memmel's correction for SR_a - SR_b on paired excess returns."""
    x = pd.concat([a, b], axis=1).dropna()
    t = len(x)
    ra, rb = x.iloc[:, 0], x.iloc[:, 1]
    sa, sb = ra.mean() / ra.std(), rb.mean() / rb.std()
    rho = float(np.corrcoef(ra, rb)[0, 1])
    var = (2 - 2 * rho + 0.5 * (sa ** 2 + sb ** 2 - 2 * sa * sb * rho ** 2)) / t
    z = (sa - sb) / np.sqrt(var) if var > 0 else np.nan
    k = np.sqrt(periods_per_year)
    return {"sr_a": sa * k, "sr_b": sb * k, "diff": (sa - sb) * k, "z": float(z),
            "p_value": float(2 * (1 - stats.norm.cdf(abs(z)))) if np.isfinite(z) else np.nan, "rho": rho, "n": t}


def stationary_bootstrap_indices(n: int, mean_block: float, rng, n_draws: int | None = None) -> np.ndarray:
    """Politis-Romano stationary bootstrap: blocks start at random points and have geometric lengths
    (mean ``mean_block``), wrapping around.  Returns (n,) or (n_draws, n) indices."""
    draws = 1 if n_draws is None else n_draws
    idx = np.empty((draws, n), dtype=int)
    idx[:, 0] = rng.integers(n, size=draws)
    new_block = rng.random((draws, n)) < 1.0 / mean_block
    jumps = rng.integers(n, size=(draws, n))
    for i in range(1, n):
        idx[:, i] = np.where(new_block[:, i], jumps[:, i], (idx[:, i - 1] + 1) % n)
    return idx[0] if n_draws is None else idx


def sharpe_diff_bootstrap(a: pd.Series, b: pd.Series, n_boot: int = 2000, mean_block: float = 6.0,
                          periods_per_year: int = 12, seed: int = 0) -> dict:
    """Paired stationary bootstrap of SR_a - SR_b: 95% interval and a two-sided p-value (centred)."""
    x = pd.concat([a, b], axis=1).dropna().values
    rng = np.random.default_rng(seed)
    def sr(v):
        s = v.std(axis=0, ddof=1)
        return v.mean(axis=0) / np.where(s > 0, s, np.nan)
    base = sr(x)
    diff0 = base[0] - base[1]
    idx = stationary_bootstrap_indices(len(x), mean_block, rng, n_boot)
    v = x[idx]                                   # n_boot x n x 2
    s = v.mean(axis=1) / v.std(axis=1, ddof=1)
    diffs = s[:, 0] - s[:, 1]
    k = np.sqrt(periods_per_year)
    centred = diffs - diffs.mean()
    p = float(np.mean(np.abs(centred) >= abs(diff0)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": diff0 * k, "ci_low": lo * k, "ci_high": hi * k, "p_value": p, "n": len(x)}


def deflated_sharpe(sr_annual: float, n_trials: int, n_obs: int, skew: float = 0.0, kurt: float = 3.0,
                    sr_trials_std_annual: float | None = None, periods_per_year: int = 12) -> dict:
    """Probability that the true Sharpe exceeds the luck benchmark from ``n_trials`` attempts.

    ``kurt`` is raw (normal = 3).  The benchmark SR0 = std(SR across trials) x ((1-g) Z(1-1/N) + g Z(1-1/(N e)))
    with g the Euler-Mascheroni constant.
    """
    k = np.sqrt(periods_per_year)
    sr = sr_annual / k
    sd = (sr_trials_std_annual / k) if sr_trials_std_annual is not None else 1.0 / np.sqrt(n_obs)
    g = 0.5772156649
    n = max(int(n_trials), 2)
    sr0 = sd * ((1 - g) * stats.norm.ppf(1 - 1 / n) + g * stats.norm.ppf(1 - 1 / (n * np.e)))
    denom = np.sqrt(max(1e-12, 1 - skew * sr + (kurt - 1) / 4 * sr ** 2))
    z = (sr - sr0) * np.sqrt(n_obs - 1) / denom
    return {"sr0_annual": sr0 * k, "dsr": float(stats.norm.cdf(z)), "z": float(z)}


def summarize(results: dict, rf: pd.Series | None = None, benchmark: str | None = None) -> pd.DataFrame:
    """Table of stats for a dict name -> BacktestResult, with tests against ``benchmark``."""
    rows = {}
    for name, res in results.items():
        s = res.stats()
        if benchmark and benchmark in results and name != benchmark:
            t = sharpe_diff_jkm(res.excess, results[benchmark].excess)
            s["sr_diff_vs_bench"] = t["diff"]
            s["p_jkm"] = t["p_value"]
        rows[name] = s
    return pd.DataFrame(rows).T
