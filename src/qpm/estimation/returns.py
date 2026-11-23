"""Expected-return estimators.

Means are the weakest input of any optimizer: with monthly data the standard error of a mean return is
about sigma / sqrt(T), so ten years of an asset with 18% volatility pins its mean down only to +/- 5.7%
a year.  The estimators below trade that noise against bias.

References
----------
Jorion, P. (1986) "Bayes-Stein estimation for portfolio analysis", JFQA.
Black, F. and Litterman, R. (1992) "Global portfolio optimization", FAJ (equilibrium returns).
Grinold, R. and Kahn, R. (2000) "Active Portfolio Management" (alpha = IC x volatility x score).
"""
from __future__ import annotations

import numpy as np

from .covariance import sample_cov


def sample_mean(returns) -> np.ndarray:
    return np.asarray(returns, dtype=float).mean(axis=0)


def bayes_stein(returns, cov=None) -> tuple[np.ndarray, float]:
    """Jorion's Bayes-Stein estimator: shrink sample means toward the minimum-variance portfolio's mean.

    Returns (means, shrinkage weight). The weight w = (n+2) / ((n+2) + T d' S^-1 d) with d = mu - mu0 1,
    using S = sample covariance x (T-1)/(T-n-2) as Jorion does.
    """
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    mu = x.mean(axis=0)
    s = sample_cov(x) if cov is None else np.asarray(cov, dtype=float)
    if cov is None and t > n + 2:
        s = s * (t - 1) / (t - n - 2)
    inv = np.linalg.pinv(s)
    ones = np.ones(n)
    mu0 = float(ones @ inv @ mu / (ones @ inv @ ones))
    d = mu - mu0
    denom = (n + 2) + t * float(d @ inv @ d)
    w = float(min(1.0, (n + 2) / denom)) if denom > 0 else 1.0
    return (1 - w) * mu + w * mu0, w


def james_stein_grand_mean(returns) -> tuple[np.ndarray, float]:
    """Positive-part James-Stein shrinkage of the means toward their cross-sectional average."""
    x = np.asarray(returns, dtype=float)
    t, n = x.shape
    mu = x.mean(axis=0)
    grand = mu.mean()
    se2 = np.mean(x.var(axis=0, ddof=1)) / t       # average sampling variance of a mean
    spread = np.sum((mu - grand) ** 2)
    w = 1.0 if spread <= 0 else float(min(1.0, max(0.0, (n - 3) * se2 / spread)))
    return (1 - w) * mu + w * grand, w


def equilibrium_returns(cov, market_weights, risk_aversion: float = 2.5) -> np.ndarray:
    """Reverse optimization: the excess returns that make ``market_weights`` mean-variance optimal."""
    w = np.asarray(market_weights, dtype=float)
    w = w / w.sum()
    return risk_aversion * np.asarray(cov, dtype=float) @ w


def implied_risk_aversion(market_excess_return: float, market_variance: float) -> float:
    return market_excess_return / market_variance


def grinold_alpha(score, volatility, ic: float) -> np.ndarray:
    """Alpha = IC x volatility x standardized score (score should be cross-sectionally z-scored)."""
    return ic * np.asarray(volatility, dtype=float) * np.asarray(score, dtype=float)


def estimate_mean(returns, method: str = "sample", cov=None) -> np.ndarray:
    if method == "sample":
        return sample_mean(returns)
    if method == "bayes_stein":
        return bayes_stein(returns, cov)[0]
    if method == "james_stein":
        return james_stein_grand_mean(returns)[0]
    if method == "zero":
        return np.zeros(np.asarray(returns).shape[1])
    raise KeyError(f"unknown mean estimator {method!r}")
