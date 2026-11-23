"""Value at risk and expected shortfall.

Losses are reported as positive numbers (a 99% VaR of 0.05 means a 5% loss is exceeded on 1% of days).
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def historical_var(returns, alpha: float = 0.99) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    return float(-np.quantile(r, 1.0 - alpha))


def historical_es(returns, alpha: float = 0.99) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    cut = np.quantile(r, 1.0 - alpha)
    tail = r[r <= cut]
    return float(-tail.mean()) if tail.size else float(-cut)


def parametric_var(mu: float, sigma: float, alpha: float = 0.99, dist: str = "normal", df: float = 5.0) -> float:
    if dist == "t":  # Student t scaled to unit variance
        q = stats.t.ppf(1 - alpha, df) * np.sqrt((df - 2) / df)
    else:
        q = stats.norm.ppf(1 - alpha)
    return float(-(mu + q * sigma))


def parametric_es(mu: float, sigma: float, alpha: float = 0.99, dist: str = "normal", df: float = 5.0) -> float:
    if dist == "t":
        x = stats.t.ppf(1 - alpha, df)
        es_std = stats.t.pdf(x, df) / (1 - alpha) * (df + x ** 2) / (df - 1) * np.sqrt((df - 2) / df)
        return float(-mu + sigma * es_std)
    z = stats.norm.ppf(1 - alpha)
    return float(-mu + sigma * stats.norm.pdf(z) / (1 - alpha))


def cornish_fisher_var(returns, alpha: float = 0.99) -> float:
    """Modified VaR (Zangari 1996): the normal quantile adjusted for sample skewness and excess kurtosis."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    mu, sd = r.mean(), r.std(ddof=1)
    s, k = stats.skew(r), stats.kurtosis(r)  # excess kurtosis
    z = stats.norm.ppf(1 - alpha)
    zcf = z + (z ** 2 - 1) * s / 6 + (z ** 3 - 3 * z) * k / 24 - (2 * z ** 3 - 5 * z) * s ** 2 / 36
    return float(-(mu + zcf * sd))


def monte_carlo_var(weights, mu, cov, alpha: float = 0.99, n_sims: int = 100_000, dist: str = "t",
                    df: float = 5.0, seed: int = 0) -> tuple[float, float]:
    """(VaR, ES) of w'r with r multivariate normal or multivariate t (unit-variance scaled)."""
    rng = np.random.default_rng(seed)
    w = np.asarray(weights, dtype=float)
    mu = np.asarray(mu, dtype=float)
    chol = np.linalg.cholesky(np.asarray(cov, dtype=float) + 1e-14 * np.eye(len(w)))
    z = rng.standard_normal((n_sims, len(w))) @ chol.T
    if dist == "t":
        g = rng.chisquare(df, size=(n_sims, 1)) / df
        z = z / np.sqrt(g) * np.sqrt((df - 2) / df)
    port = (mu + z) @ w
    return historical_var(port, alpha), historical_es(port, alpha)


def scale_horizon(value: float, days: float) -> float:
    """Square-root-of-time scaling (assumes iid returns; understates risk when losses cluster)."""
    return float(value * np.sqrt(days))


def kupiec_pof(exceptions: int, n: int, alpha: float) -> dict:
    """Kupiec (1995) proportion-of-failures likelihood-ratio test for a VaR model."""
    p = 1 - alpha
    x = exceptions
    if n == 0:
        return {"exceptions": 0, "expected": 0.0, "lr": np.nan, "p_value": np.nan}
    phat = x / n
    def ll(prob):
        prob = min(max(prob, 1e-12), 1 - 1e-12)
        return (n - x) * np.log(1 - prob) + x * np.log(prob)
    lr = -2 * (ll(p) - ll(phat))
    return {"exceptions": int(x), "expected": n * p, "rate": phat, "lr": float(lr),
            "p_value": float(1 - stats.chi2.cdf(lr, 1))}


def christoffersen_independence(hits) -> dict:
    """Christoffersen (1998) test that VaR exceptions do not cluster (first-order Markov)."""
    h = np.asarray(hits, dtype=int)
    n00 = np.sum((h[:-1] == 0) & (h[1:] == 0))
    n01 = np.sum((h[:-1] == 0) & (h[1:] == 1))
    n10 = np.sum((h[:-1] == 1) & (h[1:] == 0))
    n11 = np.sum((h[:-1] == 1) & (h[1:] == 1))
    def safe_log(x):
        return np.log(x) if x > 0 else 0.0
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = (n00 + n10) * safe_log(1 - pi) + (n01 + n11) * safe_log(pi)
    l1 = n00 * safe_log(1 - pi0) + n01 * safe_log(pi0) + n10 * safe_log(1 - pi1) + n11 * safe_log(pi1)
    lr = float(-2 * (l0 - l1))
    return {"lr": lr, "p_value": float(1 - stats.chi2.cdf(lr, 1)), "p_exception_after_exception": float(pi1),
            "p_exception_after_none": float(pi0)}


def rolling_var_backtest(returns, window: int = 500, alpha: float = 0.99, method: str = "historical") -> dict:
    """Walk-forward VaR: each day's VaR uses only the previous ``window`` days; count exceptions."""
    r = np.asarray(returns, dtype=float)
    var = np.full(r.size, np.nan)
    if method in ("ewma", "fhs"):
        # RiskMetrics variance sigma^2_t = 0.94 sigma^2_{t-1} + 0.06 r^2_{t-1} (known at t).  "ewma" uses the
        # normal quantile; "fhs" (filtered historical simulation, Barone-Adesi et al. 1999) scales the empirical
        # quantile of the past ``window`` standardised returns r/sigma by today's sigma.
        lam, z = 0.94, -stats.norm.ppf(1 - alpha)
        sig = np.full(r.size, np.nan)
        s2 = np.var(r[:window])
        for t in range(1, r.size):
            s2 = lam * s2 + (1 - lam) * r[t - 1] ** 2
            sig[t] = np.sqrt(s2)
        std = r / sig
        for t in range(window + 1, r.size):
            if method == "ewma":
                var[t] = z * sig[t]
            else:
                var[t] = -np.quantile(std[t - window:t], 1 - alpha) * sig[t]
    for t in range(window, r.size):
        if method in ("ewma", "fhs"):
            break
        past = r[t - window:t]
        if method == "historical":
            var[t] = historical_var(past, alpha)
        elif method == "normal":
            var[t] = parametric_var(past.mean(), past.std(ddof=1), alpha)
        elif method == "cornish_fisher":
            var[t] = cornish_fisher_var(past, alpha)
        else:
            raise KeyError(method)
    ok = np.isfinite(var)
    hits = (-r[ok] > var[ok]).astype(int)
    return {"var": var, "hits": hits, "kupiec": kupiec_pof(int(hits.sum()), int(hits.size), alpha),
            "independence": christoffersen_independence(hits)}
