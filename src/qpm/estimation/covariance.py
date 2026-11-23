"""Covariance estimators.

All functions take a T x n array of returns (rows = periods) and return an n x n matrix in the same
units (multiply by 12 for annual units from monthly data).  Shrinkage estimators return the matrix and
the shrinkage intensity.

References
----------
Ledoit, O. and Wolf, M. (2003) "Improved estimation of the covariance matrix of stock returns with an
application to portfolio selection", J. Empirical Finance (constant-correlation target).
Ledoit, O. and Wolf, M. (2004) "A well-conditioned estimator for large-dimensional covariance matrices",
J. Multivariate Analysis (scaled-identity target).
RiskMetrics Technical Document (1996) for exponentially weighted estimates.
"""
from __future__ import annotations

import numpy as np


def _as2d(returns) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be a T x n array")
    if np.isnan(x).any():
        raise ValueError("returns contain NaN; drop or fill them before estimating")
    return x


def sample_cov(returns, ddof: int = 1) -> np.ndarray:
    x = _as2d(returns)
    xc = x - x.mean(axis=0)
    return xc.T @ xc / (x.shape[0] - ddof)


def ewma_cov(returns, halflife: float, demean: bool = True) -> np.ndarray:
    """Exponentially weighted covariance; the most recent row gets the largest weight.

    ``halflife`` is in periods (RiskMetrics' daily lambda 0.94 is a half-life of about 11 days).
    """
    x = _as2d(returns)
    t = x.shape[0]
    lam = 0.5 ** (1.0 / halflife)
    w = lam ** np.arange(t - 1, -1, -1, dtype=float)
    w /= w.sum()
    mean = w @ x if demean else np.zeros(x.shape[1])
    xc = x - mean
    cov = (xc * w[:, None]).T @ xc
    if demean:  # bias correction for weighted samples
        cov /= (1.0 - np.sum(w ** 2))
    return cov


def ledoit_wolf_identity(returns) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage toward a scaled identity. Matches sklearn.covariance.ledoit_wolf."""
    x = _as2d(returns)
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    mu = np.trace(s) / n
    x2 = xc ** 2
    beta_bar = np.sum((x2.T @ x2) / t - s ** 2) / t      # sum_t ||x_t x_t' - S||_F^2 / t^2
    delta2 = np.sum(s ** 2) - 2 * mu * np.trace(s) + n * mu ** 2   # ||S - mu I||_F^2
    beta = min(beta_bar, delta2)
    shrink = 0.0 if delta2 == 0 else beta / delta2
    return shrink * mu * np.eye(n) + (1 - shrink) * s, float(shrink)


def ledoit_wolf_constant_correlation(returns) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2003) shrinkage toward the constant-correlation matrix (their covCor estimator)."""
    x = _as2d(returns)
    t, n = x.shape
    xc = x - x.mean(axis=0)
    s = xc.T @ xc / t
    var = np.diag(s).copy()
    sd = np.sqrt(var)
    corr = s / np.outer(sd, sd)
    rbar = (np.sum(corr) - n) / (n * (n - 1))
    prior = rbar * np.outer(sd, sd)
    np.fill_diagonal(prior, var)
    # pi-hat: asymptotic variances of the sample covariances
    y = xc ** 2
    phi_mat = y.T @ y / t - s ** 2
    phi = phi_mat.sum()
    # rho-hat
    term1 = (xc ** 3).T @ xc / t
    help_ = xc.T @ xc / t
    help_diag = np.diag(help_)
    term2 = help_diag[:, None] * s
    theta = term1 - term2
    np.fill_diagonal(theta, 0.0)
    rho = np.sum(np.diag(phi_mat)) + rbar * np.sum(np.outer(1.0 / sd, sd) * theta)
    gamma = np.linalg.norm(s - prior, "fro") ** 2
    kappa = (phi - rho) / gamma if gamma > 0 else 0.0
    shrink = float(max(0.0, min(1.0, kappa / t)))
    return shrink * prior + (1 - shrink) * s, shrink


def factor_model_cov(returns, factors, ddof: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Observable-factor covariance B Sigma_f B' + D from time-series regressions on ``factors``.

    Returns (covariance, betas n x k).  Residual covariances are set to zero (D diagonal).
    """
    x = _as2d(returns)
    f = _as2d(factors)
    if f.shape[0] != x.shape[0]:
        raise ValueError("returns and factors need the same number of rows")
    design = np.column_stack([np.ones(len(f)), f])
    coef, *_ = np.linalg.lstsq(design, x, rcond=None)
    betas = coef[1:].T
    resid = x - design @ coef
    dof = max(1, x.shape[0] - design.shape[1])
    d = np.sum(resid ** 2, axis=0) / dof
    cov_f = sample_cov(f, ddof)
    return betas @ cov_f @ betas.T + np.diag(d), betas


def pca_factor_cov(returns, k: int) -> np.ndarray:
    """Statistical factor model: top-k principal components of the sample covariance + diagonal rest."""
    s = sample_cov(returns)
    vals, vecs = np.linalg.eigh(s)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    k = int(max(1, min(k, len(vals) - 1)))
    common = (vecs[:, :k] * vals[:k]) @ vecs[:, :k].T
    resid = np.clip(np.diag(s) - np.diag(common), 1e-12, None)
    return common + np.diag(resid)


def nearest_psd(matrix, floor: float = 1e-10) -> np.ndarray:
    """Symmetrise and clip eigenvalues at ``floor`` x the largest one (keeps user-edited inputs valid)."""
    m = np.asarray(matrix, dtype=float)
    m = 0.5 * (m + m.T)
    vals, vecs = np.linalg.eigh(m)
    lo = floor * max(vals.max(), 1e-300)
    if vals.min() >= lo:
        return m
    vals = np.clip(vals, lo, None)
    out = (vecs * vals) @ vecs.T
    return 0.5 * (out + out.T)


def cov_to_corr(cov) -> tuple[np.ndarray, np.ndarray]:
    cov = np.asarray(cov, dtype=float)
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(sd, sd)
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return corr, sd


def corr_to_cov(corr, sd) -> np.ndarray:
    sd = np.asarray(sd, dtype=float)
    return np.asarray(corr, dtype=float) * np.outer(sd, sd)


ESTIMATORS = ("sample", "ewma", "lw_identity", "lw_cc", "factor", "pca")


def estimate_cov(returns, method: str = "lw_cc", factors=None, halflife: float = 36.0, k: int = 3) -> np.ndarray:
    """Dispatch by name. ``factor`` needs ``factors`` (same rows as returns)."""
    if method == "sample":
        return sample_cov(returns)
    if method == "ewma":
        return ewma_cov(returns, halflife)
    if method == "lw_identity":
        return ledoit_wolf_identity(returns)[0]
    if method == "lw_cc":
        return ledoit_wolf_constant_correlation(returns)[0]
    if method == "factor":
        if factors is None:
            raise ValueError("factor covariance needs factor returns")
        return factor_model_cov(returns, factors)[0]
    if method == "pca":
        return pca_factor_cov(returns, k)
    raise KeyError(f"unknown covariance estimator {method!r}; choose from {ESTIMATORS}")
