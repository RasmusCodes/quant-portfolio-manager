import numpy as np
import pytest

from qpm.estimation import (bayes_stein, equilibrium_returns, ewma_cov, factor_model_cov, james_stein_grand_mean,
                            ledoit_wolf_constant_correlation, ledoit_wolf_identity, nearest_psd, pca_factor_cov,
                            sample_cov)


@pytest.fixture
def x(rng):
    t, n = 120, 10
    f = rng.normal(0, 0.04, (t, 1))
    return f @ rng.uniform(0.5, 1.5, (1, n)) + rng.normal(0, 0.03, (t, n))


def test_lw_identity_matches_sklearn(x):
    sk = pytest.importorskip("sklearn.covariance")
    s, d = ledoit_wolf_identity(x)
    s2, d2 = sk.ledoit_wolf(x)
    assert np.allclose(s, s2, atol=1e-15) and abs(d - d2) < 1e-12


def _covcor_reference(x):
    """Loop transcription of Ledoit-Wolf (2003) covCor.m, independent of the vectorised version."""
    t, n = x.shape
    xc = x - x.mean(0)
    s = xc.T @ xc / t
    var = np.diag(s)
    sd = np.sqrt(var)
    rbar = sum(s[i, j] / (sd[i] * sd[j]) for i in range(n) for j in range(n) if i != j) / (n * (n - 1))
    f = np.array([[s[i, i] if i == j else rbar * sd[i] * sd[j] for j in range(n)] for i in range(n)])
    pi = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            pi[i, j] = np.mean((xc[:, i] * xc[:, j] - s[i, j]) ** 2)
    theta_ii = np.zeros((n, n))
    theta_jj = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            theta_ii[i, j] = np.mean((xc[:, i] ** 2 - s[i, i]) * (xc[:, i] * xc[:, j] - s[i, j]))
            theta_jj[i, j] = np.mean((xc[:, j] ** 2 - s[j, j]) * (xc[:, i] * xc[:, j] - s[i, j]))
    rho = np.trace(pi) + sum(rbar / 2 * (np.sqrt(var[j] / var[i]) * theta_ii[i, j] + np.sqrt(var[i] / var[j]) * theta_jj[i, j])
                             for i in range(n) for j in range(n) if i != j)
    gamma = np.sum((f - s) ** 2)
    kappa = (pi.sum() - rho) / gamma
    d = max(0, min(1, kappa / t))
    return d * f + (1 - d) * s, d


def test_lw_constant_correlation_matches_loop_reference(x):
    s, d = ledoit_wolf_constant_correlation(x)
    s2, d2 = _covcor_reference(x)
    assert abs(d - d2) < 1e-10 and np.allclose(s, s2, atol=1e-14)
    assert 0 <= d <= 1


def test_ewma_long_halflife_approaches_sample(x):
    assert np.allclose(ewma_cov(x, 1e7), sample_cov(x), rtol=1e-4, atol=1e-10)


def test_ewma_weights_recent_rows_more(rng):
    calm = rng.normal(0, 0.01, (100, 3))
    wild = rng.normal(0, 0.05, (20, 3))
    x = np.vstack([calm, wild])
    assert np.trace(ewma_cov(x, 10)) > np.trace(sample_cov(x))


def test_factor_model_recovers_betas(rng):
    t = 2000
    f = rng.normal(0, 0.04, (t, 2))
    b = np.array([[1.0, 0.2], [0.5, -0.3], [1.5, 0.8]])
    x = f @ b.T + rng.normal(0, 0.01, (t, 3))
    cov, betas = factor_model_cov(x, f)
    assert np.allclose(betas, b, atol=0.05)
    assert np.all(np.linalg.eigvalsh(cov) > 0)


def test_pca_and_nearest_psd(x):
    c = pca_factor_cov(x, 2)
    assert np.all(np.linalg.eigvalsh(c) > 0)
    bad = sample_cov(x)
    bad[0, 1] = bad[1, 0] = 10 * np.sqrt(bad[0, 0] * bad[1, 1])
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() > 0
    good = sample_cov(x)
    assert np.allclose(nearest_psd(good), good)


def test_bayes_stein_shrinks_toward_min_variance_mean(x):
    mu_bs, w = bayes_stein(x)
    mu = x.mean(0)
    assert 0 <= w <= 1
    assert np.std(mu_bs) <= np.std(mu) + 1e-15


def test_james_stein_grand_mean(x):
    mu_js, w = james_stein_grand_mean(x)
    assert 0 <= w <= 1
    assert abs(mu_js.mean() - x.mean(0).mean()) < 1e-12


def test_equilibrium_returns_reverse_optimize(rng):
    from conftest import random_cov
    s = random_cov(rng, 5)
    w = np.array([0.3, 0.2, 0.2, 0.2, 0.1])
    pi = equilibrium_returns(s, w, 2.5)
    # mean-variance with these returns and the same risk aversion returns w (budget-free)
    assert np.allclose(np.linalg.solve(2.5 * s, pi), w)
