"""A synthetic signal laboratory where the truth is known.

Real signal ICs are small and noisy, so on real data one cannot tell whether a combination method lost
because it is bad or because it was unlucky.  Here returns are generated as

    r_{i,t+1} = sum_k c_{k,t} z_{k,i,t} + e_{i,t+1},   z ~ N(0, C) across signals, e ~ N(0, 1)

with chosen ICs c_k (the IC of signal k is close to c_k when the c's are small), optional time variation
(AR(1) noise around c_k), and optional regimes in which some signals flip sign.  The oracle combination
weights signals by their true c_{k,t}.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def simulate(n_assets: int = 50, n_periods: int = 360, ics=(0.06, 0.04, 0.02, 0.0, 0.0), signal_corr: float = 0.3,
             ic_noise: float = 0.0, ic_persistence: float = 0.9, regime_flip: tuple[int, ...] = (),
             regime_stay: float = 0.97, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    ics = np.asarray(ics, dtype=float)
    k = ics.size
    corr = (1 - signal_corr) * np.eye(k) + signal_corr * np.ones((k, k))
    chol = np.linalg.cholesky(corr)
    dates = pd.date_range("1990-01-31", periods=n_periods, freq="ME")
    names = [f"sig{j + 1}" for j in range(k)]
    assets = [f"x{i:03d}" for i in range(n_assets)]
    z = rng.standard_normal((n_periods, n_assets, k)) @ chol.T
    # time-varying coefficients
    noise = np.zeros((n_periods, k))
    for t in range(1, n_periods):
        noise[t] = ic_persistence * noise[t - 1] + np.sqrt(1 - ic_persistence ** 2) * rng.standard_normal(k)
    coef = ics[None, :] + ic_noise * noise
    regime = np.zeros(n_periods, dtype=int)
    if regime_flip:
        for t in range(1, n_periods):
            regime[t] = regime[t - 1] if rng.random() < regime_stay else 1 - regime[t - 1]
        flip = np.array([j in regime_flip for j in range(k)])
        coef[regime == 1] = np.where(flip, -coef[regime == 1], coef[regime == 1])
    eps = rng.standard_normal((n_periods, n_assets))
    fwd = np.einsum("tik,tk->ti", z, coef) + eps
    zs = {names[j]: pd.DataFrame(z[:, :, j], index=dates, columns=assets) for j in range(k)}
    oracle = pd.DataFrame(np.einsum("tik,tk->ti", z, coef), index=dates, columns=assets)
    return {"zs": zs, "fwd": pd.DataFrame(fwd, index=dates, columns=assets),
            "coef": pd.DataFrame(coef, index=dates, columns=names), "oracle": oracle,
            "regime": pd.Series(regime, index=dates, name="regime")}
