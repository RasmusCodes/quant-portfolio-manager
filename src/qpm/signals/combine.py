"""Combining signals from several research systems into one score, walk-forward.

Every method sees, at decision date t, only ICs and returns realised by t: the IC stamped t-1 uses the
return over t-1..t, so rows up to t-1 of the IC table are usable at t.  Combined scores are z-scored
across instruments each date, so methods differ only in how they weight the inputs.

Methods
-------
equal        1/K over the signals available that date (the baseline every other method must beat)
ic_weighted  "dynamic confidence": weights proportional to the positive part of an EWMA of past ICs
shrunk       past mean ICs shrunk toward their cross-signal average by empirical Bayes (Morris 1983)
mean_var     weights proportional to Cov(IC)^-1 mean(IC) (maximises the combined IR), long-only
bma          Bayesian model averaging over every subset of signals in a pooled cross-sectional
             regression with Zellner's g-prior (unit information, g = n)
regime       mean IC within the current regime, shrunk toward the unconditional mean
ridge        stacked ensemble: pooled ridge regression of next returns on all signals, penalty picked
             by blocked cross-validation inside the trailing window
confidence   static weights from the producers' own confidence numbers
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .normalize import zscore
from .performance import cross_sectional_ic

METHODS = ("equal", "ic_weighted", "shrunk", "mean_var", "bma", "regime", "ridge", "confidence")


@dataclass
class Combination:
    method: str
    scores: pd.DataFrame
    weights: pd.DataFrame
    info: dict = field(default_factory=dict)


def align(zs: dict[str, pd.DataFrame], fwd: pd.DataFrame | None = None):
    dates = sorted(set.intersection(*[set(z.index) for z in zs.values()]))
    instruments = sorted(set.union(*[set(z.columns) for z in zs.values()]))
    idx = pd.DatetimeIndex(dates)
    zs = {k: v.reindex(index=idx, columns=instruments) for k, v in zs.items()}
    if fwd is not None:
        fwd = fwd.reindex(index=idx, columns=instruments)
    return zs, fwd, idx, instruments


def ic_table(zs: dict[str, pd.DataFrame], fwd: pd.DataFrame, method: str = "spearman") -> pd.DataFrame:
    return pd.DataFrame({k: cross_sectional_ic(z, fwd, method) for k, z in zs.items()}).reindex(fwd.index)


def _normalise(w: np.ndarray, available: np.ndarray, fallback: str) -> np.ndarray:
    w = np.where(available, np.maximum(w, 0.0), 0.0)
    if w.sum() > 0:
        return w / w.sum()
    if fallback == "equal" and available.any():
        return available / available.sum()
    return np.zeros_like(w)


def _morris_shrink(means: np.ndarray, sampling_var: np.ndarray) -> np.ndarray:
    ok = np.isfinite(means) & np.isfinite(sampling_var)
    if ok.sum() < 2:
        return means
    m, v = means[ok], sampling_var[ok]
    grand = m.mean()
    tau2 = max(0.0, m.var(ddof=1) - v.mean())
    b = v / (v + tau2) if tau2 > 0 else np.ones_like(v)
    out = means.copy()
    out[ok] = grand + (1 - b) * (m - grand)
    return out


def _panel_xy(zarr, fwd_arr, rows):
    """Stack rows of the z panels (dates x instruments x K) and next returns (demeaned per date) into
    X (obs x K) and y; n_eff counts one observation per instrument per date, less one for the demeaning."""
    xs, ys, n_eff = [], [], 0
    for r in rows:
        y = fwd_arr[r]
        x = zarr[r]
        ok = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
        if ok.sum() < 3:
            continue
        y = y[ok] - y[ok].mean()
        x = x[ok] - x[ok].mean(axis=0)
        xs.append(x)
        ys.append(y)
        n_eff += ok.sum() - 1
    if not xs:
        return None, None, 0
    return np.vstack(xs), np.concatenate(ys), n_eff


def _bma(x, y, n_eff, g=None):
    k = x.shape[1]
    g = float(n_eff) if g is None else g
    tss = float(y @ y)
    if tss <= 0:
        return np.zeros(k), np.zeros(k)
    logs, coefs, members = [0.0], [np.zeros(k)], [()]
    for size in range(1, k + 1):
        for subset in itertools.combinations(range(k), size):
            xs = x[:, subset]
            beta, *_ = np.linalg.lstsq(xs, y, rcond=None)
            r2 = 1.0 - float(np.sum((y - xs @ beta) ** 2)) / tss
            r2 = min(max(r2, 0.0), 1.0 - 1e-12)
            log_bf = 0.5 * (n_eff - 1 - size) * np.log1p(g) - 0.5 * (n_eff - 1) * np.log1p(g * (1 - r2))
            full = np.zeros(k)
            full[list(subset)] = beta * g / (1 + g)
            logs.append(log_bf)
            coefs.append(full)
            members.append(subset)
    logs = np.array(logs)
    post = np.exp(logs - logs.max())
    post /= post.sum()
    coef = np.sum(post[:, None] * np.array(coefs), axis=0)
    incl = np.array([sum(p for p, m in zip(post, members) if j in m) for j in range(k)])
    return coef, incl


def _ridge_cv(x_rows, y_rows, alphas, folds=5):
    """Blocked CV over contiguous groups of dates; returns the best alpha."""
    n_dates = len(x_rows)
    if n_dates < folds * 2:
        return alphas[len(alphas) // 2]
    edges = np.linspace(0, n_dates, folds + 1).astype(int)
    errs = np.zeros(len(alphas))
    for f in range(folds):
        test = range(edges[f], edges[f + 1])
        train = [i for i in range(n_dates) if i < edges[f] or i >= edges[f + 1]]
        xt = np.vstack([x_rows[i] for i in train])
        yt = np.concatenate([y_rows[i] for i in train])
        xv = np.vstack([x_rows[i] for i in test])
        yv = np.concatenate([y_rows[i] for i in test])
        xtx, xty = xt.T @ xt, xt.T @ yt
        for a_i, a in enumerate(alphas):
            beta = np.linalg.solve(xtx + a * len(yt) * np.eye(xt.shape[1]), xty)
            errs[a_i] += np.sum((yv - xv @ beta) ** 2)
    return alphas[int(np.argmin(errs))]


def combine(zs: dict[str, pd.DataFrame], fwd: pd.DataFrame, method: str = "equal", *, halflife: float = 36,
            min_history: int = 24, window: int = 120, fallback: str = "equal", regimes: pd.Series | None = None,
            regime_prior: float = 24.0, confidences: dict | None = None, ic_method: str = "spearman",
            ridge_alphas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0), bma_g: float | None = None) -> Combination:
    """Walk-forward combination. ``fwd`` row t = return over t..t+1 (see performance.forward_returns)."""
    if method not in METHODS:
        raise KeyError(f"unknown combination method {method!r}; choose from {METHODS}")
    zs, fwd, idx, instruments = align(zs, fwd)
    names = list(zs)
    k = len(names)
    zl = [zs[n] for n in names]
    ic = ic_table(zs, fwd, ic_method) if method not in ("equal", "confidence") else None
    weights = np.zeros((len(idx), k))
    scores = np.full((len(idx), len(instruments)), np.nan)
    info: dict = {"inclusion": {}, "alpha": {}}
    zarr = np.stack([z.values for z in zl], axis=2)  # dates x instruments x K
    fwd_arr = fwd.values

    for p in range(len(idx)):
        avail = np.array([np.isfinite(zarr[p, :, j]).sum() >= 3 for j in range(k)])
        if not avail.any():
            continue
        hist = ic.iloc[:p] if ic is not None else None  # ICs realised by date p
        enough = hist is not None and hist.notna().sum().min() >= min_history
        if method == "equal" or (method not in ("confidence",) and not enough and method != "equal"):
            w = avail / avail.sum()
        elif method == "confidence":
            c = np.array([float((confidences or {}).get(n, 1.0)) for n in names])
            w = _normalise(c, avail, fallback)
        elif method == "ic_weighted":
            ew = hist.ewm(halflife=halflife, min_periods=1).mean().iloc[-1].values
            w = _normalise(ew, avail, fallback)
        elif method == "shrunk":
            means = hist.mean().values
            svar = (hist.var().values / hist.notna().sum().values)
            w = _normalise(_morris_shrink(means, svar), avail, fallback)
        elif method == "mean_var":
            h = hist.dropna()
            if len(h) < min_history:
                w = avail / avail.sum()
            else:
                c = h.cov().values
                c = 0.5 * c + 0.5 * np.diag(np.diag(c))  # shrink correlations halfway to zero
                w = _normalise(np.linalg.solve(c + 1e-12 * np.eye(k), h.mean().values), avail, fallback)
        elif method == "regime":
            if regimes is None:
                raise ValueError("regime weighting needs a regimes series")
            reg = regimes.reindex(idx)
            cur = reg.iloc[p]
            past = hist.copy()
            same = (reg.iloc[:p] == cur).values
            m_all = past.mean().values
            n_s = past[same].notna().sum().values
            m_s = np.nan_to_num(past[same].mean().values, nan=0.0)
            m = (n_s * m_s + regime_prior * m_all) / (n_s + regime_prior)
            w = _normalise(m, avail, fallback)
        elif method in ("bma", "ridge"):
            rows = list(range(max(0, p - window), p))
            if method == "bma":
                x, y, n_eff = _panel_xy(zarr, fwd_arr, rows)
                if x is None:
                    w = avail / avail.sum()
                else:
                    coef, incl = _bma(x, y, n_eff, bma_g)
                    info["inclusion"][idx[p]] = incl
                    w = coef  # regression coefficients: signed, in return units per unit z
            else:
                x_rows, y_rows = [], []
                for r in rows:
                    xr, yr, _ = _panel_xy(zarr, fwd_arr, [r])
                    if xr is not None:
                        x_rows.append(xr)
                        y_rows.append(yr)
                if len(x_rows) < 10:
                    w = avail / avail.sum()
                else:
                    a = _ridge_cv(x_rows, y_rows, list(ridge_alphas))
                    xt, yt = np.vstack(x_rows), np.concatenate(y_rows)
                    w = np.linalg.solve(xt.T @ xt + a * len(yt) * np.eye(k), xt.T @ yt)
                    info["alpha"][idx[p]] = a
        weights[p] = w
        combo = np.nansum(zarr[p] * w[None, :], axis=1)
        combo[~np.any(np.isfinite(zarr[p]), axis=1)] = np.nan
        if np.allclose(w, 0):
            scores[p] = np.where(np.isfinite(combo), 0.0, np.nan)
        else:
            scores[p] = zscore(pd.Series(combo)).values
    return Combination(method, pd.DataFrame(scores, index=idx, columns=instruments),
                       pd.DataFrame(weights, index=idx, columns=names), info)
