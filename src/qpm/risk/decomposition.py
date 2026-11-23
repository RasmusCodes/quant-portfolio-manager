"""Where portfolio risk comes from: positions, factors and concentration."""
from __future__ import annotations

import numpy as np
import pandas as pd


def contributions(weights, cov) -> pd.DataFrame:
    """Marginal and component contributions to volatility (Euler decomposition; components sum to vol)."""
    names = list(weights.index) if isinstance(weights, pd.Series) else None
    w = np.asarray(weights, dtype=float)
    cov = cov.loc[names, names].values if isinstance(cov, pd.DataFrame) and names else np.asarray(cov, dtype=float)
    sigma_w = cov @ w
    vol = float(np.sqrt(max(w @ sigma_w, 0.0)))
    mcr = sigma_w / vol if vol > 0 else np.zeros_like(w)
    ccr = w * mcr
    frame = pd.DataFrame({"weight": w, "marginal": mcr, "component": ccr,
                          "share": ccr / vol if vol > 0 else 0.0}, index=names)
    frame.attrs["volatility"] = vol
    return frame


def component_es(weights, scenarios, alpha: float = 0.975) -> pd.Series:
    """Historical component expected shortfall: each position's average loss on the tail days of the
    portfolio.  Components sum to the portfolio's historical ES."""
    names = list(weights.index) if isinstance(weights, pd.Series) else None
    w = np.asarray(weights, dtype=float)
    x = np.asarray(scenarios, dtype=float)
    port = x @ w
    cut = np.quantile(port, 1 - alpha)
    tail = port <= cut
    comp = -(x[tail] * w).mean(axis=0)
    return pd.Series(comp, index=names)


def factor_decomposition(weights: pd.Series, asset_returns: pd.DataFrame, factor_returns: pd.DataFrame,
                         annualize: float = 12.0) -> dict:
    """Regress each asset on the factors, aggregate to portfolio exposures, split variance into factor
    contributions (Euler) and the idiosyncratic rest."""
    data = asset_returns.join(factor_returns, how="inner").dropna()
    a = data[asset_returns.columns].values
    f = data[factor_returns.columns].values
    design = np.column_stack([np.ones(len(f)), f])
    coef, *_ = np.linalg.lstsq(design, a, rcond=None)
    betas = coef[1:].T
    resid = a - design @ coef
    w = weights.reindex(asset_returns.columns).fillna(0.0).values
    expo = betas.T @ w
    cov_f = np.cov(f, rowvar=False) * annualize
    resid_var = np.var(resid, axis=0, ddof=design.shape[1]) * annualize
    factor_var = float(expo @ cov_f @ expo)
    contrib = expo * (cov_f @ expo)
    idio = float(np.sum(w ** 2 * resid_var))
    total = factor_var + idio
    return {
        "betas": pd.DataFrame(betas, index=asset_returns.columns, columns=factor_returns.columns),
        "exposures": pd.Series(expo, index=factor_returns.columns),
        "factor_contrib": pd.Series(contrib / total if total > 0 else contrib, index=factor_returns.columns),
        "idiosyncratic_share": idio / total if total > 0 else np.nan,
        "systematic_share": factor_var / total if total > 0 else np.nan,
        "volatility": float(np.sqrt(total)),
    }


def concentration(weights, cov=None) -> dict:
    """HHI, effective number of positions, top-5 share, diversification ratio (Choueifaty-Coignard) and
    effective number of uncorrelated bets (Meucci 2009: entropy of principal-component risk shares)."""
    w = np.asarray(weights, dtype=float)
    gross = np.abs(w).sum()
    share = np.abs(w) / gross if gross > 0 else np.zeros_like(w)
    hhi = float(np.sum(share ** 2))
    out = {"hhi": hhi, "effective_n": 1.0 / hhi if hhi > 0 else np.nan,
           "max_weight": float(np.abs(w).max()) if w.size else np.nan,
           "top5_share": float(np.sort(share)[::-1][:5].sum())}
    if cov is not None:
        cov = np.asarray(cov, dtype=float)
        vol = float(np.sqrt(max(w @ cov @ w, 0.0)))
        out["diversification_ratio"] = float(np.abs(w) @ np.sqrt(np.diag(cov)) / vol) if vol > 0 else np.nan
        vals, vecs = np.linalg.eigh(cov)
        pc_w = vecs.T @ w
        risk = pc_w ** 2 * vals
        p = risk / risk.sum() if risk.sum() > 0 else risk
        p = p[p > 1e-15]
        out["effective_bets"] = float(np.exp(-np.sum(p * np.log(p)))) if p.size else np.nan
    return out
