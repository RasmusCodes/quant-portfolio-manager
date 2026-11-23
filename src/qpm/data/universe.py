"""Investable universes built from the public data, and a loader for the processed files.

Universes
---------
``multi_asset``  12 French industry portfolios + 10-year and 5-year constant-maturity Treasuries.
``industries49`` 49 French industry portfolios, grouped into 11 sectors (grouping is mine, approximate).
``industries12`` the 12 industries alone, from 1926 (used by the signal study).
``sleeves``      Eight long-short equity factor sleeves (Mkt-RF, SMB, HML, RMW, CMA, Mom, ST_Rev, LT_Rev)
                 plus a Treasury term sleeve (UST10 minus T-bills), each collateralised in T-bills, so a
                 sleeve's return is RF + factor. Each sleeve stands in for one research system's returns.

All returns are monthly simple returns in decimals.  ``rf`` is the one-month T-bill return from the
French files.  Market caps come from French's number of firms x average firm size (start of month).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PROCESSED = ROOT / "data" / "processed"

INDUSTRY12_NAMES = {
    "NoDur": "Consumer non-durables", "Durbl": "Consumer durables", "Manuf": "Manufacturing",
    "Enrgy": "Energy", "Chems": "Chemicals", "BusEq": "Business equipment (tech)", "Telcm": "Telecom",
    "Utils": "Utilities", "Shops": "Retail and services", "Hlth": "Health care", "Money": "Finance",
    "Other": "Other",
}

SECTORS49 = {
    "Staples": ["Agric", "Food", "Soda", "Beer", "Smoke", "Hshld"],
    "Discretionary": ["Toys", "Fun", "Books", "Clths", "Autos", "PerSv", "Meals", "Rtail"],
    "Health": ["Hlth", "MedEq", "Drugs"],
    "Materials": ["Chems", "Rubbr", "Txtls", "BldMt", "Steel", "Gold", "Mines", "Paper", "Boxes"],
    "Industrials": ["Cnstr", "FabPr", "Mach", "ElcEq", "Aero", "Ships", "Guns", "Trans", "BusSv", "Whlsl"],
    "Energy": ["Coal", "Oil"],
    "Utilities": ["Util"],
    "Telecom": ["Telcm"],
    "Technology": ["Hardw", "Softw", "Chips", "LabEq"],
    "Financials": ["Banks", "Insur", "RlEst", "Fin"],
    "Other": ["Other"],
}

SLEEVE_FAMILY = {
    "Mkt-RF": "Equity beta", "SMB": "Equity factor (size)", "HML": "Equity factor (value)",
    "RMW": "Equity factor (profitability)", "CMA": "Equity factor (investment)", "Mom": "Equity factor (momentum)",
    "ST_Rev": "Statistical arbitrage (short-term reversal)", "LT_Rev": "Equity factor (long-term reversal)",
    "Term": "Macro / fixed income (term premium)",
}


@dataclass
class Universe:
    name: str
    returns: pd.DataFrame                    # monthly simple returns, decimals
    rf: pd.Series                            # monthly T-bill return
    group: dict[str, str]                    # sector / asset class per asset
    labels: dict[str, str] = field(default_factory=dict)
    mcap: pd.DataFrame | None = None         # $ millions, start of month (known at the prior month end)
    bm: pd.DataFrame | None = None           # point-in-time BE/ME on month-end decision dates
    factors: pd.DataFrame | None = None      # monthly factor returns for factor risk models
    daily: pd.DataFrame | None = None        # daily simple returns (same assets where available)
    meta: dict = field(default_factory=dict)

    @property
    def assets(self) -> list[str]:
        return list(self.returns.columns)

    def window(self, start=None, end=None) -> "Universe":
        sl = slice(start, end)
        return Universe(self.name, self.returns.loc[sl], self.rf.loc[sl], self.group, self.labels,
                        None if self.mcap is None else self.mcap.loc[sl],
                        None if self.bm is None else self.bm.loc[sl],
                        None if self.factors is None else self.factors.loc[sl],
                        None if self.daily is None else self.daily.loc[sl], dict(self.meta))


def _read(name: str, folder: Path) -> pd.DataFrame:
    path = folder / f"{name}.csv"
    frame = pd.read_csv(path, index_col=0, parse_dates=True)
    frame.index.name = "date"
    return frame


def load_universe(name: str, folder: str | Path | None = None, daily: bool = False) -> Universe:
    """Load a universe written by ``scripts/build_data.py``."""
    folder = Path(folder) if folder else PROCESSED
    factors = _read("factors_monthly", folder)
    rf = factors["RF"]
    meta_path = folder / "build_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    if name == "multi_asset":
        ind = _read("ind12_vw", folder)
        bonds = _read("ust_monthly", folder)[["UST10", "UST5"]]
        returns = ind.join(bonds, how="inner").dropna()
        group = {c: "Equity" for c in ind.columns} | {"UST10": "Treasury", "UST5": "Treasury"}
        labels = dict(INDUSTRY12_NAMES) | {"UST10": "10-year Treasury", "UST5": "5-year Treasury"}
        mcap = _read("ind12_mcap", folder).reindex(returns.index)
        bm = _read("ind12_bm", folder).reindex(returns.index)
        dly = None
        if daily:
            d_ind = _read("ind12_vw_daily", folder)
            d_b = _read("ust_daily", folder)[["UST10", "UST5"]]
            dly = d_ind.join(d_b, how="inner").dropna()
    elif name == "industries49":
        ind = _read("ind49_vw", folder)
        start = ind.dropna().index[0]
        returns = ind.loc[start:].dropna(axis=0, how="any")
        group = {a: s for s, members in SECTORS49.items() for a in members if a in returns.columns}
        labels = {c: c for c in returns.columns}
        mcap = _read("ind49_mcap", folder).reindex(returns.index)
        bm = _read("ind49_bm", folder).reindex(returns.index)
        dly = _read("ind49_vw_daily", folder).loc[start:].dropna() if daily else None
    elif name == "industries12":
        returns = _read("ind12_vw", folder).dropna()
        group = {c: c for c in returns.columns}
        labels = dict(INDUSTRY12_NAMES)
        mcap = _read("ind12_mcap", folder).reindex(returns.index)
        bm = _read("ind12_bm", folder).reindex(returns.index)
        dly = _read("ind12_vw_daily", folder) if daily else None
    elif name == "sleeves":
        # Factor returns are self-financing; a sleeve held as an asset is collateralised in T-bills, so
        # its total return is RF + factor (and its excess return is the factor itself).
        bonds = _read("ust_monthly", folder)
        cols = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "ST_Rev", "LT_Rev"]
        sl = factors[cols].copy()
        sl["Term"] = bonds["UST10"].reindex(sl.index) - rf.reindex(sl.index)
        returns = sl.dropna().add(rf, axis=0).dropna()
        group = {c: SLEEVE_FAMILY[c].split(" (")[0] for c in returns.columns}
        labels = dict(SLEEVE_FAMILY)
        mcap, bm = None, None
        dly = None
        if daily:
            fd = _read("factors_daily", folder)
            bd = _read("ust_daily", folder)
            d = fd[cols].copy()
            d["Term"] = bd["UST10"].reindex(d.index) - fd["RF"]
            dly = d.dropna().add(fd["RF"], axis=0).dropna()
    else:
        raise KeyError(f"unknown universe {name!r}; choose multi_asset, industries49 or sleeves")
    ff = factors[["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]].reindex(returns.index)
    return Universe(name, returns, rf.reindex(returns.index), group, labels, mcap, bm, ff, dly, meta)
