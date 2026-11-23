"""Adapters that turn research systems' outputs into ``SignalStore`` records.

The portfolio manager never imports a research system's code.  Each system exports files in the
documented schema (docs/SIGNALS.md), or an adapter here maps an existing export onto it.

* ``read_signal_file``      any CSV/JSON already in the qpm schema (the default integration path).
* ``statarb_signals``       the statarb platform's dashboard export ``results/dashboard/signals.parquet``
                            (S&P 500 stocks x month; nine z-scored signals as its optimizer saw them).
                            Its ``madj_next`` column is the NEXT month's market-adjusted return, i.e. a
                            label from the future: it is never read as a signal, only as realised returns
                            for track records via ``statarb_realized_returns``.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .schema import SignalStore

STATARB_FAMILIES = {
    "MOM": "equity_factor", "RMOM": "equity_factor", "IREV": "statistical_arbitrage", "LIVOL": "equity_factor",
    "VALUE": "equity_factor", "QUALITY": "equity_factor", "SUE": "earnings_event", "FUND": "equity_factor",
    "OP": "equity_factor",
}


def read_signal_file(path: str | Path, validate: bool = True, universe=None) -> SignalStore:
    store = SignalStore.read(path)
    if validate:
        store.validate(universe)
    return store


def statarb_signals(path: str | Path, columns=None, model_version: str = "statarb-dashboard") -> SignalStore:
    frame = pd.read_parquet(path)
    cols = [c for c in (columns or STATARB_FAMILIES) if c in frame.columns]
    stores = []
    for col in cols:
        panel = frame[col].unstack("sid")
        panel.index = pd.to_datetime(panel.index)
        stores.append(SignalStore.from_panel(panel, f"statarb.{col.lower()}", STATARB_FAMILIES[col], "score", 21,
                                             model_version, f"statarb {Path(path).name}: {col} z-score at formation"))
    out = stores[0]
    for s in stores[1:]:
        out = out.concat(s)
    return out


def statarb_realized_returns(path: str | Path) -> pd.DataFrame:
    """Row t = market-adjusted return over the month AFTER formation date t (known only at t+1)."""
    frame = pd.read_parquet(path, columns=["madj_next"])
    panel = frame["madj_next"].unstack("sid")
    panel.index = pd.to_datetime(panel.index)
    return panel
