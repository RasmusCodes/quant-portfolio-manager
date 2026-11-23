"""Signal schema and a point-in-time signal store.

Every research system hands the portfolio manager the same record (see docs/SIGNALS.md):

    timestamp      when the signal became known; a decision at time t may only use timestamp <= t
    instrument     identifier in the portfolio's universe
    value          expected return over the horizon (kind="expected_return", decimal) or a score
    kind           expected_return | score | direction (+1/-1)
    confidence     the producer's own confidence in [0, 1]
    horizon_days   holding horizon in trading days; the signal expires after it
    volatility     estimated annualised volatility of the instrument (optional)
    source         research system and model, e.g. "statarb.residual_reversal"
    family         one of FAMILIES
    model_version  version string of the producing model
    provenance     free text / JSON: data sources and the as-of dates of the inputs
    ic_mean, ic_t, hit_rate, n_obs   the producer's track record as of ``timestamp`` (optional)

The store's ``as_of`` is the only way the backtester reads signals, and it never returns a record
stamped after the decision time.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

FAMILIES = {
    "statistical_arbitrage": "Statistical arbitrage",
    "equity_factor": "Equity factor strategies",
    "options_volatility": "Options and volatility",
    "alternative_data": "Alternative data",
    "earnings_event": "Earnings and event-driven",
    "credit": "Credit and capital structure",
    "macro_fixed_income": "Macro and fixed income",
    "microstructure": "Market microstructure",
}
KINDS = ("expected_return", "score", "direction")
COLUMNS = ["timestamp", "instrument", "value", "kind", "confidence", "horizon_days", "volatility", "source",
           "family", "model_version", "provenance", "ic_mean", "ic_t", "hit_rate", "n_obs"]


@dataclass(frozen=True)
class Signal:
    timestamp: pd.Timestamp
    instrument: str
    value: float
    source: str
    family: str
    kind: str = "score"
    confidence: float = 0.5
    horizon_days: int = 21
    volatility: float | None = None
    model_version: str = "1"
    provenance: str = ""
    ic_mean: float | None = None
    ic_t: float | None = None
    hit_rate: float | None = None
    n_obs: int | None = None


class SignalValidationError(ValueError):
    pass


class SignalStore:
    """A table of signals with point-in-time access."""

    def __init__(self, frame: pd.DataFrame | None = None):
        frame = pd.DataFrame(columns=COLUMNS) if frame is None else frame.copy()
        for col in COLUMNS:
            if col not in frame:
                frame[col] = np.nan
        frame["timestamp"] = pd.to_datetime(frame["timestamp"])
        self.frame = frame[COLUMNS].sort_values(["timestamp", "source", "instrument"]).reset_index(drop=True)

    # construction / IO -------------------------------------------------------------------------
    @classmethod
    def from_records(cls, records: Iterable[Signal | dict]) -> "SignalStore":
        rows = [asdict(r) if isinstance(r, Signal) else dict(r) for r in records]
        return cls(pd.DataFrame(rows))

    @classmethod
    def from_panel(cls, panel: pd.DataFrame, source: str, family: str, kind: str = "score",
                   horizon_days: int = 21, model_version: str = "1", provenance: str = "",
                   confidence: float = 0.5, volatility: pd.DataFrame | None = None) -> "SignalStore":
        """A dates x instruments panel (index = knowledge times) -> store."""
        long = panel.stack().rename("value").reset_index()
        long.columns = ["timestamp", "instrument", "value"]
        if volatility is not None:
            vol = volatility.reindex_like(panel).stack().rename("volatility").reset_index()
            vol.columns = ["timestamp", "instrument", "volatility"]
            long = long.merge(vol, on=["timestamp", "instrument"], how="left")
        long["source"], long["family"], long["kind"] = source, family, kind
        long["horizon_days"], long["model_version"], long["provenance"] = horizon_days, model_version, provenance
        long["confidence"] = confidence
        return cls(long)

    @classmethod
    def read(cls, path: str | Path) -> "SignalStore":
        path = Path(path)
        if path.suffix.lower() == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            return cls.from_records(data["signals"] if isinstance(data, dict) else data)
        return cls(pd.read_csv(path))

    def write(self, path: str | Path) -> None:
        path = Path(path)
        if path.suffix.lower() == ".json":
            rows = self.frame.assign(timestamp=self.frame["timestamp"].dt.strftime("%Y-%m-%dT%H:%M:%S"))
            payload = {"schema": "qpm.signals/1", "signals": json.loads(rows.to_json(orient="records"))}
            path.write_text(json.dumps(payload, indent=1), encoding="utf-8")
        else:
            self.frame.to_csv(path, index=False)

    def concat(self, other: "SignalStore") -> "SignalStore":
        return SignalStore(pd.concat([self.frame, other.frame], ignore_index=True))

    def __len__(self) -> int:
        return len(self.frame)

    # validation ---------------------------------------------------------------------------------
    def validate(self, universe: Iterable[str] | None = None, strict: bool = True) -> list[str]:
        f = self.frame
        issues = []
        if f["timestamp"].isna().any():
            issues.append(f"{int(f['timestamp'].isna().sum())} signals have no timestamp")
        if not np.isfinite(pd.to_numeric(f["value"], errors="coerce")).all():
            issues.append("non-finite signal values")
        bad_family = set(f["family"].dropna()) - set(FAMILIES)
        if bad_family:
            issues.append(f"unknown families: {sorted(bad_family)}")
        bad_kind = set(f["kind"].dropna()) - set(KINDS)
        if bad_kind:
            issues.append(f"unknown kinds: {sorted(bad_kind)}")
        conf = pd.to_numeric(f["confidence"], errors="coerce")
        if ((conf < 0) | (conf > 1)).any():
            issues.append("confidence outside [0, 1]")
        hz = pd.to_numeric(f["horizon_days"], errors="coerce")
        if (hz <= 0).any() or hz.isna().any():
            issues.append("horizon_days must be positive")
        if f["source"].isna().any() or (f["source"].astype(str).str.len() == 0).any():
            issues.append("every signal needs a source")
        direction = f["kind"] == "direction"
        if direction.any() and not f.loc[direction, "value"].isin([-1, 0, 1]).all():
            issues.append("direction signals must be -1, 0 or +1")
        dup = f.duplicated(["timestamp", "source", "instrument"]).sum()
        if dup:
            issues.append(f"{int(dup)} duplicate (timestamp, source, instrument) rows")
        if universe is not None:
            unknown = set(f["instrument"]) - set(universe)
            if unknown:
                issues.append(f"instruments outside the universe: {sorted(unknown)[:8]}")
        if strict and issues:
            raise SignalValidationError("; ".join(issues))
        return issues

    # point-in-time access -----------------------------------------------------------------------
    def as_of(self, when, expire: bool = True) -> pd.DataFrame:
        """Latest signal per (source, instrument) known at ``when`` (timestamp <= when), dropping
        signals older than their horizon (trading days converted at 365/252 calendar days)."""
        when = pd.Timestamp(when)
        f = self.frame[self.frame["timestamp"] <= when]
        if f.empty:
            return f.copy()
        latest = f.sort_values("timestamp").groupby(["source", "instrument"], as_index=False).tail(1)
        if expire:
            age = (when - latest["timestamp"]).dt.days
            latest = latest[age <= latest["horizon_days"].astype(float) * 365.0 / 252.0 + 1e-9]
        assert (latest["timestamp"] <= when).all(), "point-in-time violation"
        return latest.reset_index(drop=True)

    def panel(self, source: str, field: str = "value") -> pd.DataFrame:
        f = self.frame[self.frame["source"] == source]
        return f.pivot_table(index="timestamp", columns="instrument", values=field, aggfunc="last")

    @property
    def sources(self) -> list[str]:
        return sorted(self.frame["source"].dropna().unique())

    def describe(self) -> pd.DataFrame:
        g = self.frame.groupby("source")
        return pd.DataFrame({"family": g["family"].first(), "signals": g.size(), "instruments": g["instrument"].nunique(),
                             "first": g["timestamp"].min(), "last": g["timestamp"].max(),
                             "model_version": g["model_version"].last()})
