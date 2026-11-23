"""Turn the raw downloads into the processed monthly/daily panels the library reads.

    python scripts/build_data.py            # writes data/processed/*.csv and build_meta.json

Inputs: data/raw/french/*.zip and data/raw/fred/*.csv (see scripts/fetch_data.py).
"""
from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qpm.data import french, fred  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"


def write(frame: pd.DataFrame | pd.Series, name: str, digits: int = 6) -> dict:
    OUT.mkdir(parents=True, exist_ok=True)
    frame = frame.to_frame() if isinstance(frame, pd.Series) else frame
    frame.index.name = "date"
    frame.round(digits).to_csv(OUT / f"{name}.csv", float_format=f"%.{digits}f")
    first, last = frame.index[0], frame.index[-1]
    return {"rows": int(len(frame)), "columns": int(frame.shape[1]),
            "first": str(getattr(first, "date", lambda: first)()), "last": str(getattr(last, "date", lambda: last)())}


def main() -> int:
    fr = RAW / "french"
    meta: dict = {"built_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "tables": {}}

    # Factors (monthly + daily)
    fm = french.factors(fr, "monthly")
    fd = french.factors(fr, "daily")
    meta["tables"]["factors_monthly"] = write(fm, "factors_monthly")
    meta["tables"]["factors_daily"] = write(fd, "factors_daily")

    # Industries: returns, market caps, point-in-time BE/ME
    for n in (12, 49):
        m = french.industry_monthly(fr / f"{n}_Industry_Portfolios_CSV.zip")
        d = french.industry_daily(fr / f"{n}_Industry_Portfolios_daily_CSV.zip")
        meta["tables"][f"ind{n}_vw"] = write(m["vw"], f"ind{n}_vw")
        meta["tables"][f"ind{n}_mcap"] = write(m["mcap"], f"ind{n}_mcap", 2)
        bm = french.bm_point_in_time(m["bm"], m["vw"].index)
        meta["tables"][f"ind{n}_bm"] = write(bm, f"ind{n}_bm", 4)
        meta["tables"][f"ind{n}_vw_daily"] = write(d["vw"], f"ind{n}_vw_daily")

    # Treasuries rebuilt from H.15 constant-maturity yields
    ys = {k: fred.read_fred_csv(RAW / "fred" / f"DGS{k}.csv") for k in ("10", "5", "2", "30")}
    monthly = pd.concat({f"UST{k}": fred.constant_maturity_returns(v, float(k), "M") for k, v in ys.items()}, axis=1)
    daily = pd.concat({f"UST{k}": fred.constant_maturity_returns(v, float(k), "D") for k, v in ys.items()}, axis=1)
    meta["tables"]["ust_monthly"] = write(monthly, "ust_monthly")
    meta["tables"]["ust_daily"] = write(daily, "ust_daily")
    yields_m = pd.concat({f"DGS{k}": fred.month_end_level(v) for k, v in ys.items()}, axis=1)
    meta["tables"]["yields_monthly"] = write(yields_m, "yields_monthly", 4)

    # Financial stress index (weekly -> month end); public St. Louis Fed series, used as a regime input
    stress = fred.month_end_level(fred.read_fred_csv(RAW / "fred" / "STLFSI4.csv")).rename("STLFSI4")
    meta["tables"]["stress_monthly"] = write(stress, "stress_monthly", 4)

    manifest = RAW / "manifest.json"
    if manifest.exists():
        meta["raw_manifest"] = json.loads(manifest.read_text(encoding="utf-8"))
    (OUT / "build_meta.json").write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    for name, info in meta["tables"].items():
        print(f"{name:18s} {info['rows']:>7,} rows x {info['columns']:>2}  {info['first']} .. {info['last']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
