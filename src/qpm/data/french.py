"""Readers for the Kenneth R. French Data Library CSV zips.

The files are plain text tables separated by section titles, for example::

      Average Value Weighted Returns -- Monthly
    ,NoDur,Durbl,...
    192607,   1.44,  15.58, ...

Returns are in percent and become decimals here; ``-99.99`` and ``-999`` mean missing.  Row labels are
``YYYYMM`` (monthly, mapped to the month-end date), ``YYYYMMDD`` (daily) or ``YYYY`` (annual).
"""
from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

MISSING = (-99.99, -999.0)


def _read_text(path: Path) -> str:
    with zipfile.ZipFile(path) as zf:
        name = zf.namelist()[0]
        return zf.read(name).decode("latin-1")


def _index_from_codes(codes: list[str]) -> pd.Index:
    width = len(codes[0])
    if width == 6:  # YYYYMM -> month end
        return pd.PeriodIndex([f"{c[:4]}-{c[4:]}" for c in codes], freq="M").to_timestamp(how="end").normalize()
    if width == 8:  # YYYYMMDD
        return pd.to_datetime(codes, format="%Y%m%d")
    if width == 4:  # YYYY
        return pd.Index([int(c) for c in codes], name="year")
    raise ValueError(f"unexpected row label width {width}: {codes[0]!r}")


def read_tables(path: str | Path) -> dict[str, pd.DataFrame]:
    """Every table in a French CSV zip, keyed by its section title ("" for an untitled first table).

    Values are returned exactly as printed (percent for returns), with missing codes set to NaN.
    A table whose rows switch label width (monthly block followed by an annual block without a title,
    as in the factor files) is split at the switch; the annual block gets the key ``"<title> annual"``.
    """
    lines = _read_text(Path(path)).splitlines()
    tables: dict[str, pd.DataFrame] = {}
    title = ""
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith(","):  # header row of a table
            columns = [c.strip() for c in stripped.split(",")[1:]]
            rows: list[list[str]] = []
            i += 1
            while i < len(lines):
                row = lines[i].strip()
                if not row or not row[0].isdigit():
                    break
                rows.append([c.strip() for c in row.split(",")])
                i += 1
            blocks: dict[int, list[list[str]]] = {}
            for r in rows:
                blocks.setdefault(len(r[0]), []).append(r)
            for width, block in blocks.items():
                key = title if width != 4 or len(blocks) == 1 else f"{title} annual".strip()
                if width == 4 and len(blocks) > 1 and not title:
                    key = "annual"
                codes = [r[0] for r in block]
                values = np.array([[float(x) if x not in ("",) else np.nan for x in r[1:1 + len(columns)]]
                                   for r in block], dtype=float)
                for code in MISSING:
                    values[np.isclose(values, code)] = np.nan
                frame = pd.DataFrame(values, index=_index_from_codes(codes), columns=columns)
                tables[key] = frame
            continue
        if stripped and not stripped[0].isdigit():
            title = re.sub(r"\s+", " ", stripped)
        i += 1
    return tables


def _pick(tables: dict[str, pd.DataFrame], pattern: str) -> pd.DataFrame:
    for key, frame in tables.items():
        if re.search(pattern, key, flags=re.I):
            return frame
    raise KeyError(f"no table matching {pattern!r}; have {list(tables)}")


def industry_monthly(path: str | Path) -> dict[str, pd.DataFrame]:
    """Monthly industry portfolio file -> dict of frames.

    ``vw`` / ``ew``: monthly total returns (decimal); ``nfirms``: number of firms; ``size``: average firm
    size in $ millions (French: average market equity at the start of the month); ``mcap``: nfirms x size
    ($ millions); ``bm``: annual Sum of BE / Sum of ME indexed by the June-formation year.
    """
    t = read_tables(path)
    vw = _pick(t, r"^Average Value Weighted Returns -- Monthly$") / 100.0
    ew = _pick(t, r"^Average Equal Weighted Returns -- Monthly$") / 100.0
    nfirms = _pick(t, r"^Number of Firms in Portfolios$")
    size = _pick(t, r"^Average Firm Size$")
    bm = _pick(t, r"^Sum of BE / Sum of ME$")
    return {"vw": vw, "ew": ew, "nfirms": nfirms, "size": size, "mcap": nfirms * size, "bm": bm}


def industry_daily(path: str | Path) -> dict[str, pd.DataFrame]:
    t = read_tables(path)
    return {"vw": _pick(t, r"Value Weighted Returns -- Daily") / 100.0,
            "ew": _pick(t, r"Equal Weighted Returns -- Daily") / 100.0}


def _first_table(path: Path) -> pd.DataFrame:
    t = read_tables(path)
    frame = t.get("") if "" in t else next(iter(t.values()))
    return frame


def factors(raw_dir: str | Path, frequency: str = "monthly") -> pd.DataFrame:
    """Mkt-RF, SMB, HML, RMW, CMA, Mom, ST_Rev, LT_Rev and RF as decimals, aligned on dates.

    The 5-factor file sets the sample start (July 1963); the momentum and reversal files go back further
    and are cut to it.  Missing files are skipped, so a partial download still loads.
    """
    raw_dir = Path(raw_dir)
    daily = frequency.startswith("d")
    suffix = "_daily_CSV.zip" if daily else "_CSV.zip"
    names = {
        "ff5": "F-F_Research_Data_5_Factors_2x3",
        "mom": "F-F_Momentum_Factor",
        "st": "F-F_ST_Reversal_Factor",
        "lt": "F-F_LT_Reversal_Factor",
    }
    parts = []
    for key, stem in names.items():
        path = raw_dir / (stem + suffix)
        if not path.exists():
            continue
        frame = _first_table(path) / 100.0
        frame.columns = [c.strip().replace(" ", "") for c in frame.columns]
        parts.append(frame)
    if not parts:
        raise FileNotFoundError(f"no French factor files in {raw_dir}")
    out = pd.concat(parts, axis=1, join="outer").sort_index()
    out = out.loc[:, ~out.columns.duplicated()]
    # Before July 1963 only the 3-factor file exists.  Its market and T-bill series are the same as the
    # 5-factor file's, so they extend Mkt-RF and RF back to 1926; its SMB/HML use a different construction
    # and are NOT spliced in (RMW, CMA, SMB, HML stay missing before 1963-07).
    three = raw_dir / ("F-F_Research_Data_Factors" + suffix)
    if three.exists():
        base = _first_table(three) / 100.0
        base.columns = [c.strip() for c in base.columns]
        out = out.reindex(out.index.union(base.index))
        for col in ("Mkt-RF", "RF"):
            if col in base:
                out[col] = out[col].combine_first(base[col]) if col in out else base[col]
    if "Mkt-RF" in out:
        out = out.loc[out["Mkt-RF"].first_valid_index():]
    order = [c for c in ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom", "ST_Rev", "LT_Rev", "RF"] if c in out]
    return out[order]


def bm_point_in_time(bm_annual: pd.DataFrame, month_ends: pd.DatetimeIndex) -> pd.DataFrame:
    """Map annual BE/ME (French's June-formation convention) onto month-end decision dates.

    The value labelled year Y uses book equity for fiscal years ending in Y-1 and market equity at
    December Y-1; French forms portfolios with it at the end of June of Y.  So it is treated as known from
    the June Y month end through May Y+1.  Earlier dates get NaN.
    """
    out = pd.DataFrame(index=month_ends, columns=bm_annual.columns, dtype=float)
    for date in month_ends:
        year = date.year if date.month >= 6 else date.year - 1
        if year in bm_annual.index:
            out.loc[date] = bm_annual.loc[year].values
    return out
