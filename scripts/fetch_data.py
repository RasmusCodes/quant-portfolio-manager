"""Download (or copy from a local cache) the public data the study uses.

Sources
-------
* Kenneth R. French Data Library (industry portfolios and factor returns), Tuck School of Business.
* FRED (Federal Reserve Bank of St. Louis): H.15 constant-maturity Treasury yields and the effective
  federal funds rate (public-domain Federal Reserve data), plus the St. Louis Fed Financial Stress Index.

Every file is recorded in ``data/raw/manifest.json`` with its URL, size, SHA-256 and fetch time, so a
rerun can tell whether the inputs changed.  Nothing here is needed to run the tests.

Usage::

    python scripts/fetch_data.py french                 # download the French files
    python scripts/fetch_data.py fred                   # download the FRED series
    python scripts/fetch_data.py french --copy-from DIR # copy files already on disk instead
    python scripts/fetch_data.py fred --copy-from DIR
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MANIFEST = RAW / "manifest.json"

FRENCH_BASE = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
FRENCH_FILES = [
    "12_Industry_Portfolios_CSV.zip",
    "12_Industry_Portfolios_daily_CSV.zip",
    "49_Industry_Portfolios_CSV.zip",
    "49_Industry_Portfolios_daily_CSV.zip",
    "F-F_Research_Data_Factors_CSV.zip",
    "F-F_Research_Data_Factors_daily_CSV.zip",
    "F-F_Research_Data_5_Factors_2x3_CSV.zip",
    "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip",
    "F-F_Momentum_Factor_CSV.zip",
    "F-F_Momentum_Factor_daily_CSV.zip",
    "F-F_ST_Reversal_Factor_CSV.zip",
    "F-F_ST_Reversal_Factor_daily_CSV.zip",
    "F-F_LT_Reversal_Factor_CSV.zip",
    "F-F_LT_Reversal_Factor_daily_CSV.zip",
]
FRED_SERIES = ["DGS10", "DGS5", "DGS2", "DGS30", "DGS3MO", "DFF", "STLFSI4"]
FRED_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
USER_AGENT = "quant-portfolio-manager data fetch (research use)"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {"files": {}}


def save_manifest(manifest: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    tmp.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(MANIFEST)


def download(url: str, target: Path, retries: int = 3) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp, open(tmp, "wb") as out:
                shutil.copyfileobj(resp, out)
            tmp.replace(target)
            return
        except Exception as exc:  # network errors: retry with backoff
            last = exc
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"download failed for {url}: {last}")


def record(manifest: dict, rel: str, url: str, path: Path, how: str) -> None:
    manifest["files"][rel] = {
        "url": url,
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
        "fetched_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "how": how,
    }


def run(kind: str, copy_from: str | None, only: list[str] | None) -> int:
    manifest = load_manifest()
    if kind == "french":
        names, folder = FRENCH_FILES, RAW / "french"
        url_of = lambda n: FRENCH_BASE + n  # noqa: E731
        file_of = lambda n: n  # noqa: E731
    else:
        names, folder = FRED_SERIES, RAW / "fred"
        url_of = lambda n: FRED_URL.format(series=n)  # noqa: E731
        file_of = lambda n: f"{n}.csv"  # noqa: E731
    if only:
        names = [n for n in names if n in only or file_of(n) in only]
    failures = 0
    for name in names:
        target = folder / file_of(name)
        rel = target.relative_to(RAW).as_posix()
        try:
            if copy_from:
                src = Path(copy_from) / file_of(name)
                if not src.exists():
                    print(f"skip   {rel}: not in {copy_from}")
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, target)
                how = f"copied from local cache {src.name} (originally {url_of(name)})"
            else:
                download(url_of(name), target)
                how = "downloaded"
            record(manifest, rel, url_of(name), target, how)
            print(f"ok     {rel}  {target.stat().st_size:,} bytes  ({how.split(' (')[0]})")
        except Exception as exc:
            failures += 1
            print(f"FAILED {rel}: {exc}")
    save_manifest(manifest)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("kind", choices=["french", "fred"])
    ap.add_argument("--copy-from", help="copy the files from this folder instead of downloading them")
    ap.add_argument("--only", nargs="*", help="restrict to these file names / series ids")
    args = ap.parse_args(argv)
    return run(args.kind, args.copy_from, args.only)


if __name__ == "__main__":
    sys.exit(main())
