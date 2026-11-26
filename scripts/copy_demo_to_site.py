"""Copy the browser demo into a static site as apps/<name>/, with content-hash version tags on its assets
so browsers never mix an old script with new data after an update.

    python scripts/copy_demo_to_site.py --site ../website --name portfolio-manager

Copies index.html, style.css, engine.js, charts.js, app.js and data/bundle.js (not the tests or the
per-section JSON files, which the bundle already contains).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEMO = ROOT / "demo"
FILES = ["style.css", "engine.js", "charts.js", "app.js", "data/bundle.js"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", required=True, help="the static site's root folder")
    ap.add_argument("--name", default="portfolio-manager")
    args = ap.parse_args(argv)
    target = Path(args.site).resolve() / "apps" / args.name
    (target / "data").mkdir(parents=True, exist_ok=True)
    tags = {}
    for rel in FILES:
        data = (DEMO / rel).read_bytes()
        tags[rel] = hashlib.sha256(data).hexdigest()[:10]
        dst = target / rel
        if not dst.exists() or dst.read_bytes() != data:
            shutil.copyfile(DEMO / rel, dst)
    html = (DEMO / "index.html").read_text(encoding="utf-8")
    for rel, tag in tags.items():
        html = re.sub(rf'(src|href)="{re.escape(rel)}(\?v=[0-9a-f]+)?"', rf'\1="{rel}?v={tag}"', html)
    (target / "index.html").write_text(html, encoding="utf-8")
    print(f"copied to {target}: " + ", ".join(f"{k}@{v}" for k, v in tags.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
