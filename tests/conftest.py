import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)


def random_cov(rng, n, scale=0.04):
    g = rng.normal(size=(n, n))
    return g @ g.T / n * scale + np.diag(rng.uniform(0.002, 0.02, n))


@pytest.fixture
def small_market(rng):
    """Synthetic monthly returns for 8 assets with a factor structure, 240 months."""
    n, t = 8, 240
    dates = pd.date_range("2000-01-31", periods=t, freq="ME")
    f = rng.normal(0.006, 0.04, size=(t, 2))
    b = rng.uniform(0.3, 1.3, size=(n, 2))
    e = rng.normal(0, 0.03, size=(t, n))
    r = f @ b.T + e + rng.uniform(-0.002, 0.004, n)
    names = [f"A{i}" for i in range(n)]
    returns = pd.DataFrame(r, index=dates, columns=names)
    rf = pd.Series(0.002, index=dates)
    return returns, rf


@pytest.fixture
def processed_data():
    folder = ROOT / "data" / "processed"
    if not (folder / "ind12_vw.csv").exists():
        pytest.skip("processed data not built (run scripts/fetch_data.py and scripts/build_data.py)")
    return folder
