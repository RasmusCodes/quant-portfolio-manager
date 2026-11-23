import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qpm.data import french, fred, load_universe
from qpm.execution import (CostModelExecution, ExecutionReport, ParentOrder, RouterContractError,
                           SmartOrderRouterAdapter, validate_report)

ROOT = Path(__file__).resolve().parents[1]


def _order(side="buy"):
    return ParentOrder("o1", "XYZ", side, 1000, 50.0, "2026-09-30T16:00:00", adv=5e7, daily_vol=0.02, spread_bps=4)


def test_cost_model_reports_pass_contract():
    for side in ("buy", "sell"):
        o = _order(side)
        rep = CostModelExecution().execute([o])[0]
        assert validate_report(o, rep) == []
        assert rep.shortfall_bps > 0                     # costs are positive for both sides
        assert np.isclose(rep.cost_fraction(o) * 1e4, rep.shortfall_bps)


def test_contract_violations_are_caught():
    o = _order()
    rep = CostModelExecution().execute([o])[0]
    rep.filled_quantity = 2000
    assert any("filled" in p for p in validate_report(o, rep))
    rep = CostModelExecution().execute([o])[0]
    rep.shortfall_bps += 5
    assert any("shortfall" in p for p in validate_report(o, rep))


def test_order_validation():
    with pytest.raises(ValueError):
        ParentOrder("o", "X", "hold", 1, 1.0, "t")
    with pytest.raises(ValueError):
        ParentOrder("o", "X", "buy", -1, 1.0, "t")


def test_python_transport_and_strictness(tmp_path, monkeypatch):
    mod = tmp_path / "fake_router.py"
    mod.write_text(
        "def good(orders):\n"
        "    out = []\n"
        "    for o in orders:\n"
        "        out.append({'schema': 'qpm.execution_report/1', 'order_id': o['order_id'], 'instrument': o['instrument'],\n"
        "                    'side': o['side'], 'filled_quantity': o['quantity'], 'avg_price': o['arrival_price'],\n"
        "                    'arrival_price': o['arrival_price'], 'fees': 0.0, 'status': 'filled', 'fills': [],\n"
        "                    'shortfall_bps': 0.0})\n"
        "    return out\n"
        "def bad(orders):\n"
        "    return [dict(r, filled_quantity=r['filled_quantity'] * 2) for r in good(orders)]\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    ok = SmartOrderRouterAdapter("python", callable_path="fake_router:good").execute([_order()])
    assert ok[0].filled_quantity == 1000
    with pytest.raises(RouterContractError):
        SmartOrderRouterAdapter("python", callable_path="fake_router:bad").execute([_order()])


def test_stub_router_cli_round_trip():
    router = SmartOrderRouterAdapter("cli", [sys.executable, str(ROOT / "examples" / "stub_router.py")])
    reps = router.execute([_order("buy"), ParentOrder("o2", "ABC", "sell", 10, 20.0, "t", adv=1e6)])
    assert [r.order_id for r in reps] == ["o1", "o2"]
    assert len(reps[0].fills) == 3 and reps[0].executor == "stub-router"


def _french_zip(tmp_path):
    text = ("Intro line\n\n  Average Value Weighted Returns -- Monthly\n,AA,BB\n202401,   1.00,  -99.99\n"
            "202402,   2.00,   3.00\n\n  Number of Firms in Portfolios\n,AA,BB\n202401, 10, 20\n202402, 11, 21\n\n"
            "  Average Firm Size\n,AA,BB\n202401, 5.0, 6.0\n202402, 5.5, 6.5\n\n"
            "  Average Equal Weighted Returns -- Monthly\n,AA,BB\n202401, 1, 2\n202402, 3, 4\n\n"
            "  Sum of BE / Sum of ME\n,AA,BB\n2023, 0.5, 0.6\n2024, 0.7, 0.8\n\nCopyright\n")
    path = tmp_path / "x.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("x.csv", text)
    return path


def test_french_parser(tmp_path):
    m = french.industry_monthly(_french_zip(tmp_path))
    assert m["vw"].index[0] == pd.Timestamp("2024-01-31")
    assert np.isnan(m["vw"].iloc[0, 1]) and np.isclose(m["vw"].iloc[1, 0], 0.02)
    assert np.isclose(m["mcap"].iloc[1, 1], 21 * 6.5)
    pit = french.bm_point_in_time(m["bm"], pd.DatetimeIndex(["2024-05-31", "2024-06-30"]))
    assert np.isclose(pit.iloc[0, 0], 0.5) and np.isclose(pit.iloc[1, 0], 0.7)


def test_par_bond_math():
    assert np.isclose(fred.par_bond_price(0.05, 0.05, 10.0), 1.0)
    days = pd.date_range("2020-01-01", "2020-12-31", freq="B")
    flat = pd.Series(4.0, index=days)
    r = fred.constant_maturity_returns(flat, 10.0, "M")
    # constant yield: return ~ coupon accrual y/12 (plus the small roll-down from the shorter maturity)
    assert np.allclose(r.values, 0.04 / 12, atol=2e-5)


def test_partial_month_is_dropped():
    s = pd.Series([1.0, 2.0, 3.0], index=pd.to_datetime(["2026-09-30", "2026-10-30", "2026-11-10"]))
    out = fred.month_end_level(s)
    assert out.index[-1] == pd.Timestamp("2026-10-31")


def test_universes_load(processed_data):
    for name, n in (("multi_asset", 14), ("industries12", 12), ("industries49", 49), ("sleeves", 9)):
        u = load_universe(name, processed_data)
        assert u.returns.shape[1] == n and not u.returns.isna().any().any()
        assert u.rf.index.equals(u.returns.index)
