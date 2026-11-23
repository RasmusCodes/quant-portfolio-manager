"""Contract test against the real router, when its repository is available (skipped otherwise)."""
import os
from pathlib import Path

import pytest

from qpm.execution import ParentOrder, SmartOrderRouterAdapter, validate_report

ROOT = Path(__file__).resolve().parents[1]
SIBLING = ROOT.parent / "multi-venue-smart-router" / "src"


def _available():
    if SIBLING.exists():
        os.environ.setdefault("QPM_SMARTROUTER_SRC", str(SIBLING))
    try:
        from qpm.execution import smartrouter_bridge
        smartrouter_bridge._import()
        return True
    except ImportError:
        return False


@pytest.mark.skipif(not _available(), reason="multi-venue-smart-router not installed")
def test_router_reports_pass_the_contract():
    router = SmartOrderRouterAdapter("python", callable_path="qpm.execution.smartrouter_bridge:route")
    orders = [ParentOrder("2026-09-30-BusEq", "BusEq", "buy", 50_000, 100.0, "2026-09-30T16:00:00"),
              ParentOrder("2026-09-30-UST10", "UST10", "sell", 20_000, 100.0, "2026-09-30T16:00:00", urgency="high")]
    reports = router.execute(orders)
    for o, r in zip(orders, reports):
        assert validate_report(o, r) == []
        assert r.executor.startswith("smartrouter:")
        assert 0 < r.filled_quantity <= o.quantity
    again = router.execute(orders[:1])
    assert again[0].avg_price == reports[0].avg_price      # seeded by order id: deterministic
