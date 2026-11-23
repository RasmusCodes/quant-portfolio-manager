"""A stand-in smart order router that speaks the qpm execution contract over stdin/stdout.

NOT a real router: it exists so the command-line integration path can be tested end to end without the
separate router repository.  It splits each parent order across three fictional venues in proportion to
their displayed size, takes liquidity with venue-specific fees, and adds square-root impact.

    python examples/stub_router.py < orders.jsonl > reports.jsonl
"""
from __future__ import annotations

import json
import math
import sys

VENUES = [  # name, share of displayed liquidity, taker fee (bps of notional), extra slippage (bps)
    ("VENUE_A", 0.5, 0.30, 0.0),
    ("VENUE_B", 0.3, 0.20, 0.5),
    ("VENUE_C", 0.2, 0.10, 1.0),
]


def route(order: dict) -> dict:
    side = 1.0 if order["side"] == "buy" else -1.0
    arrival = float(order["arrival_price"])
    qty = float(order["quantity"])
    half_spread = float(order.get("spread_bps") or 2.0) / 2.0
    vol = float(order.get("daily_vol") or 0.015)
    adv = order.get("adv")
    impact = 1e4 * 0.5 * vol * math.sqrt(float(order["notional"]) / float(adv)) if adv else 0.0
    fills = []
    for name, share, fee_bps, slip in VENUES:
        q = qty * share
        price = arrival * (1 + side * (half_spread + slip + impact) / 1e4)
        fills.append({"venue": name, "quantity": q, "price": price, "liquidity": "taker",
                      "fee": q * price * fee_bps / 1e4})
    filled = sum(f["quantity"] for f in fills)
    avg = sum(f["quantity"] * f["price"] for f in fills) / filled
    fees = sum(f["fee"] for f in fills)
    shortfall = 1e4 * (side * (avg - arrival) * filled + fees) / (arrival * filled)
    return {"schema": "qpm.execution_report/1", "order_id": order["order_id"], "instrument": order["instrument"],
            "side": order["side"], "filled_quantity": filled, "avg_price": avg, "arrival_price": arrival,
            "fees": fees, "status": "filled", "fills": fills, "shortfall_bps": shortfall,
            "breakdown_bps": {"spread": half_spread, "impact": impact}, "executor": "stub-router"}


def main() -> int:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        order = json.loads(line)
        if order.get("schema") != "qpm.parent_order/1":
            print(f"unsupported schema {order.get('schema')!r}", file=sys.stderr)
            return 2
        sys.stdout.write(json.dumps(route(order)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
