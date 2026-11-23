"""Bridge to the separate multi-venue smart order router (repository ``multi-venue-smart-router``,
package ``smartrouter``).  qpm never vendors or imports it at build time.

Make it importable either by installing it (``pip install git+https://github.com/RasmusCodes/multi-venue-smart-router``
or ``pip install -e ../multi-venue-smart-router``) or by pointing ``QPM_SMARTROUTER_SRC`` at its ``src`` folder.

Use it as the Python transport of the generic adapter::

    from qpm.execution import SmartOrderRouterAdapter
    router = SmartOrderRouterAdapter("python", callable_path="qpm.execution.smartrouter_bridge:route")

What the numbers mean.  The router simulates a fragmented market (four lit venues and a dark pool, with
fees, latency, queues and fading liquidity) and executes ONE standard parent order there with the chosen
routing strategy.  The bridge applies that order's measured shortfall per share (price and fees, in basis
points of the arrival mid) to the qpm order and scales its fill rate.  The router knows nothing about an
industry portfolio's liquidity, so size-dependent impact for large books stays with qpm's own cost model;
what the router adds is the cost of HOW an order is routed, and the difference between routing strategies.

Settings (environment variables or keyword arguments): strategy (default "smart"), scenario (default
"baseline"), urgency mapping low/medium/high -> horizon 600/300/60 s with 30/20/6 slices.
"""
from __future__ import annotations

import os
import sys
import zlib
from pathlib import Path

URGENCY = {"low": (600.0, 30), "medium": (300.0, 20), "high": (60.0, 6)}


def _import():
    try:
        import smartrouter  # noqa: F401
    except ImportError:
        src = os.environ.get("QPM_SMARTROUTER_SRC")
        if not src:
            raise ImportError("smartrouter is not installed: pip install the multi-venue-smart-router repository "
                              "or set QPM_SMARTROUTER_SRC to its src/ folder") from None
        sys.path.insert(0, str(Path(src)))
        import smartrouter  # noqa: F401
    from smartrouter.calibrate import load
    from smartrouter.config import ParentConfig, preset
    from smartrouter.episode import run_episode
    return ParentConfig, preset, run_episode, load


def _stats(load, scenario: str):
    """Fitted fill-probability statistics the model-based routers need (shipped in the router repo's results/)."""
    import smartrouter
    root = Path(smartrouter.__file__).resolve().parents[2]
    path = root / "results" / "stats" / f"{scenario}.json"
    return load(path) if path.exists() else None


def seed_for(order_id: str) -> int:
    return zlib.crc32(order_id.encode("utf-8")) % 1_000_000 + 1


def run_parent(side: int, strategy: str = "smart", scenario: str = "baseline", seed: int = 1, urgency: str = "medium",
               qty: int | None = None) -> dict:
    """Run one standard parent order through the router's simulated market; returns its metrics dict."""
    ParentConfig, preset, run_episode, load = _import()
    sc = preset(scenario)
    horizon, slices = URGENCY.get(urgency, URGENCY["medium"])
    base = sc.parent.to_dict()
    base.update(side=int(side), horizon_s=horizon, slices=slices)
    if qty:
        base["qty"] = int(qty)
    stats = _stats(load, scenario) if strategy in ("model", "post_queue", "ck", "smart") else None
    return run_episode(sc, strategy, seed, stats=stats, parent=ParentConfig.from_dict(base))


def route(orders: list[dict], strategy: str | None = None, scenario: str | None = None) -> list[dict]:
    """qpm.parent_order/1 dicts in, qpm.execution_report/1 dicts out (see docs/EXECUTION_INTERFACE.md)."""
    strategy = strategy or os.environ.get("QPM_SMARTROUTER_STRATEGY", "smart")
    scenario = scenario or os.environ.get("QPM_SMARTROUTER_SCENARIO", "baseline")
    out = []
    for o in orders:
        side = 1 if o["side"] == "buy" else -1
        res = run_parent(side, strategy, scenario, seed_for(o["order_id"]), o.get("urgency", "medium"))
        fill_frac = res["filled"] / res["qty"] if res["qty"] else 0.0
        filled = float(o["quantity"]) * fill_frac
        arrival = float(o["arrival_price"])
        notional = float(o.get("notional") or arrival * float(o["quantity"]))
        per_share_bps = res["price_bps"] / fill_frac if fill_frac > 0 else 0.0   # router normalises by the whole order
        avg = arrival * (1 + side * per_share_bps / 1e4)
        fees = notional * res["fee_bps"] / 1e4
        venues = {k: v for k, v in res.get("venue_qty", {}).items() if v}
        total = sum(venues.values()) or 1
        fills = [{"venue": k, "quantity": filled * v / total, "price": avg, "liquidity": "mixed", "fee": fees * v / total}
                 for k, v in venues.items()] if filled > 0 else []
        shortfall = 1e4 * (side * (avg - arrival) * filled + fees) / (arrival * filled) if filled > 0 else 0.0
        out.append({
            "schema": "qpm.execution_report/1", "order_id": o["order_id"], "instrument": o["instrument"], "side": o["side"],
            "filled_quantity": filled, "avg_price": avg, "arrival_price": arrival, "fees": fees,
            "status": "filled" if fill_frac >= 1 - 1e-12 else ("partial" if filled > 0 else "cancelled"),
            "fills": fills, "shortfall_bps": shortfall,
            "breakdown_bps": {"price": res["price_bps"], "fees": res["fee_bps"], "opportunity": res["opp_bps"],
                              "router_is": res["is_bps"], "maker_share": res["maker_share"]},
            "executor": f"smartrouter:{strategy}@{scenario}",
            "message": f"router seed {seed_for(o['order_id'])}; standard parent order, {o.get('urgency', 'medium')} urgency",
        })
    return out


def cost_table(strategies=("nbbo", "sync", "smart"), seeds=range(1, 13), scenario: str = "baseline",
               urgency: str = "medium") -> dict:
    """Average router shortfall (bps of arrival mid) per strategy over buy and sell orders and the given seeds."""
    table = {}
    for s in strategies:
        rows = [run_parent(side, s, scenario, seed, urgency) for seed in seeds for side in (1, -1)]
        n = len(rows)
        table[s] = {"is_bps": sum(r["is_bps"] for r in rows) / n, "price_bps": sum(r["price_bps"] for r in rows) / n,
                    "fee_bps": sum(r["fee_bps"] for r in rows) / n, "opp_bps": sum(r["opp_bps"] for r in rows) / n,
                    "fill_rate": sum(r["filled"] / r["qty"] for r in rows) / n, "episodes": n}
    return table
