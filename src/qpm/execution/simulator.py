"""Built-in executor: fills every parent order in full at a modelled cost.

cost per unit traded = half spread + commission + slippage + eta x daily_vol x sqrt(notional / ADV)

The last term is the square-root law of market impact (Almgren, Thum, Hauptmann and Li 2005; Toth et al.
2011: impact grows with the square root of the order's share of daily volume).  eta = 0.5-1 is typical.
"""
from __future__ import annotations

import math

from .interface import ExecutionReport, Fill, ParentOrder, implementation_shortfall_bps


class CostModelExecution:
    name = "cost-model"

    def __init__(self, spread_bps: float = 5.0, commission_bps: float = 1.0, slippage_bps: float = 0.0,
                 impact_eta: float = 0.5, default_daily_vol: float = 0.015, default_adv: float | None = None):
        self.spread_bps = spread_bps
        self.commission_bps = commission_bps
        self.slippage_bps = slippage_bps
        self.impact_eta = impact_eta
        self.default_daily_vol = default_daily_vol
        self.default_adv = default_adv

    def unit_cost(self, order: ParentOrder) -> dict:
        spread = (order.spread_bps if order.spread_bps is not None else self.spread_bps) / 2.0
        vol = order.daily_vol if order.daily_vol is not None else self.default_daily_vol
        adv = order.adv if order.adv is not None else self.default_adv
        impact = 0.0
        if adv and adv > 0 and self.impact_eta > 0:
            impact = 1e4 * self.impact_eta * vol * math.sqrt(order.notional / adv)
        return {"spread": spread, "slippage": self.slippage_bps, "impact": impact, "fees": self.commission_bps}

    def execute(self, orders: list[ParentOrder]) -> list[ExecutionReport]:
        reports = []
        for o in orders:
            c = self.unit_cost(o)
            px_bps = c["spread"] + c["slippage"] + c["impact"]
            sign = 1.0 if o.side == "buy" else -1.0
            price = o.arrival_price * (1.0 + sign * px_bps / 1e4)
            fee = o.notional * c["fees"] / 1e4
            fill = Fill("model", o.quantity, price, "taker", fee)
            is_bps = implementation_shortfall_bps(o.side, price, o.arrival_price, fee, o.quantity)
            reports.append(ExecutionReport(o.order_id, o.instrument, o.side, o.quantity, price, o.arrival_price, fee,
                                           "filled", [fill], is_bps, c, self.name))
        return reports
