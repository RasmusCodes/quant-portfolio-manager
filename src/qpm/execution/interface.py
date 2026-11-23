"""The execution contract between the portfolio manager and any executor.

The portfolio manager decides WHAT to trade (parent orders); an executor decides HOW (child orders,
venues, timing) and reports what happened.  The contract is two JSON messages, versioned by their
``schema`` field and documented in docs/EXECUTION_INTERFACE.md:

    qpm.parent_order/1      one per instrument to trade
    qpm.execution_report/1  one per parent order

Any object with ``execute(list[ParentOrder]) -> list[ExecutionReport]`` is an executor.  The built-in
``CostModelExecution`` is one; ``SmartOrderRouterAdapter`` forwards orders to an external smart order
router (a separate repository) through a Python entry point or a command-line process.

Sign convention: implementation shortfall is reported in basis points of arrival notional and is
POSITIVE when execution cost money (bought above / sold below the arrival price, plus fees).
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Protocol, runtime_checkable

ORDER_SCHEMA = "qpm.parent_order/1"
REPORT_SCHEMA = "qpm.execution_report/1"


@dataclass
class ParentOrder:
    order_id: str
    instrument: str
    side: str                       # "buy" | "sell"
    quantity: float                 # units (shares, contracts); positive
    arrival_price: float            # decision-time reference price
    decision_time: str              # ISO 8601
    notional: float = 0.0           # quantity x arrival_price (filled in if 0)
    urgency: str = "medium"         # low | medium | high
    participation: float | None = 0.10
    horizon_minutes: int = 390
    limit_price: float | None = None
    adv: float | None = None        # average daily traded value, same currency as notional
    daily_vol: float | None = None  # daily return volatility (decimal)
    spread_bps: float | None = None
    portfolio: str = ""
    strategy: str = ""
    schema: str = ORDER_SCHEMA

    def __post_init__(self):
        if self.side not in ("buy", "sell"):
            raise ValueError(f"side must be buy or sell, got {self.side!r}")
        if not (self.quantity > 0 and math.isfinite(self.quantity)):
            raise ValueError("quantity must be positive")
        if not (self.arrival_price > 0):
            raise ValueError("arrival_price must be positive")
        if not self.notional:
            self.notional = self.quantity * self.arrival_price

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ParentOrder":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Fill:
    venue: str
    quantity: float
    price: float
    liquidity: str = "taker"        # maker | taker | auction | midpoint | dark
    fee: float = 0.0                # currency; negative = rebate


@dataclass
class ExecutionReport:
    order_id: str
    instrument: str
    side: str
    filled_quantity: float
    avg_price: float
    arrival_price: float
    fees: float = 0.0
    status: str = "filled"          # filled | partial | rejected | cancelled
    fills: list = field(default_factory=list)
    shortfall_bps: float = 0.0      # positive = cost, includes fees
    breakdown_bps: dict = field(default_factory=dict)  # spread, impact, timing, fees, ...
    executor: str = ""
    message: str = ""
    schema: str = REPORT_SCHEMA

    def to_dict(self) -> dict:
        d = asdict(self)
        d["fills"] = [asdict(f) if isinstance(f, Fill) else dict(f) for f in self.fills]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExecutionReport":
        data = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        data["fills"] = [Fill(**{k: v for k, v in f.items() if k in Fill.__dataclass_fields__}) for f in d.get("fills", [])]
        return cls(**data)

    def cost_fraction(self, order: ParentOrder) -> float:
        """Execution cost as a fraction of the order's arrival notional (shortfall on the filled part plus
        fees).  Unfilled quantity is reported separately (opportunity cost is the caller's to judge)."""
        if self.filled_quantity <= 0:
            return 0.0
        sign = 1.0 if self.side == "buy" else -1.0
        paid = sign * (self.avg_price - self.arrival_price) * self.filled_quantity
        return (paid + self.fees) / order.notional


@runtime_checkable
class Executor(Protocol):
    name: str

    def execute(self, orders: list[ParentOrder]) -> list[ExecutionReport]:
        ...


def implementation_shortfall_bps(side: str, avg_price: float, arrival: float, fees: float, quantity: float) -> float:
    sign = 1.0 if side == "buy" else -1.0
    notional = arrival * quantity
    return 1e4 * (sign * (avg_price - arrival) * quantity + fees) / notional if notional else 0.0


def validate_report(order: ParentOrder, report: ExecutionReport, tol: float = 1e-6) -> list[str]:
    """Contract checks every executor's reports must pass (used by tests and at run time)."""
    problems = []
    if report.schema != REPORT_SCHEMA:
        problems.append(f"schema {report.schema!r} != {REPORT_SCHEMA!r}")
    if report.order_id != order.order_id:
        problems.append("order_id does not match")
    if report.instrument != order.instrument or report.side != order.side:
        problems.append("instrument/side do not match the order")
    if report.filled_quantity < -tol or report.filled_quantity > order.quantity * (1 + tol) + tol:
        problems.append(f"filled {report.filled_quantity} outside [0, {order.quantity}]")
    if report.filled_quantity > tol:
        if not (report.avg_price > 0):
            problems.append("avg_price must be positive when something filled")
        if report.fills:
            q = sum(f.quantity for f in report.fills)
            if abs(q - report.filled_quantity) > tol * max(1.0, order.quantity):
                problems.append(f"fills add up to {q}, report says {report.filled_quantity}")
            vwap = sum(f.quantity * f.price for f in report.fills) / q if q else 0.0
            if abs(vwap - report.avg_price) > 1e-6 * report.avg_price:
                problems.append("avg_price is not the fills' VWAP")
            fee_sum = sum(f.fee for f in report.fills)
            if abs(fee_sum - report.fees) > 1e-6 * max(1.0, abs(report.fees)):
                problems.append("fees do not add up over fills")
        is_bps = implementation_shortfall_bps(order.side, report.avg_price, order.arrival_price, report.fees,
                                              report.filled_quantity)
        if abs(is_bps - report.shortfall_bps) > 1e-3 + 1e-6 * abs(is_bps):
            problems.append(f"shortfall_bps {report.shortfall_bps:.4f} != recomputed {is_bps:.4f}")
    expected = "filled" if report.filled_quantity >= order.quantity * (1 - 1e-9) else ("partial" if report.filled_quantity > tol else None)
    if expected and report.status not in (expected, "cancelled") and not (expected == "partial" and report.status == "filled"):
        problems.append(f"status {report.status!r} inconsistent with fill ({expected})")
    return problems
