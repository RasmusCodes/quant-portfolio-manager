# Execution interface

The portfolio manager decides what to trade; an executor decides how. They exchange two versioned JSON messages,
so any executor (the built-in cost model, an external smart order router, a broker gateway) can be swapped in
without changing either side.

## Parent order: `qpm.parent_order/1`

```json
{"schema": "qpm.parent_order/1", "order_id": "2026-09-30-BusEq", "instrument": "BusEq", "side": "buy",
 "quantity": 50000, "arrival_price": 100.0, "decision_time": "2026-09-30T16:00:00", "notional": 5000000.0,
 "urgency": "medium", "participation": 0.1, "horizon_minutes": 390, "limit_price": null,
 "adv": 1.27e11, "daily_vol": 0.012, "spread_bps": 20.0, "portfolio": "", "strategy": ""}
```

`side` is `buy` or `sell`, `quantity` and `arrival_price` are positive, `notional` defaults to their product,
`urgency` is `low`, `medium` or `high`. `adv` (average daily traded value), `daily_vol` and `spread_bps` are hints
for executors that model costs.

## Execution report: `qpm.execution_report/1`

```json
{"schema": "qpm.execution_report/1", "order_id": "2026-09-30-BusEq", "instrument": "BusEq", "side": "buy",
 "filled_quantity": 50000, "avg_price": 100.031, "arrival_price": 100.0, "fees": 112.0, "status": "filled",
 "fills": [{"venue": "VENUE_A", "quantity": 25000, "price": 100.031, "liquidity": "taker", "fee": 56.0},
           {"venue": "VENUE_B", "quantity": 25000, "price": 100.031, "liquidity": "taker", "fee": 56.0}],
 "shortfall_bps": 3.32, "breakdown_bps": {"spread": 1.0, "impact": 2.1}, "executor": "cost-model", "message": ""}
```

Implementation shortfall is in basis points of the arrival notional of the filled quantity and is POSITIVE when
execution cost money: `1e4 x (side x (avg_price - arrival) x filled + fees) / (arrival x filled)` with side +1
for buys and -1 for sells.

## The contract (`qpm.execution.validate_report`)

Every report must: carry the right schema, order id, instrument and side; fill between 0 and the ordered quantity;
have a positive average price when anything filled; have fills that add up to the filled quantity, whose VWAP is
the average price and whose fees add up to the total; report a shortfall equal to the formula above; and use a
status consistent with the fill. `SmartOrderRouterAdapter` checks every report and raises `RouterContractError`
on a violation instead of letting a bad fill into a backtest.

## Transports

```python
from qpm.execution import SmartOrderRouterAdapter

# 1. a command that reads parent orders as JSON lines on stdin and writes reports as JSON lines on stdout
router = SmartOrderRouterAdapter("cli", ["python", "examples/stub_router.py"])

# 2. a Python callable "package.module:function" taking and returning lists of dicts
router = SmartOrderRouterAdapter("python", callable_path="qpm.execution.smartrouter_bridge:route")

reports = router.execute(orders)                  # list[ExecutionReport]
config = BacktestConfig(costs=CostModel(bps=..., aum=1e9, adv=adv), executor=router)
```

In a backtest the engine turns each rebalance's trades into parent orders (prices are the assets' total-return
indices, notional = |trade| x capital), sends them, and charges the reported shortfall and fees. With the
built-in `CostModelExecution` the result equals the vectorised cost model exactly (tested).

## The smart order router (separate repository)

`qpm.execution.smartrouter_bridge` connects to [multi-venue-smart-router](https://github.com/RasmusCodes/multi-venue-smart-router),
a latency-aware simulator of a fragmented equity market with eleven routing strategies. Install it next to qpm
(`pip install -e ../multi-venue-smart-router`) or set `QPM_SMARTROUTER_SRC` to its `src` folder. For each parent
order the bridge runs one standard parent order through the router's simulated market (strategy and scenario from
`QPM_SMARTROUTER_STRATEGY` / `QPM_SMARTROUTER_SCENARIO`, default `smart` / `baseline`; urgency sets the horizon:
600, 300 or 60 seconds), seeds it from the order id so results are reproducible, and maps the router's measured
shortfall per share, fees, fill rate and venue split onto the qpm order.

The router knows nothing about an industry portfolio's liquidity, so size-dependent impact for a large book stays
with qpm's own square-root model; what the router adds is the cost of how an order is routed and the difference
between routing strategies. `cost_table()` measures the average shortfall per strategy for use as a cost
assumption. One router episode takes about two seconds, so routing every order of a 50-year monthly backtest
through it is impractical; route single rebalances live, or calibrate costs with `cost_table()`.

`examples/stub_router.py` is a stand-in that speaks the contract over the command line (three fictional venues,
fixed fees, square-root impact). It exists so the command-line path is tested without the router repository.
