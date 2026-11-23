"""Walk-forward portfolio backtester.

Timing (monthly data): at the close of month t the strategy sees returns, signals and fundamentals known
by t (the context is sliced with ``.loc[:t]``, so it cannot see later rows), picks target weights, trades
at that close paying costs, then earns month t+1 returns.  Between rebalances weights drift with returns.
Cash (1 - sum of weights) earns the T-bill rate; shorts pay a borrow fee; negative cash pays the bill rate
plus a financing spread.

Costs per unit of weight traded = proportional cost (half spread + commission + slippage) +
eta x daily volatility x sqrt(trade value / ADV) (square-root impact), or whatever an ``Executor`` (for
example the smart-order-router adapter) reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from ..execution.interface import Executor, ParentOrder


@dataclass
class CostModel:
    bps: float | dict = 10.0                   # one-way cost per unit traded, scalar or {asset: bps}
    impact_eta: float = 0.0                    # square-root impact coefficient (0 = off)
    aum: float = 1e8                           # capital in currency units (for impact and orders)
    adv: pd.DataFrame | None = None            # dates x assets average daily traded value (currency)

    def per_asset_bps(self, assets) -> np.ndarray:
        if isinstance(self.bps, dict):
            default = self.bps.get("default", 10.0)
            return np.array([float(self.bps.get(a, default)) for a in assets])
        return np.full(len(assets), float(self.bps))


@dataclass
class BacktestConfig:
    rebalance: str = "M"                       # M (monthly), Q (quarterly), A (annual, December)
    window: int = 120                          # months of history required before the first decision
    start: str | None = None                   # first decision date (default: after ``window`` months)
    end: str | None = None
    costs: CostModel = field(default_factory=CostModel)
    executor: Executor | None = None           # replaces the cost model when given
    borrow_bps: float = 30.0                   # annual fee on short notional
    financing_spread_bps: float = 0.0          # annual spread over the bill rate on borrowed cash
    vol_target: float | None = None            # annualised; scales the target using trailing covariance
    max_leverage: float = 1.0                  # gross cap for the vol-target overlay
    drawdown_limit: float | None = None        # drawdown governor: exposure x clip((limit - dd) / limit, floor, 1)
    drawdown_floor: float = 0.25
    name: str = ""


@dataclass
class DecisionContext:
    date: pd.Timestamp
    returns: pd.DataFrame          # history through ``date``
    rf: pd.Series
    weights: pd.Series             # current (drifted) weights before trading
    universe: object | None = None # qpm.data.Universe sliced through ``date``
    signals: dict = field(default_factory=dict)  # name -> panel sliced through ``date``
    state: dict = field(default_factory=dict)    # strategy's own memory between decisions


class Strategy(Protocol):
    name: str

    def target(self, ctx: DecisionContext) -> pd.Series:
        ...


@dataclass
class BacktestResult:
    name: str
    returns: pd.Series             # net monthly returns (after costs)
    gross_returns: pd.Series
    rf: pd.Series
    weights: pd.DataFrame          # post-trade weights at each decision date
    turnover: pd.Series            # sum |trade| at each decision date (fraction of capital)
    costs: pd.Series               # cost paid at each decision date (fraction of capital)
    exposure: pd.DataFrame         # gross, net, cash per period
    log: list = field(default_factory=list)

    @property
    def nav(self) -> pd.Series:
        return (1 + self.returns).cumprod()

    @property
    def excess(self) -> pd.Series:
        return self.returns - self.rf.reindex(self.returns.index)

    def stats(self, periods_per_year: int = 12) -> dict:
        from .metrics import performance_stats
        s = performance_stats(self.returns, self.rf, periods_per_year)
        years = len(self.returns) / periods_per_year
        s["turnover"] = float(self.turnover.sum() / years) if years > 0 else np.nan
        s["cost_drag"] = float(self.costs.sum() / years) if years > 0 else np.nan
        return s


def rebalance_mask(dates: pd.DatetimeIndex, freq: str) -> np.ndarray:
    if freq == "M":
        return np.ones(len(dates), dtype=bool)
    if freq == "Q":
        return np.isin(dates.month, (3, 6, 9, 12))
    if freq == "A":
        return dates.month == 12
    raise ValueError("rebalance must be M, Q or A")


def _slice_universe(universe, date):
    if universe is None:
        return None
    return universe.window(None, date)


def _drawdown(nav_path: list[float]) -> float:
    if not nav_path:
        return 0.0
    peak = max(1.0, max(nav_path))
    return 1.0 - nav_path[-1] / peak


def run_backtest(strategy: Strategy, returns: pd.DataFrame, rf: pd.Series, config: BacktestConfig | None = None,
                 universe=None, signals: dict | None = None, daily_vol: pd.DataFrame | None = None,
                 prices: pd.DataFrame | None = None) -> BacktestResult:
    cfg = config or BacktestConfig()
    returns = returns.sort_index()
    rf = rf.reindex(returns.index).fillna(0.0)
    assets = list(returns.columns)
    dates = returns.index
    first = cfg.window
    if cfg.start is not None:
        first = max(first, int(dates.searchsorted(pd.Timestamp(cfg.start))))
    last = len(dates) - 1 if cfg.end is None else min(len(dates) - 1, int(dates.searchsorted(pd.Timestamp(cfg.end), "right")) - 1)
    if first >= last:
        raise ValueError("not enough data for the window and sample period")
    mask = rebalance_mask(dates, cfg.rebalance)
    bps = cfg.costs.per_asset_bps(assets) / 1e4
    if daily_vol is None:
        daily_vol = returns.rolling(36, min_periods=12).std() / np.sqrt(21.0)
    if prices is None:
        prices = 100.0 * (1 + returns.fillna(0.0)).cumprod()
    signals = signals or {}

    w = np.zeros(len(assets))
    nav = 1.0
    nav_path: list[float] = []
    state: dict = {}
    out_net, out_gross, out_dates = [], [], []
    w_rec, to_rec, cost_rec, rec_dates, expo = [], [], [], [], []
    log = []
    for t in range(first, last):
        date = dates[t]
        cost = 0.0
        if mask[t] or t == first:
            ctx = DecisionContext(date, returns.iloc[:t + 1], rf.iloc[:t + 1], pd.Series(w, index=assets),
                                  _slice_universe(universe, date),
                                  {k: v.loc[:date] for k, v in signals.items()}, state)
            try:
                target = strategy.target(ctx)
                target = target.reindex(assets).fillna(0.0).values.astype(float)
                if not np.all(np.isfinite(target)):
                    raise ValueError("non-finite target weights")
            except Exception as exc:  # keep the current book and record why
                log.append((date, f"{type(exc).__name__}: {exc}"))
                target = w.copy()
            if cfg.vol_target is not None and np.abs(target).sum() > 0:
                hist = returns.iloc[max(0, t + 1 - cfg.window):t + 1]
                cov = np.cov(hist.values, rowvar=False) * 12
                vol = float(np.sqrt(max(target @ cov @ target, 0.0)))
                if vol > 0:
                    scale = min(cfg.vol_target / vol, cfg.max_leverage / np.abs(target).sum())
                    target = target * scale
            if cfg.drawdown_limit is not None:
                dd = _drawdown(nav_path)
                target = target * float(np.clip((cfg.drawdown_limit - dd) / cfg.drawdown_limit, cfg.drawdown_floor, 1.0))
            trade = target - w
            traded = np.abs(trade)
            if cfg.executor is not None and traded.sum() > 1e-12:
                cost = _execute(cfg, trade, assets, date, nav, prices.iloc[t], daily_vol.iloc[t], t)
            else:
                cost = float(traded @ bps)
                if cfg.costs.impact_eta > 0 and cfg.costs.adv is not None:
                    adv = cfg.costs.adv.reindex(index=[date], columns=assets).ffill().values[0] if date in cfg.costs.adv.index \
                        else cfg.costs.adv.reindex(columns=assets).loc[:date].iloc[-1].values
                    dv = np.nan_to_num(daily_vol.iloc[t].values, nan=0.01)
                    impact = cfg.costs.impact_eta * dv * np.sqrt(traded * cfg.costs.aum * nav / np.where(adv > 0, adv, np.inf))
                    cost += float(traded @ impact)
            nav *= (1.0 - cost)
            w = target
            w_rec.append(w.copy())
            to_rec.append(float(traded.sum()))
            cost_rec.append(cost)
            rec_dates.append(date)
        r = returns.iloc[t + 1].values
        r = np.nan_to_num(r, nan=0.0)
        cash = 1.0 - w.sum()
        short = np.clip(-w, 0, None).sum()
        port = float(w @ r + cash * rf.iloc[t + 1] - cfg.borrow_bps / 1e4 / 12 * short
                     - cfg.financing_spread_bps / 1e4 / 12 * max(-cash, 0.0))
        nav_before = nav / (1.0 - cost) if cost < 1 else nav
        nav *= (1.0 + port)
        nav_path.append(nav)
        out_dates.append(dates[t + 1])
        out_gross.append(port)
        out_net.append(nav / nav_before - 1.0)
        expo.append((np.abs(w).sum(), w.sum(), cash))
        grow = w * (1.0 + r)
        denom = 1.0 + port
        w = grow / denom if denom != 0 else grow
    idx = pd.DatetimeIndex(out_dates)
    rd = pd.DatetimeIndex(rec_dates)
    return BacktestResult(cfg.name or getattr(strategy, "name", "strategy"), pd.Series(out_net, idx),
                          pd.Series(out_gross, idx), rf.reindex(idx), pd.DataFrame(w_rec, rd, assets),
                          pd.Series(to_rec, rd), pd.Series(cost_rec, rd),
                          pd.DataFrame(expo, idx, ["gross", "net", "cash"]), log)


def _execute(cfg: BacktestConfig, trade, assets, date, nav, prices, daily_vol, t) -> float:
    """Send the trades to the executor as parent orders; return the cost as a fraction of capital."""
    capital = cfg.costs.aum * nav
    orders = []
    adv_row = None
    if cfg.costs.adv is not None:
        adv_row = cfg.costs.adv.reindex(columns=assets).loc[:date].iloc[-1]
    bps = cfg.costs.per_asset_bps(assets)
    for i, a in enumerate(assets):
        notional = abs(trade[i]) * capital
        if notional < 1e-9 * capital:
            continue
        px = float(prices.iloc[i])
        # the cost model's proportional cost is a half spread, so the quoted spread sent is twice it
        orders.append(ParentOrder(order_id=f"{date:%Y%m%d}-{a}", instrument=a, side="buy" if trade[i] > 0 else "sell",
                                  quantity=notional / px, arrival_price=px, decision_time=f"{date:%Y-%m-%d}T16:00:00",
                                  notional=notional, adv=None if adv_row is None else float(adv_row.iloc[i]),
                                  daily_vol=float(np.nan_to_num(daily_vol.iloc[i], nan=0.01)), spread_bps=2.0 * float(bps[i])))
    reports = cfg.executor.execute(orders)
    total = sum(rep.cost_fraction(o) * o.notional for o, rep in zip(orders, reports))
    return float(total / capital)
