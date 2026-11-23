# Architecture

```
research systems ──signals (qpm.signals/1)──► SignalStore ──as_of(t)──► combination ──► expected returns
                                                                                            │
public data ──► data/ (French, FRED) ──► Universe ──► estimators (means, covariances) ─────┤
                                                                                            ▼
                                       constraints ──► construction methods (QP solver) ──► target weights
                                                                                            │
                     risk engine ◄──────────────────────────────────────────────────────────┤
                (VaR/ES, decomposition, stress, liquidity, drawdowns)                        ▼
                                                     backtester ──parent orders (qpm.parent_order/1)──► executor
                                                          ▲                                   (cost model, router)
                                                          └──────execution reports (qpm.execution_report/1)
```

## Packages (`src/qpm`)

| Package | Contents |
|---|---|
| `data` | Readers for Ken French CSV zips (sections, missing codes, annual blocks, June-formation BE/ME made point in time) and FRED series; constant-maturity Treasury returns rebuilt from H.15 yields; the universes (`multi_asset`, `industries12`, `industries49`, `sleeves`). |
| `estimation` | Sample, EWMA, Ledoit-Wolf (identity and constant-correlation targets), observable-factor and PCA covariances, nearest-PSD repair; sample, Bayes-Stein, James-Stein and equilibrium means, Grinold alphas. |
| `optimize` | `qp.py`: a dense ADMM solver for convex QPs (the OSQP algorithm: Ruiz scaling, adaptive step, infeasibility certificates) with an active-set polish that makes solutions exact. `constraints.py`: bounds, budget, net and gross exposure, group (sector/industry/asset-class) limits, factor exposure bounds, turnover, linear trading costs, per-asset trade caps, liquidity caps, volatility cap, drawdown cap; every constraint row is labelled. `methods.py`: the construction methods. `explain.py`: why each weight. `frontier.py`. |
| `signals` | Schema and point-in-time store, normalisation, track records, combination methods, public reference signals, a synthetic signal lab, adapters. |
| `risk` | VaR and expected shortfall (historical, normal, Student t, Cornish-Fisher, Monte Carlo, EWMA, filtered historical simulation) with Kupiec and Christoffersen backtests; Euler risk contributions, component ES, factor decomposition, concentration measures; JSON stress scenarios; liquidity profile and liquidation cost; drawdowns, CDaR, correlation breakdown, absorption ratio. |
| `execution` | The parent-order / execution-report contract, the built-in cost-model executor, the adapter for external routers, the bridge to the smart order router repository. |
| `backtest` | Walk-forward engine (costs, square-root impact, borrow, financing, cash, drift, volatility-target and drawdown overlays, executor hook), strategies, performance statistics and Sharpe-difference tests (Jobson-Korkie-Memmel, paired stationary bootstrap, deflated Sharpe ratio). |

## Construction methods

| Key | Method | Uses expected returns |
|---|---|---|
| `equal_weight` | 1/N | no |
| `market_cap` | Market (or 60/40) weights | no |
| `inverse_vol` | Naive risk parity, w proportional to 1/vol | no |
| `min_variance` | Minimum variance under constraints | no |
| `erc` | Equal risk contribution / risk budgeting (cyclical coordinate descent on Spinu's convex form) | no |
| `hrp` | Hierarchical risk parity (López de Prado 2016; single linkage on the distance of distances, as the paper's code) | no |
| `mean_variance` | max mu'w - (lambda/2) w'Cov w under constraints; optional volatility cap by bisection on lambda | yes |
| `max_sharpe` | Exact maximum Sharpe as one homogenized QP (works for any linear constraints) | yes |
| `black_litterman` | He-Litterman posterior from equilibrium returns and views with confidences, then mean-variance | prior + views |
| `kelly`, `kelly_half` | Growth-optimal sizing under the normal approximation (mean-variance with risk aversion 1/fraction, budget free) | yes |
| `vol_target` | Overlay that scales any portfolio to a volatility target within a leverage cap | — |
| `factor_neutral` | Mean-variance with zero exposure to chosen factors, dollar neutral | yes |
| `constrained` | Mean-variance or minimum variance with any mix of constraints | yes |

## Explaining weights

For a mean-variance-type solution the KKT conditions give lambda Cov w = mu - A'y, so the weights are an
unconstrained solution for "effective" expected returns (inputs minus the shadow costs of binding constraints).
Inverting asset by asset (Stevens 1998), w_i = (mu_eff_i - b_i'mu_eff_-i) / (lambda s_i^2), where b_i are the
coefficients of a regression of asset i on the others and s_i^2 its residual variance. `explain()` reports, per
asset, the input return, each constraint family's adjustment, the return explained by the other assets, the unique
part, the unhedgeable volatility and a sentence; tests check the formula reproduces the solver's weights to 1e-8.

## Browser demo

`demo/` is a static site: `engine.js` is a line-by-line port of the solver, estimators, methods, explanations,
risk, stress and backtester; `charts.js` draws SVG charts; `app.js` is the interface; `data/bundle.js` is written
by `scripts/export_demo.py`. `demo/tests.html` checks the JavaScript engine against fixtures produced by the Python
engine (54 checks: every method's weights, the explanation formula, Ledoit-Wolf, Bayes-Stein, VaR) and
`index.html?selftest=1` runs every method, tab, the estimation-error lab, a backtest and every scenario.

## Testing

`python -m pytest` runs 109 tests in under a minute: KKT conditions and agreement with Clarabel to 1e-6 on random
QPs, closed forms (minimum variance, mean-variance, tangency, Kelly), scipy's linkage, sklearn's Ledoit-Wolf, a loop
transcription of Ledoit-Wolf's covCor, look-ahead tests for the backtester and every combination method, the
execution contract over both transports, the parsers, and the data build when present.
