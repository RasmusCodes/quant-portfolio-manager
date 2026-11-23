# Signal interface

Research systems never call the portfolio manager and the portfolio manager never imports them. A research
system writes records in the schema below (CSV, JSON or a pandas frame); `qpm.signals.SignalStore` reads them,
validates them and serves them point in time.

## The record

| Field | Type | Meaning |
|---|---|---|
| `timestamp` | datetime | When the signal became known. A decision at time t may only use records with `timestamp <= t`. |
| `instrument` | string | Identifier in the portfolio's universe. |
| `value` | float | Expected return over the horizon (`kind = expected_return`, decimal), a score, or +1/0/-1. |
| `kind` | `expected_return` \| `score` \| `direction` | How to read `value`. |
| `confidence` | float in [0, 1] | The producer's own confidence (used by the `confidence` combination). |
| `horizon_days` | int > 0 | Holding horizon in trading days; the record expires after it. |
| `volatility` | float, optional | Annualised volatility of the instrument, for score-to-return conversion. |
| `source` | string | Research system and model, e.g. `statarb.residual_reversal`. |
| `family` | see below | One of the eight research families. |
| `model_version` | string | Version of the producing model. |
| `provenance` | string | Data sources and the as-of dates of the inputs. |
| `ic_mean`, `ic_t`, `hit_rate`, `n_obs` | optional | The producer's track record as of `timestamp`. |

Families: `statistical_arbitrage`, `equity_factor`, `options_volatility`, `alternative_data`, `earnings_event`,
`credit`, `macro_fixed_income`, `microstructure`.

JSON files carry `{"schema": "qpm.signals/1", "signals": [ ... ]}`; CSV files have one column per field.

```python
from qpm.signals import SignalStore
store = SignalStore.read("my_system/signals.csv")
store.validate(universe=assets)          # raises SignalValidationError listing every problem
snapshot = store.as_of("2026-09-30")    # latest unexpired record per (source, instrument), never later ones
```

Validation rejects unknown families or kinds, confidence outside [0, 1], non-positive horizons, non-finite values,
direction values other than -1/0/1, duplicate (timestamp, source, instrument) rows and instruments outside the
universe.

## Point-in-time rules

- `SignalStore.as_of(t)` filters `timestamp <= t` and drops records older than their horizon (trading days at
  365/252 calendar days each). It asserts the filter before returning.
- Track records use forward returns stamped at the formation date: the IC stamped t uses the return over t..t+1,
  which is only known at t+1, so `track_record_asof(t)` and every combination method use ICs up to t-1 only.
- The backtester hands strategies `DecisionContext` objects whose returns and signal panels are sliced with
  `.loc[:t]`. Tests rewrite the future (multiply later returns by -3) and check that no earlier decision moves.

## Normalisation

`store_to_scores(snapshot, instruments)` turns each source into a cross-sectional z-score: expected-return records
are converted to the rebalancing horizon (linear in time) and divided by volatility first; scores are winsorised at
three standard deviations and re-standardised. `grinold_expected_return(z, vol, ic)` maps a combined z-score to
an expected return, `alpha = IC x volatility x z` (Grinold and Kahn 2000).

## Combination methods (`qpm.signals.combine`)

All run walk-forward on panels of z-scores and forward returns and output z-scored combined signals.

| Method | Weights at date t |
|---|---|
| `equal` | 1/K over the signals present |
| `ic_weighted` | positive part of an EWMA of past ICs ("dynamic confidence") |
| `shrunk` | past mean ICs shrunk toward their cross-signal average by empirical Bayes (Morris 1983) |
| `mean_var` | Cov(IC)^-1 mean(IC), correlations shrunk halfway to zero, long-only |
| `bma` | Bayesian model averaging over all 2^K subsets of signals in a pooled cross-sectional regression with Zellner's g-prior (g = n); coefficients may be negative, and posterior inclusion probabilities are kept |
| `regime` | mean IC within the current regime, shrunk toward the unconditional mean (prior weight 24 months) |
| `ridge` | pooled ridge regression of next returns on all signals, penalty chosen by blocked cross-validation inside the trailing window |
| `confidence` | static weights from the producers' confidence numbers |

The study (reports/REPORT.md, E4 and E5) shows when each helps: learned weights beat equal weights mainly when
some input is junk or wrongly signed.

## Adapters

- `adapters.read_signal_file(path)`: any file already in the schema (the default path for every research system).
- `adapters.statarb_signals(path)`: the statarb platform's dashboard export (`results/dashboard/signals.parquet`,
  S&P 500 stocks by month, nine z-scored signals). Its `madj_next` column is the next month's return, a label from
  the future: the adapter never reads it as a signal; `statarb_realized_returns` exposes it for track records only.
- The reference systems in `qpm.signals.reference` (momentum, value, low beta, residual reversal, time-series
  momentum, bond carry) are built from public data and produce panels in the same shape.
