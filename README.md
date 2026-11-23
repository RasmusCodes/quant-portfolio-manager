# quant-portfolio-manager

A portfolio construction and risk engine that turns signals from independent research systems into positions,
explains every weight, stress-tests the result and backtests it walk-forward with realistic costs, plus a
pre-registered study of what optimization is and is not worth on 55 years of public data.

**Live demo:** [rasmuscodes.github.io/projects/portfolio-manager](https://rasmuscodes.github.io/projects/portfolio-manager/)
builds, explains, stresses and backtests portfolios in your browser with a JavaScript port of this engine.
**Research report:** [reports/REPORT.md](reports/REPORT.md). **Design fixed before the runs:**
[reports/PREREGISTRATION.md](reports/PREREGISTRATION.md).

## What the study found

Monthly walk-forward tests, 1972-2026, after costs, on 12 US industry portfolios plus 10- and 5-year Treasuries
(and 49 industries and nine factor sleeves as robustness checks):

- **No optimizer beat equal weights by more than chance.** Sample-mean maximum Sharpe trailed 1/N by 0.13 Sharpe
  (95% interval −0.28 to +0.02); the best, Black-Litterman with momentum views, led by 0.06 (−0.03 to +0.15).
  Nothing survives a Holm correction.
- **Risk-based methods delivered the same Sharpe with far less risk:** minimum variance 7.7% volatility and an 18%
  worst drawdown, against 13.1% and 43% for 1/N.
- **Errors in expected returns cost six times more than errors in the covariance** (0.066 vs 0.011 of Sharpe in
  simulations with known parameters). Optimizers promised a Sharpe ratio of 0.79 and delivered 0.51.
- **Learned signal combinations beat equal weights, but only by catching a wrong-signed signal** (industry
  residual reversal has a negative IC); with that sign fixed by hand, equal weights did as well.
- **Kelly sizing on sample means grew fastest and fell 78-86% from peak.**

Optimization did not create alpha; better inputs and less risk did the work.

## What is inside

| | |
|---|---|
| Construction | Mean-variance, minimum variance, maximum Sharpe (exact, one QP), inverse-volatility risk parity, equal risk contribution, hierarchical risk parity, Black-Litterman, full and fractional Kelly, volatility targeting, factor-neutral and fully constrained optimization, efficient frontiers |
| Constraints | Position bounds, budget, net and gross exposure (leverage), sector / industry / asset-class limits, factor exposure bounds, turnover, trading costs, liquidity caps, volatility cap, drawdown cap |
| Solver | A dense ADMM QP solver (the OSQP algorithm) with an exact active-set polish; agrees with Clarabel to 1e-8 on random problems; the browser runs a line-by-line port |
| Explanations | Every mean-variance-type weight decomposed into its input return, constraint shadow costs, the part other assets replicate and the unhedgeable rest, with a sentence per asset |
| Signals | A schema every research system writes (timestamp, instrument, value, confidence, horizon, volatility, provenance, model version, track record), a point-in-time store, normalisation, and combination by equal weights, dynamic confidence, empirical-Bayes shrinkage, mean-variance on ICs, Bayesian model averaging, regime-dependent weights and a ridge ensemble |
| Risk | Historical / parametric / Cornish-Fisher / Monte Carlo / EWMA / filtered-historical VaR and ES with backtests, Euler and factor risk decomposition, concentration, configurable stress scenarios, liquidity, drawdowns, correlation breakdown |
| Backtesting | Walk-forward engine with proportional costs, square-root impact, borrow and financing costs, cash, drift, overlays, and an executor hook; Sharpe tests (Jobson-Korkie-Memmel, paired bootstrap, deflated Sharpe) |
| Execution | A two-message JSON contract (parent orders out, execution reports back), a built-in cost model, an adapter for external routers (command line or Python), and a bridge to the separate [multi-venue-smart-router](https://github.com/RasmusCodes/multi-venue-smart-router) repository |

Details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md), [docs/SIGNALS.md](docs/SIGNALS.md),
[docs/EXECUTION_INTERFACE.md](docs/EXECUTION_INTERFACE.md), [docs/RISK.md](docs/RISK.md).

## Quick start

```
python -m venv .venv
.venv\Scripts\activate            # or: source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
python -m pytest                  # under a minute; data-dependent tests skip without data
```

```python
import numpy as np, pandas as pd
from qpm.optimize import Constraints, max_sharpe, explain

names = ["Stocks", "Bonds", "Gold"]
mu = pd.Series([0.05, 0.015, 0.02], index=names)                 # expected excess returns
vol = np.array([0.16, 0.06, 0.15]); corr = np.array([[1, -0.1, 0.1], [-0.1, 1, 0.2], [0.1, 0.2, 1]])
cov = pd.DataFrame(corr * np.outer(vol, vol), index=names, columns=names)

p = max_sharpe(mu, cov, Constraints(lower=0.0, upper=0.6))
print(p.weights.round(3), p.sharpe)
for line in explain(p, cov, mu)["sentences"].values():
    print(line)
```

## Reproduce the study

```
python scripts/fetch_data.py french       # Ken French library, 14 small zips (about 6 MB)
python scripts/fetch_data.py fred         # H.15 Treasury yields and the St. Louis Fed stress index
python scripts/build_data.py              # data/processed/
python scripts/run_experiments.py         # E1-E10 plus exploratory X1-X2, about 35 minutes, results/
python scripts/make_report.py             # reports/figures/ and reports/tables/
python scripts/export_demo.py             # demo/data/ and the JavaScript parity fixtures
```

Exploratory X3 routes orders through the smart order router and needs that repository next to this one (or
`QPM_SMARTROUTER_SRC` pointing at its `src` folder): `python scripts/run_experiments.py --only X3`.

## Using your own research systems

Each system writes signals in the schema (CSV or JSON) and the manager reads them point in time:

```python
from qpm.signals import SignalStore, store_to_scores
store = SignalStore.read("my_system/signals.csv")
store.validate(universe=assets)
scores = store_to_scores(store.as_of("2026-09-30"), assets)      # instruments x sources, z-scored
```

`qpm.signals.adapters.statarb_signals` maps the statarb platform's export onto the schema. Execution goes through
any object with `execute(orders) -> reports`; see the interface document.

## The browser demo

`demo/` is a static page (no server, no external requests). `demo/tests.html` checks the JavaScript engine against
outputs of this Python engine (54 checks) and `demo/index.html?selftest=1` exercises every method and tab.
Open `demo/index.html` directly or serve the folder.

## Data and what is not in the repository

Raw downloads (`data/`) and experiment outputs (`results/`) are not committed; the scripts rebuild them. The demo
bundles monthly returns derived from the Kenneth R. French Data Library (© Eugene F. Fama and Kenneth R. French,
used with attribution) and Treasury returns rebuilt from public-domain Federal Reserve H.15 yields.

## Layout

```
src/qpm/         data, estimation, optimize, signals, risk, execution, backtest
scripts/         fetch_data, build_data, run_experiments, make_report, export_demo
scenarios/       stress scenarios as JSON
reports/         PREREGISTRATION.md, REPORT.md, figures/, tables/
demo/            browser demo and its parity tests
examples/        stub_router.py (stand-in router for the command-line transport)
tests/           pytest suite
docs/            architecture, signals, execution interface, risk
```

## License

MIT, see [LICENSE](LICENSE).
