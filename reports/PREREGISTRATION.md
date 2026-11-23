# Pre-registration: portfolio construction study

Written 2026-11-21, before the experiments below were run. Settings here are fixed; anything changed
afterwards is listed in the report's "Deviations" section with the reason.

**Disclosure.** While building the backtester I ran one smoke test of nine methods on the multi-asset
universe (window 120, costs 10/2 bp, caps 40%): equal weight Sharpe 0.51, sample-mean maximum Sharpe 0.26
(Jobson-Korkie p = 0.009), minimum variance 0.44, ERC 0.52, HRP 0.48. H1 and H2 were therefore not formed
blind; they also follow DeMiguel, Garlappi and Uppal (2009). The main runs below use different position caps
(25%) and add tests, shrinkage variants and other universes.

## Data

Public data only: Ken French industry portfolios (value-weighted, total returns), French factors, H.15
constant-maturity Treasury yields rebuilt into total returns, one-month T-bill as cash. Monthly, through
2026-09.

| Universe | Assets | Returns from | First decision |
|---|---|---|---|
| multi_asset | 12 industries + 10y and 5y Treasuries | 1962-02 | 1972-01 |
| industries49 | 49 industries (11 sectors, my grouping) | 1969-07 | 1979-06 |
| sleeves | Mkt-RF, SMB, HML, RMW, CMA, Mom, ST_Rev, LT_Rev, Term (10y - bills) | 1963-07 | 1973-06 |

## Common settings

Rolling 120-month estimation window, monthly rebalancing at month-end close, trading costs per unit of
weight traded of 10 bp (industries, sleeves) and 2 bp (Treasuries) in the base case and 30/5 bp in the
high-cost case, borrow 30 bp a year on shorts. Long-only with a 25% position cap (multi_asset), 10%
(industries49) or 40% (sleeves). Covariance: Ledoit-Wolf constant-correlation shrinkage unless the
experiment varies it. Mean-variance risk aversion 4 (annual units). No parameter is tuned on the
evaluation period.

## Experiments and hypotheses

**E1 Horse race (multi_asset).** Baselines: equal weight (1/N), 60/40 (60% cap-weighted industries, 20%
each Treasury), inverse volatility. Risk-based: minimum variance, ERC, HRP. Return-based: mean-variance
with sample means, with Bayes-Stein means, maximum Sharpe with sample means and with Bayes-Stein means,
Black-Litterman with an equilibrium prior and momentum views, half Kelly (sample means, gross <= 1.5),
cost-aware mean-variance (Bayes-Stein, trading cost in the objective), equal weight with a 10% volatility
target.

- H1: no method that uses estimated means beats 1/N on net Sharpe ratio. Test: Jobson-Korkie-Memmel and a
  paired stationary bootstrap (mean block 6 months, 5,000 draws) on each method vs 1/N, Holm-Bonferroni
  across the E1 family at 5%. "Beats" requires a positive difference that survives Holm.
- H2: minimum variance, ERC and HRP have lower volatility and maximum drawdown than 1/N, with Sharpe ratios
  not significantly different from it (same tests).
- H3: Bayes-Stein means beat sample means for mean-variance and maximum Sharpe (paired tests, 5%).

**E2 Estimation error (Monte Carlo).** True parameters = full-sample mean and covariance of multi_asset
excess returns. Draw T = 60, 120, 240 months of normal returns (300 draws each); build portfolios from (a)
estimated means and covariance, (b) true means + estimated covariance, (c) estimated means + true
covariance, (d) the truth. Score each by its TRUE Sharpe ratio. H5: errors in means cost more Sharpe than
errors in the covariance for maximum Sharpe and mean-variance (Chopra and Ziemba 1993).

**E3 Sensitivity.** (i) Covariance estimators (sample, Ledoit-Wolf identity, Ledoit-Wolf constant
correlation, EWMA half-life 36 months, six-factor model, 3-component PCA) for minimum variance on
multi_asset and industries49, windows 60 and 120: out-of-sample volatility and turnover. H6: shrinkage and
factor estimators give lower out-of-sample minimum-variance volatility than the sample covariance on
industries49 (paired test on squared returns). (ii) Expected-return inputs for mean-variance (sample,
Bayes-Stein, James-Stein, equilibrium, zero = minimum variance, signal-based): descriptive.
(iii) Window length 60/120/240: descriptive.

**E4 Signal combination (industries49, also industries12).** Reference signals: momentum 12-1, value
(book-to-market), low beta, residual reversal. Combination methods: equal, IC-weighted (EWMA half-life 36),
empirical-Bayes shrinkage, mean-variance on ICs, Bayesian model averaging, regime-dependent (regime = trailing
12-month market volatility above its expanding median), ridge ensemble; trailing windows 120 months, at
least 60 months of history. Evaluation 1975-01 to 2026-09 (12 industries from 1965-01).
H4: no learned method beats equal weights in out-of-sample rank IC. Test: Newey-West t on the monthly IC
difference, Holm across the six learned methods. Portfolio check (descriptive): dollar-neutral book
proportional to the combined score, gross 1, net of 10 bp costs.

**E5 Synthetic signal lab.** Known ICs; vary signal strength, time variation of ICs, regime sign flips and
junk signals; report each method's out-of-sample IC relative to the oracle. Descriptive.

**E6 Strategy sleeves.** E1's risk-based methods, mean-variance (Bayes-Stein) and 1/N on the nine sleeves.
H7: ERC beats 1/N on net Sharpe (expected yes: sleeve volatilities differ by a factor of about three).

**E7 Risk overlays.** 1/N and ERC with a 10% volatility target and with a drawdown governor (limit 20%,
floor 25%) on multi_asset and sleeves. Descriptive: Sharpe, volatility, maximum drawdown, turnover.

**E8 Kelly.** Full, half and quarter Kelly with sample means (gross <= 2) vs 1/N: realised growth and
drawdown. Descriptive, plus a simulation with known parameters.

**E9 Capacity.** Square-root impact (eta 0.5, ADV = 0.4% of industry market cap a day, Treasuries $50bn)
at $0.1bn, $1bn and $10bn for 1/N, ERC and cost-aware mean-variance. Descriptive. One run goes through
the router adapter to check the integration path end to end.

**E10 Risk model checks.** Rolling 500-day historical, normal and Cornish-Fisher VaR (99%) on daily
multi-asset 1/N returns, Kupiec and Christoffersen tests; current-portfolio risk report and stress tests.

## What would change the conclusions

Any optimizer with estimated means beating 1/N after Holm (H1 false); a learned combination beating equal
weights after Holm (H4 false). Both would be reported as found.
