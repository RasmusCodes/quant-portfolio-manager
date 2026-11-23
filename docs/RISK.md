# Risk engine

## Value at risk and expected shortfall (`qpm.risk.var`)

Losses are positive numbers. `historical_var/es`, `parametric_var/es` (normal or unit-variance Student t),
`cornish_fisher_var` (skewness and excess kurtosis; unreliable at very high kurtosis), `monte_carlo_var`
(multivariate normal or t). `rolling_var_backtest` re-estimates each day from the previous window only and supports
`historical`, `normal`, `cornish_fisher`, `ewma` (RiskMetrics variance, normal quantile) and `fhs` (filtered
historical simulation: empirical quantile of EWMA-standardised returns times today's volatility). `kupiec_pof`
tests the breach count and `christoffersen_independence` tests clustering.

## Decomposition (`qpm.risk.decomposition`)

- `contributions`: marginal and component contributions to volatility (Euler; components sum to volatility).
- `component_es`: each position's average loss on the portfolio's tail days (sums to historical ES).
- `factor_decomposition`: betas from time-series regressions, portfolio exposures, factor variance contributions and
  the idiosyncratic share.
- `concentration`: HHI, effective number of positions, top-5 share, diversification ratio, effective number of
  uncorrelated bets (entropy of principal-component risk shares, Meucci 2009).

## Stress scenarios (`qpm.risk.stress`, `scenarios/*.json`)

Scenarios are data. Kinds:

| Kind | Fields | What happens |
|---|---|---|
| `historical` | `start`, `end` | The assets' actual returns over the window, compounded, with today's weights held fixed |
| `factor` | `factor_shocks`, optional `asset_shocks`, `group_shocks` | Factor moves mapped to assets through betas; named assets or groups overridden |
| `asset` | `asset_shocks`, `default_shock` | Direct moves |
| `volatility` | `vol_multiplier`, `correlation_target`, `correlation_weight`, optional shocks | VaR and ES under a stressed covariance (volatilities scaled, correlations pulled toward the target, repaired to PSD) |

Shipped: 2008 credit crisis (replay and hypothetical), 2020 liquidity shock, 2022 inflation and rate shock, 1987
crash, dot-com bust, AI infrastructure capex reversal (hypothetical), sudden volatility spike, +150 bp bond
sell-off. Factor names used by the shipped scenarios: `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `Mom` (French) and
`Rates` (10-year Treasury excess return). Add a file to `scenarios/` to add a scenario; `load_scenarios()` validates
the kind and required fields.

## Liquidity (`qpm.risk.liquidity`)

Days to liquidate at a participation rate of ADV, the share of the book sellable within 1/5/20 days, liquidation
cost from the square-root impact law, and liquidity-adjusted VaR (Bangia et al. 1999). For industry portfolios the
study assumes ADV = 0.4% of industry market value a day; Treasuries $50bn a day.

## Drawdowns and correlation (`qpm.risk.drawdown`)

Drawdown series and episodes (depth, months to trough, months to recover), conditional drawdown at risk, the
average pairwise correlation and absorption ratio, and `correlation_breakdown`, which compares correlations in
high-volatility regimes (trailing 12-month market volatility in its top fifth) with the rest and reports the
stock-bond correlation in each. Stress months are deliberately not chosen by their own market return: conditioning
on the common factor's tail shrinks its variance inside the subsample and biases correlations down (Forbes and
Rigobon 2002).
