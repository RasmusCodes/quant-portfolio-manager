"""qpm: a quantitative portfolio manager.

Layers (each usable on its own):

* ``qpm.estimation``  expected-return and covariance estimators (shrinkage, factor models, EWMA)
* ``qpm.optimize``    a dense ADMM quadratic-program solver and the construction methods built on it
                      (mean-variance, minimum variance, risk parity, ERC, HRP, Black-Litterman, Kelly,
                      volatility targeting, factor-neutral and fully constrained optimization), plus
                      weight explanations and efficient frontiers
* ``qpm.signals``     signal schema, point-in-time store, normalisation and combination methods
* ``qpm.risk``        VaR / expected shortfall, risk decomposition, stress scenarios, liquidity,
                      concentration and drawdown analytics
* ``qpm.execution``   execution interface (parent orders in, fill reports out), a built-in cost
                      simulator and an adapter for an external smart order router
* ``qpm.backtest``    walk-forward portfolio backtester with costs, turnover and cash management
"""
__version__ = "1.0.0"
