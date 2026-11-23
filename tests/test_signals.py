import numpy as np
import pandas as pd
import pytest

from qpm.signals import (Signal, SignalStore, SignalValidationError, combine, cross_sectional_ic, forward_returns,
                         known_ics, reference, store_to_scores, synthetic, zscore)
from qpm.signals.adapters import statarb_realized_returns, statarb_signals


def _store():
    t0 = pd.Timestamp("2024-01-31")
    recs = [Signal(t0, "A", 0.5, "sys.one", "equity_factor"),
            Signal(t0, "B", -0.5, "sys.one", "equity_factor"),
            Signal(t0 + pd.Timedelta(days=29), "A", 1.5, "sys.one", "equity_factor"),
            Signal(t0, "A", 0.02, "sys.two", "credit", kind="expected_return", horizon_days=5, volatility=0.2)]
    return SignalStore.from_records(recs)


def test_as_of_never_returns_future_and_expires():
    s = _store()
    snap = s.as_of("2024-02-15")
    assert (snap["timestamp"] <= pd.Timestamp("2024-02-15")).all()
    one = snap[snap.source == "sys.one"].set_index("instrument")["value"]
    assert one["A"] == 0.5                      # the 2024-02-29 update is not known yet
    assert "sys.two" not in set(snap.source)    # 5-trading-day horizon expired
    later = s.as_of("2024-03-01")
    assert later[(later.source == "sys.one") & (later.instrument == "A")]["value"].iloc[0] == 1.5


def test_validation_catches_bad_records():
    t = pd.Timestamp("2024-01-31")
    bad = SignalStore.from_records([Signal(t, "A", 1.0, "x", "astrology"), Signal(t, "A", 1.0, "x", "astrology")])
    with pytest.raises(SignalValidationError):
        bad.validate()
    issues = SignalStore.from_records([Signal(t, "A", 1.0, "x", "credit", confidence=1.5)]).validate(strict=False)
    assert any("confidence" in i for i in issues)


def test_round_trip_csv_json(tmp_path):
    s = _store()
    s.write(tmp_path / "s.csv")
    s.write(tmp_path / "s.json")
    a = SignalStore.read(tmp_path / "s.csv")
    b = SignalStore.read(tmp_path / "s.json")
    assert len(a) == len(b) == len(s)
    assert np.allclose(a.frame["value"], s.frame["value"]) and np.allclose(b.frame["value"], s.frame["value"])


def test_store_to_scores_converts_expected_returns():
    s = _store()
    snap = s.as_of("2024-01-31")
    sc = store_to_scores(snap, ["A", "B"])
    assert set(sc.columns) == {"sys.one", "sys.two"}
    assert sc.loc["A", "sys.one"] > sc.loc["B", "sys.one"]


def test_ic_and_known_cutoff(rng):
    dates = pd.date_range("2000-01-31", periods=30, freq="ME")
    scores = pd.DataFrame(rng.normal(size=(30, 20)), dates)
    rets = scores.shift(1) * 0.01 + rng.normal(0, 0.001, (30, 20))  # next return follows the score
    fwd = forward_returns(rets)
    ic = cross_sectional_ic(scores, fwd)
    assert ic.mean() > 0.9
    known = known_ics(ic, dates[10], 1, dates)
    assert known.index.max() == dates[9]


def test_zscore_clips_and_standardises():
    z = zscore(pd.Series([0, 0, 0, 0, 100.0, 1, 2]))
    assert abs(z.mean()) < 1e-12 and abs(z.std(ddof=0) - 1) < 1e-12


@pytest.mark.parametrize("method", ["equal", "ic_weighted", "shrunk", "mean_var", "bma", "regime", "ridge"])
def test_combination_is_point_in_time(method):
    lab = synthetic.simulate(n_assets=30, n_periods=90, ics=(0.15, 0.0, 0.05), seed=4, regime_flip=(2,))
    zs, fwd = lab["zs"], lab["fwd"]
    kw = {"min_history": 12, "window": 40}
    if method == "regime":
        kw["regimes"] = lab["regime"]
    full = combine(zs, fwd, method, **kw)
    cut = fwd.index[60]
    trunc_fwd = fwd.copy()
    # row t of fwd is realised at t+1: at the cut (row 60) rows up to 59 are known, row 60 onward is not
    trunc_fwd.loc[fwd.index >= cut] = np.nan
    part = combine({k: v.loc[:cut] for k, v in zs.items()}, trunc_fwd.loc[:cut], method, **kw)
    a = full.scores.loc[:cut].values
    b = part.scores.loc[:cut].values
    assert np.allclose(np.nan_to_num(a), np.nan_to_num(b), atol=1e-12)


def test_learned_combinations_find_the_strong_signal():
    lab = synthetic.simulate(n_assets=60, n_periods=300, ics=(0.12, 0.0, 0.0, 0.0), signal_corr=0.0, seed=1)
    out = {}
    for m in ("equal", "ic_weighted", "bma", "ridge"):
        c = combine(lab["zs"], lab["fwd"], m, min_history=24, window=60)
        out[m] = cross_sectional_ic(c.scores.iloc[100:], lab["fwd"].iloc[100:]).mean()
    assert out["ic_weighted"] > out["equal"] and out["bma"] > out["equal"] and out["ridge"] > out["equal"]


def test_reference_signals_are_point_in_time(small_market):
    returns, rf = small_market
    ex = returns.sub(rf, axis=0)
    mkt = ex.mean(axis=1)
    cut = returns.index[150]
    for fn in (lambda r, m: reference.momentum_12_1(r), lambda r, m: reference.residual_reversal(r, m),
               lambda r, m: reference.low_beta(r, m), lambda r, m: reference.tsmom(r)):
        full = fn(ex, mkt).loc[:cut]
        part = fn(ex.loc[:cut], mkt.loc[:cut])
        assert np.allclose(full.fillna(0).values, part.fillna(0).values, atol=1e-12)
    # momentum at t ignores month t itself
    mom = reference.momentum_12_1(returns)
    bumped = returns.copy()
    bumped.iloc[150, 0] += 0.5
    assert np.isclose(reference.momentum_12_1(bumped).iloc[150, 0], mom.iloc[150, 0])


def test_statarb_adapter_never_reads_label(tmp_path):
    pytest.importorskip("pyarrow")
    idx = pd.MultiIndex.from_product([pd.to_datetime(["2020-01-31", "2020-02-29"]), ["AAA", "BBB", "CCC"]],
                                     names=["formation", "sid"])
    frame = pd.DataFrame({"MOM": [1.0, 0.0, -1.0, 0.5, 0.2, -0.7], "SUE": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
                          "madj_next": [9, 9, 9, 9, 9, 9]}, index=idx)
    path = tmp_path / "signals.parquet"
    frame.to_parquet(path)
    store = statarb_signals(path)
    assert set(store.sources) == {"statarb.mom", "statarb.sue"}
    assert not (store.frame["value"] == 9).any()
    assert set(store.frame.family) == {"equity_factor", "earnings_event"}
    assert statarb_realized_returns(path).shape == (2, 3)
