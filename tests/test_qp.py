import numpy as np
import pytest

from conftest import random_cov
from qpm.optimize.qp import INF, solve_qp


def kkt_errors(P, q, A, l, u, sol):
    x, y = sol.x, sol.y
    z = A @ x
    stat = np.abs(P @ x + q + A.T @ y).max()
    feas = max(0.0, np.max(l - z), np.max(z - u))
    # complementary slackness: y < 0 only at lower bound, y > 0 only at upper bound
    lo_gap = np.where(y < -1e-9, np.abs(z - l), 0.0)
    hi_gap = np.where(y > 1e-9, np.abs(u - z), 0.0)
    comp = max(lo_gap.max(initial=0), hi_gap.max(initial=0))
    return stat, feas, comp


def portfolio_problem(rng, n, long_short=False):
    S = random_cov(rng, n)
    mu = rng.normal(0.06, 0.05, n)
    lam = rng.uniform(1, 10)
    lo = -0.3 * np.ones(n) if long_short else np.zeros(n)
    hi = rng.uniform(0.15, 0.6, n)
    A = np.vstack([np.eye(n), np.ones(n), (rng.random(n) < 0.5).astype(float)])
    l = np.concatenate([lo, [1.0, -INF]])
    u = np.concatenate([hi, [1.0, 0.7]])
    return lam * S, -mu, A, l, u


@pytest.mark.parametrize("seed", range(25))
def test_kkt_conditions_hold(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 30))
    P, q, A, l, u = portfolio_problem(rng, n, long_short=seed % 2 == 1)
    if l[:n].sum() > 1 or u[:n].sum() < 1:
        pytest.skip("infeasible draw")
    sol = solve_qp(P, q, A, l, u)
    assert sol.status == "solved"
    stat, feas, comp = kkt_errors(P, q, A, l, u, sol)
    assert stat < 1e-8 and feas < 1e-9 and comp < 1e-8


def test_matches_cvxpy_with_tight_tolerances():
    cp = pytest.importorskip("cvxpy")
    rng = np.random.default_rng(7)
    worst, compared = 0.0, 0
    for _ in range(30):
        n = int(rng.integers(3, 25))
        P, q, A, l, u = portfolio_problem(rng, n)
        if u[:n].sum() < 1:
            continue
        sol = solve_qp(P, q, A, l, u)
        x = cp.Variable(n)
        prob = cp.Problem(cp.Minimize(0.5 * cp.quad_form(x, cp.psd_wrap(P)) + q @ x),
                          [A @ x >= np.maximum(l, -1e6), A @ x <= np.minimum(u, 1e6)])
        try:
            prob.solve(solver=cp.CLARABEL, tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)
        except Exception:
            pytest.skip("Clarabel not available")
        if prob.status != "optimal":          # infeasible draw (group cap vs position caps): both must agree
            assert sol.status == "primal_infeasible"
            continue
        compared += 1
        worst = max(worst, np.abs(sol.x - x.value).max())
    assert compared >= 15 and worst < 1e-6


def test_detects_primal_infeasibility():
    n = 5
    A = np.vstack([np.eye(n), np.ones(n)])
    l = np.concatenate([np.zeros(n), [1.0]])
    u = np.concatenate([0.1 * np.ones(n), [1.0]])
    sol = solve_qp(np.eye(n), np.zeros(n), A, l, u)
    assert sol.status == "primal_infeasible"


def test_contradictory_bounds_are_infeasible():
    sol = solve_qp(np.eye(2), np.zeros(2), np.eye(2), np.array([1.0, 0.0]), np.array([0.0, 1.0]))
    assert sol.status == "primal_infeasible"


def test_unconstrained_solve():
    rng = np.random.default_rng(1)
    P = random_cov(rng, 6)
    q = rng.normal(size=6)
    sol = solve_qp(P, q)
    assert np.allclose(sol.x, np.linalg.solve(P, -q))


def test_equality_only_matches_closed_form():
    rng = np.random.default_rng(2)
    n = 10
    S = random_cov(rng, n)
    sol = solve_qp(S, np.zeros(n), np.ones((1, n)), np.array([1.0]), np.array([1.0]))
    w = np.linalg.solve(S, np.ones(n))
    w /= w.sum()
    assert np.allclose(sol.x, w, atol=1e-9)


def test_warm_start_gives_same_answer():
    rng = np.random.default_rng(3)
    P, q, A, l, u = portfolio_problem(rng, 12)
    a = solve_qp(P, q, A, l, u)
    b = solve_qp(P, q, A, l, u, x0=a.x, y0=a.y)
    assert np.allclose(a.x, b.x, atol=1e-9)
    assert b.iterations <= a.iterations
