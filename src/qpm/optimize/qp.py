"""Dense solver for convex quadratic programs

    minimise   1/2 x'Px + q'x
    subject to l <= Ax <= u          (equality rows have l = u; infinite bounds allowed)

The algorithm is OSQP's ADMM (Stellato, Banjac, Goulart, Bemporad and Boyd, 2020, "OSQP: an operator
splitting solver for quadratic programs", Math. Prog. Comp.): Ruiz equilibration, over-relaxed ADMM with
an adaptive step size, and infeasibility certificates.  Portfolio problems are small and dense, so the
linear system is solved with a dense Cholesky factor.

After ADMM converges, an active-set "polish" solves the KKT system restricted to the constraints ADMM
found active, then repairs the active set (drops rows whose multiplier has the wrong sign, adds violated
rows) until the solution is exact.  The browser demo runs a line-by-line port of this file.

Sign convention: at a solution Px + q + A'y = 0, with y_i < 0 when row i sits at its lower bound and
y_i > 0 when it sits at its upper bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INF = 1e20


@dataclass
class QPSolution:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    status: str            # solved | solved_inaccurate | primal_infeasible | dual_infeasible | max_iter
    iterations: int
    objective: float
    primal_residual: float
    dual_residual: float
    polished: bool
    rho: float

    @property
    def ok(self) -> bool:
        return self.status in ("solved", "solved_inaccurate")


def _cholesky_solver(k: np.ndarray):
    lower = np.linalg.cholesky(k)

    def solve(b: np.ndarray) -> np.ndarray:
        tmp = np.linalg.solve(lower, b)
        return np.linalg.solve(lower.T, tmp)

    return solve


def _ruiz(P, q, A, iters: int):
    n, m = P.shape[0], A.shape[0]
    D, E, c = np.ones(n), np.ones(m), 1.0
    Ps, qs, As = P.copy(), q.copy(), A.copy()
    for _ in range(iters):
        col_x = np.abs(Ps).max(axis=0)
        if m:
            col_x = np.maximum(col_x, np.abs(As).max(axis=0))
            row_z = np.abs(As).max(axis=1)
        else:
            row_z = np.ones(0)
        col_x = np.clip(np.where(col_x < 1e-8, 1.0, col_x), 1e-4, 1e4)
        row_z = np.clip(np.where(row_z < 1e-8, 1.0, row_z), 1e-4, 1e4)
        dx, dz = 1.0 / np.sqrt(col_x), 1.0 / np.sqrt(row_z)
        Ps = dx[:, None] * Ps * dx[None, :]
        qs = dx * qs
        if m:
            As = dz[:, None] * As * dx[None, :]
        D *= dx
        E *= dz
        mean_col = np.abs(Ps).max(axis=0).mean() if n else 1.0
        gamma = max(mean_col, np.abs(qs).max() if n else 0.0)
        gamma = 1.0 / np.clip(gamma if gamma > 1e-8 else 1.0, 1e-4, 1e4)
        Ps *= gamma
        qs *= gamma
        c *= gamma
    return Ps, qs, As, D, E, c


def _rho_vector(l, u, rho):
    vec = np.full(l.shape, rho)
    eq = np.abs(u - l) < 1e-10
    loose = (l <= -INF / 2) & (u >= INF / 2)
    vec[eq] = 1e3 * rho
    vec[loose] = 1e-6
    return vec


def _objective(P, q, x):
    return float(0.5 * x @ P @ x + q @ x)


def _polish(P, q, A, l, u, x, y, tol=1e-9, rounds=12):
    """Active-set refinement of an ADMM solution. Returns (x, y, z) or None if it fails."""
    n, m = P.shape[0], A.shape[0]
    z = A @ x
    eq = np.abs(u - l) < 1e-10
    lower = (eq & (y <= 0)) | (~eq & (z - l < -y))
    upper = (eq & (y > 0)) | (~eq & (u - z < y))
    lower &= l > -INF / 2
    upper &= u < INF / 2
    upper &= ~lower
    scale = max(1.0, np.abs(l[l > -INF / 2]).max() if np.any(l > -INF / 2) else 1.0,
                np.abs(u[u < INF / 2]).max() if np.any(u < INF / 2) else 1.0)
    for _ in range(rounds):
        act = np.flatnonzero(lower | upper)
        rhs_b = np.where(lower[act], l[act], u[act])
        k = act.size
        kkt = np.zeros((n + k, n + k))
        kkt[:n, :n] = P
        kkt[:n, n:] = A[act].T
        kkt[n:, :n] = A[act]
        rhs = np.concatenate([-q, rhs_b])
        try:
            sol = np.linalg.solve(kkt, rhs)
            if not np.all(np.isfinite(sol)):
                raise np.linalg.LinAlgError
        except np.linalg.LinAlgError:
            reg = kkt.copy()
            reg[:n, :n] += 1e-10 * np.eye(n)
            reg[n:, n:] -= 1e-10 * np.eye(k)
            sol = np.linalg.solve(reg, rhs)
            for _r in range(5):  # iterative refinement toward the unregularised system
                sol = sol + np.linalg.solve(reg, rhs - kkt @ sol)
        xp = sol[:n]
        yp = np.zeros(m)
        yp[act] = sol[n:]
        zp = A @ xp
        viol_lo = (l - zp > tol * scale) & ~lower & ~upper & (l > -INF / 2)
        viol_hi = (zp - u > tol * scale) & ~lower & ~upper & (u < INF / 2)
        wrong_lo = lower & ~eq & (yp > tol)
        wrong_hi = upper & ~eq & (yp < -tol)
        if not (viol_lo.any() or viol_hi.any() or wrong_lo.any() or wrong_hi.any()):
            return xp, yp, zp
        # repair: drop the worst wrong-signed row, add every violated row
        if wrong_lo.any() or wrong_hi.any():
            bad = np.where(wrong_lo, yp, 0.0) - np.where(wrong_hi, yp, 0.0)
            worst = int(np.argmax(bad))
            lower[worst] = upper[worst] = False
        lower |= viol_lo
        upper |= viol_hi
    return None


def solve_qp(P, q, A=None, l=None, u=None, *, x0=None, y0=None, eps_abs: float = 1e-8, eps_rel: float = 1e-8,
             max_iter: int = 20000, rho: float = 0.1, sigma: float = 1e-6, alpha: float = 1.6,
             scaling: int = 10, adaptive_rho: bool = True, polish: bool = True,
             check_every: int = 5) -> QPSolution:
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float)
    n = q.size
    if A is None or np.size(A) == 0:
        A = np.zeros((0, n))
        l = np.zeros(0)
        u = np.zeros(0)
    A = np.asarray(A, dtype=float).reshape(-1, n)
    l = np.clip(np.asarray(l, dtype=float), -INF, INF)
    u = np.clip(np.asarray(u, dtype=float), -INF, INF)
    m = A.shape[0]
    P = 0.5 * (P + P.T)
    if np.any(l > u + 1e-12):  # a row whose lower bound exceeds its upper bound
        return QPSolution(np.full(n, np.nan), np.zeros(m), np.full(m, np.nan), "primal_infeasible", 0,
                          np.nan, np.inf, np.inf, False, rho)

    if m == 0:
        try:
            x = np.linalg.solve(P, -q)
            return QPSolution(x, np.zeros(0), np.zeros(0), "solved", 0, _objective(P, q, x), 0.0,
                              float(np.abs(P @ x + q).max(initial=0.0)), False, rho)
        except np.linalg.LinAlgError:
            return QPSolution(np.full(n, np.nan), np.zeros(0), np.zeros(0), "dual_infeasible", 0, np.nan,
                              np.inf, np.inf, False, rho)

    Ps, qs, As, D, E, c = _ruiz(P, q, A, scaling)
    ls = np.where(l > -INF / 2, E * l, -INF)
    us = np.where(u < INF / 2, E * u, INF)

    rho_vec = _rho_vector(ls, us, rho)
    solve = _cholesky_solver(Ps + sigma * np.eye(n) + As.T @ (rho_vec[:, None] * As))

    x = np.zeros(n) if x0 is None else np.asarray(x0, dtype=float) / D
    z = np.clip(As @ x, ls, us)
    y = np.zeros(m) if y0 is None else c * np.asarray(y0, dtype=float) / E

    status = "max_iter"
    prim_res = dual_res = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        rhs = sigma * x - qs + As.T @ (rho_vec * z - y)
        xt = solve(rhs)
        zt = As @ xt
        x_new = alpha * xt + (1 - alpha) * x
        z_relax = alpha * zt + (1 - alpha) * z
        z_new = np.clip(z_relax + y / rho_vec, ls, us)
        y_new = y + rho_vec * (z_relax - z_new)
        dx, dy = x_new - x, y_new - y
        x, z, y = x_new, z_new, y_new

        if it % check_every and it != max_iter:
            continue
        # residuals in the original (unscaled) problem
        ax_s = As @ x
        px_s = Ps @ x
        aty_s = As.T @ y
        prim_vec = (ax_s - z) / E
        dual_vec = (px_s + qs + aty_s) / (c * D)
        prim_res = float(np.abs(prim_vec).max())
        dual_res = float(np.abs(dual_vec).max())
        eps_p = eps_abs + eps_rel * max(np.abs(ax_s / E).max(), np.abs(z / E).max())
        eps_d = eps_abs + eps_rel * max(np.abs(px_s / (c * D)).max(), np.abs(aty_s / (c * D)).max(),
                                        np.abs(qs / (c * D)).max())
        if prim_res <= eps_p and dual_res <= eps_d:
            status = "solved"
            break

        # infeasibility certificates (unscaled)
        if it > 50:
            dyu = E * dy / c
            ndy = np.abs(dyu).max()
            if ndy > 1e-12:
                atdy = np.abs(A.T @ dyu).max()
                finite_hi, finite_lo = u < INF / 2, l > -INF / 2
                bad_dir = np.any((~finite_hi) & (dyu > 1e-9 * ndy)) or np.any((~finite_lo) & (dyu < -1e-9 * ndy))
                support = np.sum(np.where(finite_hi, u, 0.0) * np.maximum(dyu, 0.0)) + \
                    np.sum(np.where(finite_lo, l, 0.0) * np.minimum(dyu, 0.0))
                if not bad_dir and atdy <= 1e-6 * ndy and support < -1e-6 * ndy:
                    status = "primal_infeasible"
                    break
            dxu = D * dx
            ndx = np.abs(dxu).max()
            if ndx > 1e-12:
                pdx = np.abs(P @ dxu).max()
                qdx = q @ dxu
                adx = A @ dxu
                fin_hi, fin_lo = u < INF / 2, l > -INF / 2
                okA = np.all(np.where(fin_hi & fin_lo, np.abs(adx) <= 1e-6 * ndx, True)) and \
                    np.all(np.where(fin_lo & ~fin_hi, adx >= -1e-6 * ndx, True)) and \
                    np.all(np.where(fin_hi & ~fin_lo, adx <= 1e-6 * ndx, True))
                if pdx <= 1e-6 * ndx and qdx < -1e-6 * ndx and okA:
                    status = "dual_infeasible"
                    break

        if adaptive_rho and it % (5 * check_every) == 0:
            prim_norm = max(np.abs(ax_s).max(), np.abs(z).max(), 1e-12)
            dual_norm = max(np.abs(px_s).max(), np.abs(aty_s).max(), np.abs(qs).max(), 1e-12)
            ps = np.abs(ax_s - z).max() / prim_norm
            ds = np.abs(px_s + qs + aty_s).max() / dual_norm
            if ds > 1e-14:
                new_rho = float(np.clip(rho * np.sqrt(ps / ds), 1e-6, 1e6))
                if new_rho > 5 * rho or new_rho < rho / 5:
                    rho = new_rho
                    rho_vec = _rho_vector(ls, us, rho)
                    solve = _cholesky_solver(Ps + sigma * np.eye(n) + As.T @ (rho_vec[:, None] * As))

    xu = D * x
    yu = E * y / c
    zu = z / E
    if status in ("primal_infeasible", "dual_infeasible"):
        return QPSolution(xu, yu, zu, status, it, np.nan, prim_res, dual_res, False, rho)
    polished = False
    if polish:
        out = _polish(P, q, A, l, u, xu, yu)
        if out is not None:
            xp, yp, zp = out
            pr = float(np.maximum(np.maximum(l - zp, zp - u), 0.0).max(initial=0.0))
            dr = float(np.abs(P @ xp + q + A.T @ yp).max())
            tol = 1e-7 * max(1.0, np.abs(q).max(), np.abs(P).max())
            if pr <= max(tol, prim_res) and dr <= max(tol, dual_res):
                xu, yu, zu, prim_res, dual_res, polished = xp, yp, zp, pr, dr, True
                status = "solved"
    if status == "max_iter" and prim_res < 1e-4 and dual_res < 1e-4:
        status = "solved_inaccurate"
    return QPSolution(xu, yu, zu, status, it, _objective(P, q, xu), prim_res, dual_res, polished, rho)
