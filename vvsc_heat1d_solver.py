#!/usr/bin/env python3
"""
===============================================================================
vvsc_heat1d_solver.py — 1D Transient Heat Conduction Solver
===============================================================================
Course:  AOE/CS/ME 6444 — Verification and Validation in Scientific Computing
Author:  Dawood Wasif
===============================================================================
Provides two solvers:
  1. solve_heat_1d_CN          -- Dirichlet BCs (verified in HW3 via MMS)
  2. solve_heat_1d_CN_neumann  -- Neumann at z=0 + Dirichlet at z=L
                                  (Newton iteration for nonlinear surface BC)
===============================================================================
"""
import numpy as np

def thomas_solve(a, b, c, d):
    """Thomas algorithm, constant coefficients."""
    n = len(d)
    cp = np.zeros(n, dtype=d.dtype)
    dp = np.zeros(n, dtype=d.dtype)
    cp[0] = c / b; dp[0] = d[0] / b
    for i in range(1, n):
        denom = b - a * cp[i - 1]
        cp[i] = (c / denom) if i < n - 1 else 0.0
        dp[i] = (d[i] - a * dp[i - 1]) / denom
    x = np.zeros(n, dtype=d.dtype)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

def thomas_solve_var(a_sub, b_diag, c_sup, d):
    """Thomas algorithm, variable coefficients."""
    n = len(d)
    cp = np.zeros(n, dtype=d.dtype)
    dp = np.zeros(n, dtype=d.dtype)
    cp[0] = c_sup[0] / b_diag[0]; dp[0] = d[0] / b_diag[0]
    for i in range(1, n):
        denom = b_diag[i] - a_sub[i] * cp[i - 1]
        cp[i] = (c_sup[i] / denom) if i < n - 1 else 0.0
        dp[i] = (d[i] - a_sub[i] * dp[i - 1]) / denom
    x = np.zeros(n, dtype=d.dtype)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x

# ── Dirichlet-only solver (HW3, verified via MMS) ───────────────────────
def solve_heat_1d_CN(
    Nz, Nt, L, t_final, rho, cp, k,
    T_init_fn, T_left_fn, T_right_fn,
    source_fn=None, store_history=False,
):
    alpha = k / (rho * cp)
    dz = L / Nz; dt = t_final / Nt; Fo = alpha * dt / dz**2
    z = np.linspace(0.0, L, Nz + 1); t = np.linspace(0.0, t_final, Nt + 1)
    T = np.asarray(T_init_fn(z), dtype=float).copy()
    T_history = None
    if store_history:
        T_history = np.zeros((Nt + 1, Nz + 1)); T_history[0] = T.copy()
    a_imp = -Fo/2; b_imp = 1+Fo; c_imp = -Fo/2
    a_exp = Fo/2;  b_exp = 1-Fo; c_exp = Fo/2
    def esrc(zv, tt):
        if source_fn is None: return np.zeros_like(zv)
        return np.asarray(source_fn(zv, tt), dtype=float)
    for ns in range(Nt):
        tn=t[ns]; tnp1=t[ns+1]
        rhs = a_exp*T[0:Nz-1] + b_exp*T[1:Nz] + c_exp*T[2:Nz+1]
        zi=z[1:Nz]; rhs += (dt/(2*rho*cp))*(esrc(zi,tn)+esrc(zi,tnp1))
        Tl=float(T_left_fn(tnp1)); Tr=float(T_right_fn(tnp1))
        rhs[0] -= a_imp*Tl; rhs[-1] -= c_imp*Tr
        T[1:Nz] = thomas_solve(a_imp, b_imp, c_imp, rhs)
        T[0]=Tl; T[Nz]=Tr
        if store_history: T_history[ns+1]=T.copy()
    result = {"z":z,"t":t,"dz":dz,"dt":dt,"Fo":Fo,"T_final":T.copy()}
    if store_history: result["T_history"]=T_history
    return result

# ── Neumann+Dirichlet solver (HW4, Newton iteration) ────────────────────
def solve_heat_1d_CN_neumann(
    Nz, Nt, L, t_final, rho, cp, k,
    T_init_fn, T_right_fn,
    q_net_fn, dq_net_dTs_fn,
    newton_tol=1e-12, newton_maxiter=50,
    store_history=False, dtype=np.float64,
):
    FT = dtype
    alpha = k / (rho * cp)
    dz = FT(L / Nz); dt = FT(t_final / Nt)
    Fo = FT(alpha * dt / dz**2)
    Fo_h = FT(Fo / 2); opFo = FT(1 + Fo); omFo = FT(1 - Fo)
    cq = FT(Fo * float(dz) / k)

    z = np.linspace(FT(0), FT(L), Nz+1).astype(FT)
    t_arr = np.linspace(FT(0), FT(t_final), Nt+1).astype(FT)
    T = np.asarray(T_init_fn(z), dtype=FT).copy()

    T_history = None
    if store_history:
        T_history = np.zeros((Nt+1, Nz+1), dtype=FT); T_history[0]=T.copy()

    Ts_hist = np.zeros(Nt+1, dtype=FT); Ts_hist[0]=T[0]
    newton_total = 0

    # Pre-allocate Jacobian arrays
    a_sub  = np.zeros(Nz, dtype=FT)
    b_diag = np.zeros(Nz, dtype=FT)
    c_sup  = np.zeros(Nz, dtype=FT)
    R      = np.zeros(Nz, dtype=FT)

    # Interior Jacobian (constant, set once)
    a_sub[1:] = -Fo_h
    c_sup[:-1] = -Fo_h
    c_sup[0] = FT(-Fo)  # surface row super-diagonal

    for ns in range(Nt):
        tn = float(t_arr[ns]); tnp1 = float(t_arr[ns+1])
        T_old = T.copy()
        Tr_new = FT(T_right_fn(tnp1))

        # ── Explicit RHS (vectorised) ───────────────────────────────────
        rhs_ex = np.zeros(Nz, dtype=FT)
        q_n = FT(q_net_fn(float(T_old[0]), tn))
        rhs_ex[0] = omFo*T_old[0] + Fo*T_old[1] + cq*q_n
        rhs_ex[1:Nz-1] = Fo_h*T_old[0:Nz-2] + omFo*T_old[1:Nz-1] + Fo_h*T_old[2:Nz]
        rhs_ex[Nz-1] = Fo_h*T_old[Nz-2] + omFo*T_old[Nz-1] + Fo_h*T_old[Nz] + Fo_h*Tr_new

        # ── Newton iteration ────────────────────────────────────────────
        Tg = T_old[:Nz].copy()

        for nit in range(newton_maxiter):
            # Surface residual
            Ts_g = float(Tg[0])
            q_np1 = FT(q_net_fn(Ts_g, tnp1))
            dq    = FT(dq_net_dTs_fn(Ts_g, tnp1))
            R[0] = opFo*Tg[0] - Fo*Tg[1] - rhs_ex[0] - cq*q_np1
            b_diag[0] = opFo - cq*dq

            # Interior residual (vectorised)
            R[1:Nz-1] = (-Fo_h*Tg[0:Nz-2] + opFo*Tg[1:Nz-1]
                         - Fo_h*Tg[2:Nz] - rhs_ex[1:Nz-1])
            b_diag[1:Nz-1] = opFo

            # Last node
            R[Nz-1] = -Fo_h*Tg[Nz-2] + opFo*Tg[Nz-1] - rhs_ex[Nz-1]
            b_diag[Nz-1] = opFo

            res_norm = float(np.max(np.abs(R)))
            if res_norm < float(newton_tol):
                newton_total += (nit + 1)
                break

            delta = thomas_solve_var(a_sub, b_diag, c_sup, -R)
            Tg += delta
        else:
            newton_total += newton_maxiter

        T[:Nz] = Tg; T[Nz] = Tr_new
        Ts_hist[ns+1] = T[0]
        if store_history: T_history[ns+1] = T.copy()

    result = {
        "z": z.astype(np.float64), "t": t_arr.astype(np.float64),
        "dz": float(dz), "dt": float(dt), "Fo": float(Fo),
        "T_final": T.astype(np.float64).copy(),
        "T_surface_history": Ts_hist.astype(np.float64).copy(),
        "newton_iters_total": newton_total,
    }
    if store_history: result["T_history"] = T_history.astype(np.float64)
    return result