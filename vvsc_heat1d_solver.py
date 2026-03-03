#!/usr/bin/env python3
"""
===============================================================================
vvsc_heat1d_solver.py — 1D Transient Heat Conduction Solver
===============================================================================
Course:  AOE/CS/ME 6444 — Verification and Validation in Scientific Computing
Author:  Dawood Wasif
===============================================================================

Numerical Method:
    Spatial:  2nd-order central finite differences   O(Δz²)
    Temporal: Crank-Nicolson (implicit trapezoidal)  O(Δt²)
    Linear system: Thomas algorithm (tridiagonal)

Governing PDE:
    ρ c_p ∂T/∂t = k ∂²T/∂z² + s(z,t)

This module provides a clean, reusable solver that is imported by the MMS
verification driver.  The solver accepts callable functions for initial
conditions, boundary conditions, and source terms so it can be used with
any manufactured solution.
===============================================================================
"""

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Thomas Algorithm (tridiagonal solver)
# ─────────────────────────────────────────────────────────────────────────────
def thomas_solve(a: float, b: float, c: float, d: np.ndarray) -> np.ndarray:
    """
    Thomas algorithm for tridiagonal systems with constant coefficients.

    Solves  A x = d  where A is tridiagonal with:
        sub-diagonal  = a  (scalar, constant)
        main diagonal = b  (scalar, constant)
        super-diagonal= c  (scalar, constant)

    Parameters
    ----------
    a : float — sub-diagonal coefficient
    b : float — main diagonal coefficient
    c : float — super-diagonal coefficient
    d : ndarray, shape (n,) — right-hand side vector

    Returns
    -------
    x : ndarray, shape (n,) — solution vector
    """
    n = len(d)
    cp = np.zeros(n, dtype=float)
    dp = np.zeros(n, dtype=float)

    # Forward sweep
    cp[0] = c / b
    dp[0] = d[0] / b
    for i in range(1, n):
        denom = b - a * cp[i - 1]
        cp[i] = (c / denom) if i < n - 1 else 0.0
        dp[i] = (d[i] - a * dp[i - 1]) / denom

    # Back substitution
    x = np.zeros(n, dtype=float)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]

    return x


# ─────────────────────────────────────────────────────────────────────────────
# Crank-Nicolson Solver
# ─────────────────────────────────────────────────────────────────────────────
def solve_heat_1d_CN(
    Nz: int,
    Nt: int,
    L: float,
    t_final: float,
    rho: float,
    cp: float,
    k: float,
    T_init_fn,
    T_left_fn,
    T_right_fn,
    source_fn=None,
    store_history: bool = False,
):
    """
    Solve 1D heat equation with Dirichlet BCs via Crank-Nicolson.

    PDE:  ρ c_p ∂T/∂t = k ∂²T/∂z² + s(z, t)

    Space: 2nd-order central FD on uniform grid (Nz cells, Nz+1 nodes)
    Time:  Crank-Nicolson with trapezoidal source evaluation (Nt steps)
    Solve: Thomas algorithm for the tridiagonal system

    Parameters
    ----------
    Nz, Nt      : spatial cells and time steps
    L, t_final  : domain length [m] and final time [s]
    rho, cp, k  : density, specific heat, thermal conductivity
    T_init_fn   : callable(z_array) → initial T array
    T_left_fn   : callable(t_scalar) → left BC value
    T_right_fn  : callable(t_scalar) → right BC value
    source_fn   : callable(z_array, t_scalar) → source array (or None)
    store_history : if True, store the full space-time solution

    Returns
    -------
    dict with keys:
        z, t, dz, dt, Fo, T_final
        T_history (Nt+1 × Nz+1 array, only if store_history=True)
    """
    alpha = k / (rho * cp)
    dz = L / Nz
    dt = t_final / Nt
    Fo = alpha * dt / dz**2

    z = np.linspace(0.0, L, Nz + 1)
    t = np.linspace(0.0, t_final, Nt + 1)

    # Solution vector — initialise from T_init_fn
    T = np.asarray(T_init_fn(z), dtype=float).copy()

    T_history = None
    if store_history:
        T_history = np.zeros((Nt + 1, Nz + 1), dtype=float)
        T_history[0, :] = T.copy()

    # Tridiagonal coefficients (constant for uniform grid & properties)
    # Implicit side (LHS)
    a_imp = -Fo / 2.0      # sub-diagonal
    b_imp =  1.0 + Fo       # main diagonal
    c_imp = -Fo / 2.0      # super-diagonal
    # Explicit side (RHS)
    a_exp =  Fo / 2.0
    b_exp =  1.0 - Fo
    c_exp =  Fo / 2.0

    N_int = Nz - 1   # number of interior unknowns

    # Vectorised source evaluation wrapper
    def eval_source(z_vec, tt):
        if source_fn is None:
            return np.zeros_like(z_vec, dtype=float)
        return np.asarray(source_fn(z_vec, tt), dtype=float)

    # ── Time-stepping loop ──────────────────────────────────────────────
    for n in range(Nt):
        tn   = t[n]
        tnp1 = t[n + 1]

        # Build RHS using vectorised slicing (no Python for-loop)
        T_im1 = T[0:Nz - 1]     # T_{i-1}^n   for i = 1..Nz-1
        T_i   = T[1:Nz]         # T_i^n
        T_ip1 = T[2:Nz + 1]     # T_{i+1}^n

        rhs = a_exp * T_im1 + b_exp * T_i + c_exp * T_ip1

        # Source term — Crank-Nicolson trapezoidal average
        z_int = z[1:Nz]
        s_n   = eval_source(z_int, tn)
        s_np1 = eval_source(z_int, tnp1)
        rhs += (dt / (2.0 * rho * cp)) * (s_n + s_np1)

        # Dirichlet BCs at t^{n+1}
        T_left  = float(T_left_fn(tnp1))
        T_right = float(T_right_fn(tnp1))

        # Move known BC values to RHS
        rhs[0]  -= a_imp * T_left
        rhs[-1] -= c_imp * T_right

        # Solve interior tridiagonal system
        T_int = thomas_solve(a_imp, b_imp, c_imp, rhs)

        # Assemble full solution vector
        T[0]    = T_left
        T[1:Nz] = T_int
        T[Nz]   = T_right

        if store_history:
            T_history[n + 1, :] = T.copy()

    result = {
        "z": z,
        "t": t,
        "dz": dz,
        "dt": dt,
        "Fo": Fo,
        "T_final": T.copy(),
    }
    if store_history:
        result["T_history"] = T_history

    return result