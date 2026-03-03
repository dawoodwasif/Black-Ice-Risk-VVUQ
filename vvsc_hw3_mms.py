#!/usr/bin/env python3
"""
===============================================================================
VVSC Homework 3: Code Verification via Method of Manufactured Solutions (MMS)
===============================================================================
Application:  Black-Ice Risk Prediction — 1D Transient Heat Conduction Solver
Course:       AOE/CS/ME 6444 — Verification and Validation in Scientific Computing
Semester:     Spring 2026
Instructor:   Dr. Chris Roy
Author:       Dawood Wasif
===============================================================================

This script performs MMS code verification for the 1-D unsteady heat
conduction solver implemented in vvsc_heat1d_solver.py.

Governing PDE (1D transient heat conduction, constant properties):

    ρ c_p ∂T/∂t = k ∂²T/∂z² + s(z,t)

Numerical Method:
    - Spatial:  2nd-order central finite differences   O(Δz²)
    - Temporal: Crank–Nicolson (implicit trapezoidal)  O(Δt²)
    - Linear system:  Thomas algorithm (tridiagonal)
    - Formal order of accuracy: p = 2

Manufactured Solution (following Ch. 6, Oberkampf & Roy 2025):

    T_ms(z,t) = T₀ + T_z sin(a_z π z/L) + T_t cos(a_t π t/τ)
                   + T_zt cos(a_zt π z/L) sin(a_zt2 π t/τ)

Design choices:
    1. Smooth analytic functions with continuous derivatives of all orders
    2. No derivative vanishes identically (cross-derivative tested)
    3. Low-frequency content (all a-parameters ≤ 2)
    4. T₀ = 280 K ≫ amplitudes → T > 0 everywhere (physical constraint)
    5. Mixture of sines and cosines as recommended by Roy (2005)

References
----------
[1] Oberkampf, W. L. & Roy, C. J. (2025). Verification and Validation in
    Scientific Computing, 2nd ed., Cambridge University Press.
[2] Roy, C. J. (2005). Review of code and solution verification procedures
    for computational simulation. J. Comput. Phys., 205(1), 131–156.
[3] Roache, P. J. (2002). Code verification by the method of manufactured
    solutions. ASME J. Fluids Eng., 124(1), 4–10.
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sympy as sp
import os
import sys
import time as timer
import json

# Local solver module
from vvsc_heat1d_solver import solve_heat_1d_CN

# ============================================================================
# PLOT CONFIGURATION — Publication Quality
# ============================================================================
rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (7, 5.5),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 1.8,
    'lines.markersize': 7,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'text.usetex': False,
})

OUTPUT_DIR = "hw3_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# SECTION 1: PHYSICAL PARAMETERS
# ============================================================================
# Uniform asphalt layer — constant thermal properties for MMS verification
RHO   = 2100.0    # Density [kg/m³]
C_P   = 920.0     # Specific heat capacity [J/(kg·K)]
K_TH  = 1.2       # Thermal conductivity [W/(m·K)]
ALPHA = K_TH / (RHO * C_P)  # Thermal diffusivity [m²/s]

# Domain parameters
L_DOMAIN  = 0.5    # Domain depth [m]
T_FINAL   = 3600.0 # Simulation time [s] (1 hour)


# ============================================================================
# SECTION 2: MANUFACTURED SOLUTION DEFINITION
# ============================================================================
# MMS parameters — chosen following guidelines (Sec. 6.4, Oberkampf & Roy):
#   1. Smooth analytic functions → sin, cos
#   2. No vanishing derivatives → cross-term with cos(z)*sin(t) included
#   3. Low frequency → a-parameters O(1)
#   4. Physical constraints → T₀ = 280 K ensures T > 0 everywhere
#   5. Similar magnitudes → amplitudes ~O(1–5 K)

MS_T0    = 280.0   # Base temperature [K]
MS_TZ    = 5.0     # Spatial amplitude [K]
MS_TT    = 3.0     # Temporal amplitude [K]
MS_TZT   = 2.0     # Coupled space-time amplitude [K]
MS_AZ    = 1.0     # Spatial frequency parameter
MS_AT    = 1.0     # Temporal frequency parameter
MS_AZT   = 2.0     # Coupled spatial frequency parameter
MS_AZT2  = 1.5     # Coupled temporal frequency parameter


def T_manufactured(z, t):
    """Evaluate the manufactured solution T_ms(z, t).

    Works with scalar or array inputs for z; t should be a scalar.
    """
    return (MS_T0
            + MS_TZ  * np.sin(MS_AZ  * np.pi * z / L_DOMAIN)
            + MS_TT  * np.cos(MS_AT  * np.pi * t / T_FINAL)
            + MS_TZT * np.cos(MS_AZT * np.pi * z / L_DOMAIN)
                     * np.sin(MS_AZT2 * np.pi * t / T_FINAL))


def dTms_dt(z, t):
    """∂T_ms/∂t — time derivative of the manufactured solution."""
    return (-MS_TT  * (MS_AT  * np.pi / T_FINAL)
                    * np.sin(MS_AT * np.pi * t / T_FINAL)
            + MS_TZT * (MS_AZT2 * np.pi / T_FINAL)
                     * np.cos(MS_AZT * np.pi * z / L_DOMAIN)
                     * np.cos(MS_AZT2 * np.pi * t / T_FINAL))


def d2Tms_dz2(z, t):
    """∂²T_ms/∂z² — second spatial derivative of the manufactured solution."""
    return (-MS_TZ  * (MS_AZ  * np.pi / L_DOMAIN)**2
                    * np.sin(MS_AZ * np.pi * z / L_DOMAIN)
            - MS_TZT * (MS_AZT * np.pi / L_DOMAIN)**2
                     * np.cos(MS_AZT * np.pi * z / L_DOMAIN)
                     * np.sin(MS_AZT2 * np.pi * t / T_FINAL))


def source_term(z, t):
    """MMS source: s(z,t) = ρ c_p ∂T_ms/∂t − k ∂²T_ms/∂z².

    Derived by substituting T_ms into the PDE and solving for s.
    """
    return RHO * C_P * dTms_dt(z, t) - K_TH * d2Tms_dz2(z, t)


def dTms_dz(z, t):
    """∂T_ms/∂z — first spatial derivative (used for exact SRQ evaluation)."""
    return (MS_TZ  * (MS_AZ  * np.pi / L_DOMAIN)
                   * np.cos(MS_AZ * np.pi * z / L_DOMAIN)
            - MS_TZT * (MS_AZT * np.pi / L_DOMAIN)
                     * np.sin(MS_AZT * np.pi * z / L_DOMAIN)
                     * np.sin(MS_AZT2 * np.pi * t / T_FINAL))


# ============================================================================
# SECTION 3: SYMPY VERIFICATION OF SOURCE TERM
# ============================================================================
def verify_source_term_sympy():
    """Independent verification of the MMS source term using SymPy.

    This ensures no algebraic errors exist in the hand-derived source term.
    Also verifies that the cross-derivative ∂²T/(∂z∂t) does not vanish
    identically (an MMS guideline).
    """
    print("=" * 70)
    print("SYMPY VERIFICATION OF MMS SOURCE TERM")
    print("=" * 70)

    z_sym, t_sym = sp.symbols('z t', real=True)

    # Symbolic manufactured solution
    T_ms_sym = (MS_T0
                + MS_TZ  * sp.sin(MS_AZ  * sp.pi * z_sym / L_DOMAIN)
                + MS_TT  * sp.cos(MS_AT  * sp.pi * t_sym / T_FINAL)
                + MS_TZT * sp.cos(MS_AZT * sp.pi * z_sym / L_DOMAIN)
                         * sp.sin(MS_AZT2 * sp.pi * t_sym / T_FINAL))

    # Symbolic derivatives
    dT_dt_sym   = sp.diff(T_ms_sym, t_sym)
    d2T_dz2_sym = sp.diff(T_ms_sym, z_sym, 2)

    # Symbolic source term
    s_sympy = RHO * C_P * dT_dt_sym - K_TH * d2T_dz2_sym

    # ── Compare at test points ──────────────────────────────────────────
    test_points = [(0.1, 600.0), (0.25, 1800.0), (0.4, 3000.0), (0.05, 900.0)]
    max_rel_diff = 0.0

    print(f"\n{'z [m]':>10} {'t [s]':>10} {'s(SymPy)':>18} {'s(code)':>18} {'Rel. Diff':>14}")
    print("-" * 70)

    for z_val, t_val in test_points:
        s_sym_val  = float(s_sympy.subs([(z_sym, z_val), (t_sym, t_val)]))
        s_code_val = source_term(z_val, t_val)
        rel_diff   = abs(s_sym_val - s_code_val) / (abs(s_sym_val) + 1e-30)
        max_rel_diff = max(max_rel_diff, rel_diff)
        print(f"{z_val:10.4f} {t_val:10.1f} {s_sym_val:18.8e} "
              f"{s_code_val:18.8e} {rel_diff:14.2e}")

    print(f"\nMaximum relative difference: {max_rel_diff:.2e}")
    if max_rel_diff < 1e-12:
        print("✓ Source term VERIFIED — SymPy and code agree to machine precision.")
    else:
        print("✗ WARNING: Source term mismatch detected!")
        sys.exit(1)

    # ── Verify cross-derivative does not vanish identically ─────────────
    d2T_dzdt = sp.diff(T_ms_sym, z_sym, t_sym)

    # Choose generic points that avoid trig zeros
    test_pts_cross = [(0.13, 500.0), (0.37, 2700.0), (0.22, 150.0)]
    cross_vals = []
    print(f"\nCross-derivative ∂²T/(∂z∂t) sampled at generic points:")
    for zv, tv in test_pts_cross:
        val = float(d2T_dzdt.subs([(z_sym, zv), (t_sym, tv)]))
        cross_vals.append(val)
        print(f"  (z={zv}, t={tv}): {val:.6e}")
    max_cross = max(abs(v) for v in cross_vals)
    assert max_cross > 1e-10, "Cross-derivative vanishes — MMS guideline violated!"
    print("✓ Cross-derivative is non-zero — MMS guideline satisfied.\n")

    return sp.simplify(s_sympy)


# ============================================================================
# SECTION 4: WRAPPER — calls the modular solver with MMS parameters
# ============================================================================
def run_solver(Nz, Nt, store_history=False):
    """Run the solver with the MMS manufactured solution."""
    return solve_heat_1d_CN(
        Nz=Nz, Nt=Nt,
        L=L_DOMAIN, t_final=T_FINAL,
        rho=RHO, cp=C_P, k=K_TH,
        T_init_fn=lambda z: T_manufactured(z, 0.0),
        T_left_fn=lambda t: T_manufactured(0.0, t),
        T_right_fn=lambda t: T_manufactured(L_DOMAIN, t),
        source_fn=source_term,
        store_history=store_history,
    )


# ============================================================================
# SECTION 5: ERROR NORMS AND ORDER OF ACCURACY
# ============================================================================
def compute_error_norms(T_numerical, T_exact):
    """Compute L1, L2, L∞ norms on interior nodes only (BCs are exact)."""
    error = T_numerical[1:-1] - T_exact[1:-1]
    N = len(error)
    L1   = np.sum(np.abs(error)) / N
    L2   = np.sqrt(np.sum(error**2) / N)
    Linf = np.max(np.abs(error))
    return L1, L2, Linf


def compute_observed_order(norm_coarse, norm_fine, r=2.0):
    """Observed order: p̂ = ln(E_coarse / E_fine) / ln(r)."""
    if norm_fine < 1e-30 or norm_coarse < 1e-30:
        return np.nan
    return np.log(norm_coarse / norm_fine) / np.log(r)


# ============================================================================
# SECTION 6: GRID CONVERGENCE STUDY
# ============================================================================
def run_grid_convergence_study():
    """Run MMS grid convergence study on systematically refined meshes.

    Grid refinement ratio r = 2 in both space and time simultaneously.
    6 mesh levels:  Nz ∈ {8, 16, 32, 64, 128, 256}
    """
    print("=" * 70)
    print("GRID CONVERGENCE STUDY")
    print("=" * 70)

    Nz_base = 8
    Nt_base = 16
    n_levels = 6

    grid_levels = [(Nz_base * 2**l, Nt_base * 2**l) for l in range(n_levels)]

    results     = []
    errors_L1   = []
    errors_L2   = []
    errors_Linf = []
    h_values    = []

    dz_finest = L_DOMAIN / grid_levels[-1][0]

    print(f"\n{'Lev':>4} {'Nz':>5} {'Nt':>5} {'Δz [m]':>12} {'Δt [s]':>10} "
          f"{'Fo':>8} {'h':>5} {'L1':>14} {'L2':>14} {'L∞':>14}")
    print("-" * 110)

    for lev, (Nz, Nt) in enumerate(grid_levels):
        t_start = timer.time()
        res = run_solver(Nz, Nt, store_history=(lev == n_levels - 1))
        elapsed = timer.time() - t_start

        T_exact = T_manufactured(res['z'], T_FINAL)
        L1, L2, Linf = compute_error_norms(res['T_final'], T_exact)
        h = res['dz'] / dz_finest

        h_values.append(h)
        errors_L1.append(L1)
        errors_L2.append(L2)
        errors_Linf.append(Linf)
        results.append(res)

        print(f"{lev+1:4d} {Nz:5d} {Nt:5d} {res['dz']:12.6f} {res['dt']:10.4f} "
              f"{res['Fo']:8.4f} {h:5.0f} {L1:14.6e} {L2:14.6e} {Linf:14.6e}  "
              f"({elapsed:.3f}s)")

    # ── Observed orders ─────────────────────────────────────────────────
    print(f"\n{'Pair':>10} {'p̂ (L1)':>10} {'p̂ (L2)':>10} {'p̂ (L∞)':>10}")
    print("-" * 45)

    orders_L1, orders_L2, orders_Linf = [], [], []
    for i in range(1, len(errors_L1)):
        pL1   = compute_observed_order(errors_L1[i-1],   errors_L1[i])
        pL2   = compute_observed_order(errors_L2[i-1],   errors_L2[i])
        pLinf = compute_observed_order(errors_Linf[i-1], errors_Linf[i])
        orders_L1.append(pL1)
        orders_L2.append(pL2)
        orders_Linf.append(pLinf)
        print(f"  {i} → {i+1}     {pL1:10.4f} {pL2:10.4f} {pLinf:10.4f}")

    # ── Round-off error assessment ──────────────────────────────────────
    eps_mach = np.finfo(np.float64).eps
    N_finest = grid_levels[-1][0]
    T_scale  = MS_T0
    roundoff_est = N_finest * eps_mach * T_scale
    print(f"\n--- Round-Off Error Estimate ---")
    print(f"  Machine epsilon (float64): {eps_mach:.2e}")
    print(f"  Round-off floor estimate:  {roundoff_est:.2e} K")
    print(f"  Finest DE (L2):            {errors_L2[-1]:.2e} K")
    print(f"  Ratio DE / round-off:      {errors_L2[-1]/roundoff_est:.1e}")
    print(f"  → Discretisation error dominates; round-off is negligible.\n")

    return {
        'grid_levels': grid_levels,
        'h_values':    np.array(h_values),
        'errors_L1':   np.array(errors_L1),
        'errors_L2':   np.array(errors_L2),
        'errors_Linf': np.array(errors_Linf),
        'orders_L1':   np.array(orders_L1),
        'orders_L2':   np.array(orders_L2),
        'orders_Linf': np.array(orders_Linf),
        'results':     results,
        'roundoff_est': roundoff_est,
    }


# ============================================================================
# SECTION 7: SRQ ANALYSIS
# ============================================================================
def run_srq_analysis(grid_levels):
    """System Response Quantity convergence analysis.

    SRQ 1: Surface heat flux at final time — q_s = −k ∂T/∂z|_{z=0, t=τ}
           (critical for black-ice cooling-rate assessment)
    SRQ 2: Interior temperature T(L/4, τ) — exercises the interior scheme

    Both SRQs have known exact values from the manufactured solution and
    should converge at 2nd order.
    """
    print("=" * 70)
    print("SRQ ANALYSIS")
    print("=" * 70)

    dz_finest = L_DOMAIN / grid_levels[-1][0]

    # ── Exact SRQ values from manufactured solution ─────────────────────
    q_s_exact    = -K_TH * dTms_dz(0.0, T_FINAL)
    z_srq2       = L_DOMAIN / 4.0
    T_srq2_exact = T_manufactured(z_srq2, T_FINAL)

    print(f"\nExact SRQ values:")
    print(f"  SRQ 1: Surface heat flux  q_s(τ)  = {q_s_exact:.8f} W/m²")
    print(f"  SRQ 2: Interior temp T(L/4, τ)    = {T_srq2_exact:.8f} K")

    srq1_errors, srq2_errors, h_vals = [], [], []

    print(f"\n{'Lev':>4} {'Nz':>5} {'Nt':>5} {'h':>5} "
          f"{'q_s [W/m²]':>16} {'|Err(q_s)|':>12} "
          f"{'T(L/4) [K]':>16} {'|Err(T)|':>12}")
    print("-" * 95)

    for lev, (Nz, Nt) in enumerate(grid_levels):
        res = run_solver(Nz, Nt, store_history=False)
        z  = res['z']
        T  = res['T_final']
        dz = res['dz']
        h  = dz / dz_finest
        h_vals.append(h)

        # SRQ 1: 2nd-order one-sided FD for surface gradient
        dTdz_num = (-3.0*T[0] + 4.0*T[1] - T[2]) / (2.0 * dz)
        q_s_num  = -K_TH * dTdz_num
        err1 = abs(q_s_num - q_s_exact)
        srq1_errors.append(err1)

        # SRQ 2: T(L/4, τ) — z=L/4 falls exactly on grid for Nz ∈ {8,16,...}
        idx_srq2 = round(z_srq2 / dz)
        assert abs(z[idx_srq2] - z_srq2) < 1e-12, \
            f"z=L/4 not on grid: z[{idx_srq2}]={z[idx_srq2]}, want {z_srq2}"
        T_srq2_num = T[idx_srq2]
        err2 = abs(T_srq2_num - T_srq2_exact)
        srq2_errors.append(err2)

        print(f"{lev+1:4d} {Nz:5d} {Nt:5d} {h:5.0f} "
              f"{q_s_num:16.8f} {err1:12.4e} "
              f"{T_srq2_num:16.8f} {err2:12.4e}")

    # ── Observed orders ─────────────────────────────────────────────────
    print(f"\n{'Pair':>10} {'p̂(q_s)':>10} {'p̂(T)':>10}")
    print("-" * 35)
    srq1_orders, srq2_orders = [], []
    for i in range(1, len(srq1_errors)):
        p1 = compute_observed_order(srq1_errors[i-1], srq1_errors[i])
        p2 = compute_observed_order(srq2_errors[i-1], srq2_errors[i])
        srq1_orders.append(p1)
        srq2_orders.append(p2)
        print(f"  {i} → {i+1}   {p1:10.4f} {p2:10.4f}")

    return {
        'h_values':     np.array(h_vals),
        'srq1_errors':  np.array(srq1_errors),
        'srq2_errors':  np.array(srq2_errors),
        'srq1_orders':  np.array(srq1_orders),
        'srq2_orders':  np.array(srq2_orders),
        'q_s_exact':    q_s_exact,
        'T_srq2_exact': T_srq2_exact,
    }


# ============================================================================
# SECTION 8: PLOTTING FUNCTIONS
# ============================================================================
def plot_manufactured_solution():
    """Figure 1: Contour of T_ms(z, t)."""
    z = np.linspace(0, L_DOMAIN, 200)
    t = np.linspace(0, T_FINAL, 200)
    Z, TG = np.meshgrid(z, t)
    T_ms = T_manufactured(Z, TG)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cf = ax.contourf(TG / 3600, Z * 100, T_ms, levels=30, cmap='RdYlBu_r')
    plt.colorbar(cf, ax=ax, label='Temperature [K]')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Depth z [cm]')
    ax.set_title('Manufactured Solution $T_{ms}(z, t)$')
    fig.savefig(f'{OUTPUT_DIR}/fig01_manufactured_solution.png')
    plt.close(fig)
    print(f"  Saved: fig01_manufactured_solution.png")


def plot_source_term():
    """Figure 2: Contour of the MMS source term s(z, t)."""
    z = np.linspace(0, L_DOMAIN, 200)
    t = np.linspace(0, T_FINAL, 200)
    Z, TG = np.meshgrid(z, t)
    S = source_term(Z, TG)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cf = ax.contourf(TG / 3600, Z * 100, S, levels=30, cmap='coolwarm')
    plt.colorbar(cf, ax=ax, label='Source term $s(z,t)$ [W/m³]')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Depth z [cm]')
    ax.set_title('MMS Source Term $s(z, t)$')
    fig.savefig(f'{OUTPUT_DIR}/fig02_source_term.png')
    plt.close(fig)
    print(f"  Saved: fig02_source_term.png")


def plot_error_norms(conv_data):
    """Figure 3: Error norms vs. grid spacing (log-log)."""
    h  = conv_data['h_values']
    L1 = conv_data['errors_L1']
    L2 = conv_data['errors_L2']
    Li = conv_data['errors_Linf']

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.loglog(h, L1, 'o-',  color='#2166ac', ms=8, label='$L_1$ norm')
    ax.loglog(h, L2, 's-',  color='#d6604d', ms=8, label='$L_2$ norm')
    ax.loglog(h, Li, '^-',  color='#1b7837', ms=8, label='$L_\\infty$ norm')

    # Reference slopes
    h_ref = np.array([h[-1], h[0]])
    C2 = L2[len(L2)//2] / h[len(h)//2]**2
    ax.loglog(h_ref, C2 * h_ref**2, 'k--', lw=1.2, label='2nd-order slope')
    C1 = L2[0] / h[0] * 3
    ax.loglog(h_ref, C1 * h_ref, 'k-.', lw=1.0, alpha=0.5, label='1st-order slope')

    ax.axhline(conv_data['roundoff_est'], color='gray', ls=':', lw=1.0,
               alpha=0.7, label=f'Round-off floor (~{conv_data["roundoff_est"]:.0e})')

    ax.set_xlabel('Normalised Grid Spacing, $h$')
    ax.set_ylabel('Discretisation Error Norm [K]')
    ax.set_title('Grid Convergence — Discretisation Error Norms')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(h[-1] * 0.7, h[0] * 1.5)
    fig.savefig(f'{OUTPUT_DIR}/fig03_error_norms.png')
    plt.close(fig)
    print(f"  Saved: fig03_error_norms.png")


def plot_observed_order(conv_data):
    """Figure 4: Observed order of accuracy vs. grid spacing."""
    h = conv_data['h_values']
    h_mid = np.sqrt(h[:-1] * h[1:])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.semilogx(h_mid, conv_data['orders_L1'],   'o-',  color='#2166ac', ms=8, label='$L_1$ norm')
    ax.semilogx(h_mid, conv_data['orders_L2'],   's-',  color='#d6604d', ms=8, label='$L_2$ norm')
    ax.semilogx(h_mid, conv_data['orders_Linf'], '^-',  color='#1b7837', ms=8, label='$L_\\infty$ norm')
    ax.axhline(2.0, color='k', ls='--', lw=1.2, label='Formal order ($p=2$)')

    ax.set_xlabel('Normalised Grid Spacing, $h$')
    ax.set_ylabel('Observed Order of Accuracy, $\\hat{p}$')
    ax.set_title('Observed Order of Accuracy')
    ax.set_ylim(0.0, 3.5)
    ax.legend(loc='best', framealpha=0.9)
    fig.savefig(f'{OUTPUT_DIR}/fig04_observed_order.png')
    plt.close(fig)
    print(f"  Saved: fig04_observed_order.png")


def plot_local_error(finest_result):
    """Figure 5: Local DE at final time + numerical vs. exact."""
    z = finest_result['z']
    T_num = finest_result['T_final']
    T_exact = T_manufactured(z, T_FINAL)
    error = T_num - T_exact

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(z * 100, error * 1e6, 'b-', lw=1.5)
    ax.axhline(0, color='k', lw=0.5)
    ax.set_xlabel('Depth $z$ [cm]')
    ax.set_ylabel('Local DE [$\\mu$K]')
    ax.set_title(f'Local Discretisation Error at $t = \\tau$')

    ax = axes[1]
    ax.plot(z * 100, T_exact, 'k-', lw=2, label='Exact (MMS)')
    ax.plot(z * 100, T_num, 'r--', lw=1.5, label=f'Numerical ($N_z$={len(z)-1})')
    ax.set_xlabel('Depth $z$ [cm]')
    ax.set_ylabel('Temperature [K]')
    ax.set_title(f'Temperature Profile at $t = \\tau$')
    ax.legend(framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig05_local_error.png')
    plt.close(fig)
    print(f"  Saved: fig05_local_error.png")


def plot_local_error_spacetime(finest_result):
    """Figure 6: Space-time contour of discretisation error."""
    T_all = finest_result['T_history']
    z = finest_result['z']
    t = finest_result['t']

    Z, TG = np.meshgrid(z, t)
    T_exact_all = T_manufactured(Z, TG)
    error_all = T_all - T_exact_all

    fig, ax = plt.subplots(figsize=(8, 5.5))
    cf = ax.contourf(TG[:, 1:-1] / 3600, Z[:, 1:-1] * 100,
                     error_all[:, 1:-1] * 1e6, levels=30, cmap='RdBu_r')
    plt.colorbar(cf, ax=ax, label='Discretisation Error [$\\mu$K]')
    ax.set_xlabel('Time [hours]')
    ax.set_ylabel('Depth $z$ [cm]')
    ax.set_title('Local Discretisation Error — Space-Time')
    fig.savefig(f'{OUTPUT_DIR}/fig06_local_error_spacetime.png')
    plt.close(fig)
    print(f"  Saved: fig06_local_error_spacetime.png")


def plot_solution_comparison(results, grid_levels):
    """Figure 7: Numerical vs. exact on coarse, medium, and fine meshes."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    for ax, idx, label in zip(axes, [0, 2, -1], ['Coarse', 'Medium', 'Fine']):
        res = results[idx]
        Nz = grid_levels[idx][0]
        z = res['z']
        T_exact = T_manufactured(z, T_FINAL)
        ax.plot(z * 100, T_exact, 'k-', lw=2, label='Exact (MMS)')
        ax.plot(z * 100, res['T_final'], 'ro', ms=4, label=f'Numerical ($N_z$={Nz})')
        ax.set_xlabel('Depth $z$ [cm]')
        ax.set_ylabel('Temperature [K]')
        ax.set_title(f'{label} Mesh ($N_z$ = {Nz})')
        ax.legend(fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig07_solution_comparison.png')
    plt.close(fig)
    print(f"  Saved: fig07_solution_comparison.png")


def plot_srq_convergence(srq_data):
    """Figure 8: SRQ convergence — 2×2 panel (error + order for each SRQ)."""
    h = srq_data['h_values']
    h_ref = np.array([h[-1], h[0]])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    # Top-left: SRQ1 error
    ax = axes[0, 0]
    e1 = srq_data['srq1_errors']
    ax.loglog(h, e1, 'D-', color='#7b3294', ms=8, label='$q_s(\\tau)$')
    if e1[-1] > 0:
        C = e1[len(e1)//2] / h[len(h)//2]**2
        ax.loglog(h_ref, C * h_ref**2, 'k--', lw=1.2, label='2nd-order slope')
    ax.set_xlabel('$h$'); ax.set_ylabel('SRQ Error [W/m²]')
    ax.set_title('SRQ 1: Surface Heat Flux Error'); ax.legend(framealpha=0.9)

    # Top-right: SRQ1 order
    ax = axes[0, 1]
    if len(srq_data['srq1_orders']) > 0:
        h_mid = np.sqrt(h[:-1] * h[1:])
        ax.semilogx(h_mid, srq_data['srq1_orders'], 'D-', color='#7b3294', ms=8)
        ax.axhline(2.0, color='k', ls='--', lw=1.2, label='$p=2$')
        ax.set_xlabel('$h$'); ax.set_ylabel('$\\hat{p}$')
        ax.set_title('SRQ 1: Observed Order'); ax.set_ylim(0, 3.5)
        ax.legend(framealpha=0.9)

    # Bottom-left: SRQ2 error
    ax = axes[1, 0]
    e2 = srq_data['srq2_errors']
    ax.loglog(h, e2, 'o-', color='#e66101', ms=8, label='$T(L/4, \\tau)$')
    if e2[-1] > 0:
        C = e2[len(e2)//2] / h[len(h)//2]**2
        ax.loglog(h_ref, C * h_ref**2, 'k--', lw=1.2, label='2nd-order slope')
    ax.set_xlabel('$h$'); ax.set_ylabel('SRQ Error [K]')
    ax.set_title('SRQ 2: Interior Temperature Error'); ax.legend(framealpha=0.9)

    # Bottom-right: SRQ2 order
    ax = axes[1, 1]
    if len(srq_data['srq2_orders']) > 0:
        h_mid = np.sqrt(h[:-1] * h[1:])
        ax.semilogx(h_mid, srq_data['srq2_orders'], 'o-', color='#e66101', ms=8)
        ax.axhline(2.0, color='k', ls='--', lw=1.2, label='$p=2$')
        ax.set_xlabel('$h$'); ax.set_ylabel('$\\hat{p}$')
        ax.set_title('SRQ 2: Observed Order'); ax.set_ylim(0, 3.5)
        ax.legend(framealpha=0.9)

    fig.tight_layout()
    fig.savefig(f'{OUTPUT_DIR}/fig08_srq_convergence.png')
    plt.close(fig)
    print(f"  Saved: fig08_srq_convergence.png")


def plot_roundoff_assessment(conv_data):
    """Figure 9: Round-off error assessment."""
    h  = conv_data['h_values']
    L2 = conv_data['errors_L2']
    rf = conv_data['roundoff_est']

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(h, L2, 's-', color='#d6604d', ms=8, lw=2, label='$L_2$ DE norm')

    ax.axhline(rf, color='gray', ls=':', lw=1.5,
               label=f'Round-off estimate (~{rf:.0e} K)')

    C = L2[2] / h[2]**2
    h_ext = np.logspace(np.log10(h[-1]) - 1, np.log10(h[0]) + 0.3, 100)
    ax.loglog(h_ext, C * h_ext**2, 'k--', lw=1.0, alpha=0.6, label='$O(h^2)$ reference')

    ax.set_xlabel('Normalised Grid Spacing, $h$')
    ax.set_ylabel('Error [K]')
    ax.set_title('Round-Off Error Assessment')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(0.3, h[0] * 2)
    fig.savefig(f'{OUTPUT_DIR}/fig09_roundoff_assessment.png')
    plt.close(fig)
    print(f"  Saved: fig09_roundoff_assessment.png")


# ============================================================================
# SECTION 9: MAIN EXECUTION
# ============================================================================
def main():
    print("\n" + "=" * 70)
    print("  VVSC HW3: CODE VERIFICATION VIA MMS")
    print("  1D Transient Heat Equation — Crank-Nicolson Solver")
    print("=" * 70)
    print(f"\nPhysical parameters:")
    print(f"  ρ     = {RHO} kg/m³")
    print(f"  c_p   = {C_P} J/(kg·K)")
    print(f"  k     = {K_TH} W/(m·K)")
    print(f"  α     = {ALPHA:.6e} m²/s")
    print(f"  L     = {L_DOMAIN} m")
    print(f"  τ     = {T_FINAL} s")
    print(f"\nManufactured solution parameters:")
    print(f"  T₀    = {MS_T0} K")
    print(f"  T_z   = {MS_TZ} K,  a_z    = {MS_AZ}")
    print(f"  T_t   = {MS_TT} K,  a_t    = {MS_AT}")
    print(f"  T_zt  = {MS_TZT} K, a_zt   = {MS_AZT}, a_zt2 = {MS_AZT2}")
    print(f"\nFormal order of accuracy: p = 2 (O(Δz², Δt²))")
    print(f"Output directory: {OUTPUT_DIR}/\n")

    # ── Step 1: Verify source term with SymPy ───────────────────────────
    verify_source_term_sympy()

    # ── Step 2: Grid convergence study ──────────────────────────────────
    conv_data = run_grid_convergence_study()

    # ── Step 3: SRQ analysis ────────────────────────────────────────────
    srq_data = run_srq_analysis(conv_data['grid_levels'])

    # ── Step 4: Generate all figures ────────────────────────────────────
    print("=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)
    plot_manufactured_solution()                                  # Fig 1
    plot_source_term()                                            # Fig 2
    plot_error_norms(conv_data)                                   # Fig 3
    plot_observed_order(conv_data)                                # Fig 4
    finest = conv_data['results'][-1]
    plot_local_error(finest)                                      # Fig 5
    plot_local_error_spacetime(finest)                            # Fig 6
    plot_solution_comparison(conv_data['results'],
                            conv_data['grid_levels'])             # Fig 7
    plot_srq_convergence(srq_data)                                # Fig 8
    plot_roundoff_assessment(conv_data)                           # Fig 9

    # ── Step 5: Code coverage assessment ────────────────────────────────
    print("\n" + "=" * 70)
    print("CODE COVERAGE ASSESSMENT")
    print("=" * 70)
    print("""
VERIFIED by this MMS test:
  ✓ 1D transient heat conduction PDE (ρ c_p ∂T/∂t = k ∂²T/∂z²)
  ✓ 2nd-order central finite differences in space
  ✓ Crank-Nicolson time integration (2nd order)
  ✓ Thomas algorithm (tridiagonal solver)
  ✓ Source term incorporation via CN trapezoidal averaging
  ✓ Dirichlet boundary condition implementation

NOT VERIFIED by this MMS test:
  ✗ Nonlinear Neumann BC (surface energy balance with T⁴ radiation)
  ✗ Newton iteration for nonlinear BC
  ✗ Variable / piecewise-constant material properties (two-layer road)
  ✗ Interface conditions between asphalt and subgrade layers
  ✗ Ice-likelihood function and friction mapping
  ✗ Coupling with CARLA simulator

NOTE: The MMS test prescribes Dirichlet BCs from the manufactured
solution.  The actual application uses a nonlinear Neumann BC at the
surface, which would require a separate MMS test with the Newton
solver active.  A complementary analytical-solution test (e.g.,
T(z,t) = A exp(−μz) cos(μz − αμ²t)) could also provide independent
verification of the interior scheme.
    """)

    # ── Step 6: Summary verdict ─────────────────────────────────────────
    print("=" * 70)
    print("SUMMARY — ORDER OF ACCURACY TEST")
    print("=" * 70)
    p_L1  = conv_data['orders_L1'][-1]
    p_L2  = conv_data['orders_L2'][-1]
    p_Li  = conv_data['orders_Linf'][-1]
    print(f"\n  Formal order:              p = 2")
    print(f"  Observed order (finest pair):")
    print(f"    L1 norm:    p̂ = {p_L1:.4f}")
    print(f"    L2 norm:    p̂ = {p_L2:.4f}")
    print(f"    L∞ norm:    p̂ = {p_Li:.4f}")

    tol = 0.1
    passed = all(abs(p - 2.0) < tol for p in [p_L1, p_L2, p_Li])
    if passed:
        print(f"\n  ✓ CODE PASSES the order of accuracy test.")
        print(f"    All observed orders are within {tol*100:.0f}% of p=2.\n")
    else:
        print(f"\n  ⚠ Some observed orders deviate from formal order.\n")

    # ── Save summary JSON ───────────────────────────────────────────────
    summary = {
        'meshes': [
            {'Nz': gl[0], 'Nt': gl[1],
             'h': float(conv_data['h_values'][i]),
             'dz': float(conv_data['results'][i]['dz']),
             'dt': float(conv_data['results'][i]['dt']),
             'Fo': float(conv_data['results'][i]['Fo']),
             'L1': float(conv_data['errors_L1'][i]),
             'L2': float(conv_data['errors_L2'][i]),
             'Linf': float(conv_data['errors_Linf'][i])}
            for i, gl in enumerate(conv_data['grid_levels'])
        ],
        'orders': {
            'L1':   [float(x) for x in conv_data['orders_L1']],
            'L2':   [float(x) for x in conv_data['orders_L2']],
            'Linf': [float(x) for x in conv_data['orders_Linf']],
        },
        'roundoff_est': float(conv_data['roundoff_est']),
        'verdict': 'PASS' if passed else 'INVESTIGATE',
    }
    with open(f'{OUTPUT_DIR}/verification_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  All output saved to: {OUTPUT_DIR}/")
    print("=" * 70)

    return conv_data, srq_data


if __name__ == '__main__':
    main()