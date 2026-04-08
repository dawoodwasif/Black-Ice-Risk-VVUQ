#!/usr/bin/env python3
"""
===============================================================================
VVSC Homework 4: Solution Verification
===============================================================================
Application:  Black-Ice Risk Prediction — 1D Transient Heat Conduction
Course:       AOE/CS/ME 6444 — Verification and Validation in Scientific Computing
Author:       Dawood Wasif
===============================================================================

Physical scenario: Clear winter night in Blacksburg, VA
  - 6-hour simulation (sunset to midnight)
  - T_air drops linearly from 275 K to 268 K
  - Clear sky (no shortwave, atmospheric emissivity ~ 0.74)
  - Wind speed 2.5 m/s
  - Asphalt road column, depth 0.5 m

SRQs:
  1. Surface temperature at t = 6 hr:  T_s(tau)
  2. Surface heat flux at t = 6 hr:    q_s(tau) = -k dT/dz|_{z=0}
  3. Ice-onset time:                    t_ice (T_s first < 273.15 K)

Studies:
  A. Discretization error  (6 meshes, r=2, Newton tol=1e-14)
  B. Iterative error       (tolerance sweep on fine grid)
  C. Round-off error       (float64 vs float32)
  D. Total numerical uncertainty for fine grid & parametric study grid
===============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os, time

from vvsc_heat1d_solver import solve_heat_1d_CN_neumann

# ═════════════════════════════════════════════════════════════════════════════
# Physical Scenario Parameters
# ═════════════════════════════════════════════════════════════════════════════
RHO   = 2100.0      # density [kg/m^3]
CP    = 920.0       # specific heat [J/(kg K)]
K_MAT = 1.2         # thermal conductivity [W/(m K)]
L     = 0.5         # domain depth [m]
TAU   = 21600.0     # simulation time [s] (6 hours)

EPSILON     = 0.93   # asphalt emissivity
SIGMA_SB    = 5.670374419e-8   # Stefan-Boltzmann [W/(m^2 K^4)]
EPSILON_ATM = 0.74   # clear-sky atmospheric emissivity
V_WIND      = 2.5    # wind speed [m/s]
H_CONV      = 5.7 + 3.8 * V_WIND   # convective coeff [W/(m^2 K)] = 15.2

T_AIR_START = 275.0  # air temp at t=0 [K]
T_AIR_END   = 268.0  # air temp at t=tau [K]
T_DEEP      = 283.0  # deep-ground temp [K]
T_SURFACE_INIT = 278.0  # daytime equilibrium surface temp [K]
T_FREEZE    = 273.15    # freezing point [K]

OUT_DIR = "hw4_output"


def T_air(t):
    """Air temperature: linear drop from 275 to 268 K over 6 hours."""
    return T_AIR_START + (T_AIR_END - T_AIR_START) * (t / TAU)


def T_init(z):
    """Steady-state daytime profile: linear from T_surface to T_deep."""
    return T_SURFACE_INIT + (T_DEEP - T_SURFACE_INIT) * (z / L)


def T_right(t):
    """Bottom Dirichlet BC: constant deep-ground temperature."""
    return T_DEEP


def q_net(Ts, t):
    """
    Net heat flux INTO the surface [W/m^2].
    Positive = heating the surface.

    Components (nighttime, clear sky):
      Incoming LW:   eps * eps_atm * sigma * T_air^4
      Outgoing LW:  -eps * sigma * T_s^4
      Convection:   -h_conv * (T_s - T_air)
    """
    Ta = T_air(t)
    q_lw = EPSILON * SIGMA_SB * (EPSILON_ATM * Ta**4 - Ts**4)
    q_conv = -H_CONV * (Ts - Ta)
    return q_lw + q_conv


def dq_net_dTs(Ts, t):
    """Derivative of q_net w.r.t. T_s [W/(m^2 K)]."""
    return -4.0 * EPSILON * SIGMA_SB * Ts**3 - H_CONV


# ═════════════════════════════════════════════════════════════════════════════
# SRQ Extraction
# ═════════════════════════════════════════════════════════════════════════════
def extract_srqs(result):
    """Extract the three SRQs from a solver result dict."""
    z = result["z"]
    dz = result["dz"]
    T_final = result["T_final"]
    Ts_hist = result["T_surface_history"]
    t_arr = result["t"]

    # SRQ1: Surface temperature at final time
    Ts_tau = T_final[0]

    # SRQ2: Surface heat flux via 2nd-order one-sided FD
    #   dT/dz|_{z=0} = (-3T_0 + 4T_1 - T_2) / (2 dz)
    dTdz = (-3.0 * T_final[0] + 4.0 * T_final[1] - T_final[2]) / (2.0 * dz)
    qs_tau = -K_MAT * dTdz

    # SRQ3: Ice-onset time (first crossing below T_freeze)
    t_ice = np.nan
    for i in range(1, len(Ts_hist)):
        if Ts_hist[i] < T_FREEZE and Ts_hist[i - 1] >= T_FREEZE:
            # Linear interpolation
            frac = (T_FREEZE - Ts_hist[i - 1]) / (Ts_hist[i] - Ts_hist[i - 1])
            t_ice = t_arr[i - 1] + frac * (t_arr[i] - t_arr[i - 1])
            break

    return {"Ts_tau": float(Ts_tau),
            "qs_tau": float(qs_tau),
            "t_ice": float(t_ice)}


# ═════════════════════════════════════════════════════════════════════════════
# Richardson Extrapolation & GCI
# ═════════════════════════════════════════════════════════════════════════════
def richardson_extrap(f1, f2, f3, r=2.0):
    """
    Richardson extrapolation from three solutions on systematically
    refined grids with refinement ratio r.

    f1 = finest, f2 = next, f3 = coarsest.
    Returns p_hat, f_exact, DE_1, GCI_12.
    """
    num = f3 - f2
    den = f2 - f1
    if abs(den) < 1e-30:
        return np.nan, f1, 0.0, 0.0

    ratio = num / den
    if ratio <= 0:
        # Non-monotonic convergence: cannot compute p_hat
        return np.nan, np.nan, np.nan, np.nan

    p_hat = np.log(ratio) / np.log(r)

    # Extrapolated exact value
    DE_1 = (f1 - f2) / (r**p_hat - 1.0)
    f_exact = f1 + DE_1

    # GCI with safety factor
    p_formal = 2.0
    if abs(p_hat - p_formal) / p_formal < 0.1:
        Fs = 1.25
    else:
        Fs = 3.0

    GCI_12 = Fs * abs(DE_1)

    return p_hat, f_exact, DE_1, GCI_12


# ═════════════════════════════════════════════════════════════════════════════
# A. Discretization Error Study
# ═════════════════════════════════════════════════════════════════════════════
def run_de_study():
    """Run 6 systematically refined meshes (r=2)."""
    print("=" * 70)
    print("A. DISCRETIZATION ERROR STUDY")
    print("=" * 70)

    mesh_levels = [
        (16,   128),
        (32,   256),
        (64,   512),
        (128,  1024),
        (256,  2048),
        
    ]

    results = []
    for Nz, Nt in mesh_levels:
        print(f"  Running Nz={Nz:4d}, Nt={Nt:5d} ...", end="", flush=True)
        t0 = time.time()
        res = solve_heat_1d_CN_neumann(
            Nz, Nt, L, TAU, RHO, CP, K_MAT,
            T_init, T_right, q_net, dq_net_dTs,
            newton_tol=1e-12, newton_maxiter=100,
            store_history=False, dtype=np.float64,
        )
        elapsed = time.time() - t0
        srqs = extract_srqs(res)
        srqs["Nz"] = Nz
        srqs["Nt"] = Nt
        srqs["dz"] = res["dz"]
        srqs["dt"] = res["dt"]
        srqs["Fo"] = res["Fo"]
        srqs["newton_iters"] = res["newton_iters_total"]
        srqs["time_s"] = elapsed
        results.append(srqs)
        print(f"  Ts={srqs['Ts_tau']:.6f} K, qs={srqs['qs_tau']:.4f} W/m2"
              f", t_ice={srqs['t_ice']:.1f} s  ({elapsed:.2f}s)")

    return results


def analyze_de(results):
    """Richardson extrapolation and GCI for all SRQs."""
    n = len(results)
    srq_names = ["Ts_tau", "qs_tau", "t_ice"]
    srq_labels = ["T_s(tau) [K]", "q_s(tau) [W/m^2]", "t_ice [s]"]
    r = 2.0

    analysis = {}
    for sname, slabel in zip(srq_names, srq_labels):
        vals = [res[sname] for res in results]
        print(f"\n  {slabel}:")

        # Observed order from consecutive triplets
        orders = []
        for i in range(n - 2):
            f3, f2, f1 = vals[i], vals[i + 1], vals[i + 2]
            denom = f2 - f1
            numer = f3 - f2
            if abs(denom) > 1e-30 and numer / denom > 0:
                p = np.log(numer / denom) / np.log(r)
                orders.append(p)
            else:
                orders.append(np.nan)

        for i, p in enumerate(orders):
            print(f"    Triplet {i+1}-{i+2}-{i+3}: p_hat = {p:.4f}")

        # Richardson extrapolation using 3 finest grids
        f1, f2, f3 = vals[-1], vals[-2], vals[-3]
        p_hat, f_exact, DE_1, GCI_12 = richardson_extrap(f1, f2, f3, r)

        p_formal = 2.0
        if not np.isnan(p_hat) and abs(p_hat - p_formal) / p_formal < 0.1:
            Fs = 1.25
        else:
            Fs = 3.0

        print(f"    Richardson (finest 3): p_hat={p_hat:.4f}, "
              f"f_exact={f_exact:.6f}")
        print(f"    DE_1 = {DE_1:.6e}, GCI_12 = {GCI_12:.6e} (Fs={Fs:.2f})")

        analysis[sname] = {
            "values": vals,
            "orders": orders,
            "p_hat": p_hat,
            "f_exact": f_exact,
            "DE_fine": DE_1,
            "GCI_fine": GCI_12,
            "Fs": Fs,
        }

        # GCI for all consecutive pairs (for parametric study grid)
        gci_all = []
        for i in range(n - 2):
            f1p, f2p, f3p = vals[i + 2], vals[i + 1], vals[i]
            _, _, de, gci = richardson_extrap(f1p, f2p, f3p, r)
            gci_all.append(gci)
        analysis[sname]["gci_all"] = gci_all

    return analysis


# ═════════════════════════════════════════════════════════════════════════════
# B. Iterative Error Study
# ═════════════════════════════════════════════════════════════════════════════
def run_it_study():
    """Newton tolerance sweep on the fine grid (Nz=256)."""
    print("\n" + "=" * 70)
    print("B. ITERATIVE ERROR STUDY")
    print("=" * 70)

    Nz_fine, Nt_fine = 256, 2048
    tolerances = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-12]

    it_results = []
    for tol in tolerances:
        print(f"  Newton tol = {tol:.0e} ...", end="", flush=True)
        res = solve_heat_1d_CN_neumann(
            Nz_fine, Nt_fine, L, TAU, RHO, CP, K_MAT,
            T_init, T_right, q_net, dq_net_dTs,
            newton_tol=tol, newton_maxiter=100,
            store_history=False, dtype=np.float64,
        )
        srqs = extract_srqs(res)
        srqs["tol"] = tol
        srqs["newton_iters"] = res["newton_iters_total"]
        it_results.append(srqs)
        print(f"  Ts={srqs['Ts_tau']:.10f}, iters={srqs['newton_iters']}")

    return it_results


def analyze_it(it_results):
    """Compute iterative error relative to machine-zero solution."""
    ref = it_results[-1]  # tightest tolerance = reference
    srq_names = ["Ts_tau", "qs_tau", "t_ice"]

    analysis = {}
    for sname in srq_names:
        ref_val = ref[sname]
        errors = []
        for r in it_results:
            errors.append(abs(r[sname] - ref_val))
        analysis[sname] = {
            "ref_val": ref_val,
            "tolerances": [r["tol"] for r in it_results],
            "errors": errors,
        }
        print(f"\n  {sname} iterative errors (ref = {ref_val:.10f}):")
        for r, e in zip(it_results, errors):
            print(f"    tol={r['tol']:.0e}: error = {e:.6e}")

    return analysis


# ═════════════════════════════════════════════════════════════════════════════
# C. Round-Off Error Study
# ═════════════════════════════════════════════════════════════════════════════
def run_ro_study():
    """Compare float64 vs float32 on the fine grid."""
    print("\n" + "=" * 70)
    print("C. ROUND-OFF ERROR STUDY")
    print("=" * 70)

    Nz_fine, Nt_fine = 256, 2048

    ro_results = {}
    for dtype_name, dtype in [("float64", np.float64), ("float32", np.float32)]:
        tol = 1e-12 if dtype == np.float64 else 1e-6
        print(f"  Running {dtype_name} (tol={tol:.0e}) ...", end="", flush=True)
        res = solve_heat_1d_CN_neumann(
            Nz_fine, Nt_fine, L, TAU, RHO, CP, K_MAT,
            T_init, T_right, q_net, dq_net_dTs,
            newton_tol=tol, newton_maxiter=100,
            store_history=False, dtype=dtype,
        )
        srqs = extract_srqs(res)
        ro_results[dtype_name] = srqs
        print(f"  Ts={srqs['Ts_tau']:.10f}")

    analysis = {}
    for sname in ["Ts_tau", "qs_tau", "t_ice"]:
        diff = abs(ro_results["float64"][sname] - ro_results["float32"][sname])
        analysis[sname] = {
            "f64": ro_results["float64"][sname],
            "f32": ro_results["float32"][sname],
            "U_RO": diff,
        }
        print(f"  {sname}: f64={analysis[sname]['f64']:.10f}, "
              f"f32={analysis[sname]['f32']:.10f}, U_RO={diff:.6e}")

    return analysis


# ═════════════════════════════════════════════════════════════════════════════
# D. Total Numerical Uncertainty
# ═════════════════════════════════════════════════════════════════════════════
def compute_total_uncertainty(de_analysis, it_analysis, ro_analysis, de_results):
    """Compute U_NUM = U_DE + U_IT + U_RO for fine grid and parametric grid."""
    print("\n" + "=" * 70)
    print("D. TOTAL NUMERICAL UNCERTAINTY")
    print("=" * 70)

    srq_names = ["Ts_tau", "qs_tau", "t_ice"]
    srq_labels = ["T_s(tau)", "q_s(tau)", "t_ice"]

    summary = {}
    for sname, slabel in zip(srq_names, srq_labels):
        # ── Fine grid (mesh 5: Nz=256) ──
        U_DE_fine = de_analysis[sname]["GCI_fine"]
        # Iterative error at tol=1e-12
        U_IT_fine = it_analysis[sname]["errors"][-2]  # tol=1e-12
        # Use analytical float64 round-off estimate (not float32 comparison)
        # HW3 formula: U_RO ~ Nz * eps_mach * scale
        eps_mach = np.finfo(np.float64).eps
        Nz_fine = 256
        if sname == "Ts_tau":
            U_RO = Nz_fine * eps_mach * 280.0
        elif sname == "qs_tau":
            U_RO = Nz_fine * eps_mach * 100.0
        else:  # t_ice
            dTsdt = (T_SURFACE_INIT - 269.0) / TAU
            U_RO = Nz_fine * eps_mach * 280.0 / max(dTsdt, 1e-10)
        U_NUM_fine = U_DE_fine + U_IT_fine + U_RO

        # ── Parametric grid (mesh 2: Nz=32) ──
        # GCI for mesh 2 (need triplet 2,3,4 => indices 1,2,3)
        vals = de_analysis[sname]["values"]
        f1p, f2p, f3p = vals[3], vals[2], vals[1]
        _, _, _, GCI_param = richardson_extrap(f1p, f2p, f3p, 2.0)
        U_DE_param = GCI_param if not np.isnan(GCI_param) else abs(vals[1] - vals[2])
        # Iterative error at tol=1e-6
        U_IT_param = it_analysis[sname]["errors"][1]  # tol=1e-6
        # Parametric grid round-off (Nz=32)
        Nz_param = 32
        if sname == "Ts_tau":
            U_RO_param = Nz_param * eps_mach * 280.0
        elif sname == "qs_tau":
            U_RO_param = Nz_param * eps_mach * 100.0
        else:
            dTsdt_p = (T_SURFACE_INIT - 269.0) / TAU
            U_RO_param = Nz_param * eps_mach * 280.0 / max(dTsdt_p, 1e-10)
        U_NUM_param = U_DE_param + U_IT_param + U_RO_param

        summary[sname] = {
            "fine": {"U_DE": U_DE_fine, "U_IT": U_IT_fine,
                     "U_RO": U_RO, "U_NUM": U_NUM_fine,
                     "value": de_analysis[sname]["values"][-2]},
            "param": {"U_DE": U_DE_param, "U_IT": U_IT_param,
                      "U_RO": U_RO_param, "U_NUM": U_NUM_param,
                      "value": de_analysis[sname]["values"][1]},
            "DE_sign": de_analysis[sname]["DE_fine"],
            "p_hat": de_analysis[sname]["p_hat"],
            "Fs": de_analysis[sname]["Fs"],
        }

        print(f"\n  {slabel}:")
        print(f"    FINE GRID (Nz=256):  value={summary[sname]['fine']['value']:.6f}")
        print(f"      U_DE={U_DE_fine:.6e}, U_IT={U_IT_fine:.6e}, "
              f"U_RO={U_RO_param:.6e}")
        print(f"      U_NUM = {U_NUM_fine:.6e}")
        print(f"    PARAMETRIC (Nz=32):  value={summary[sname]['param']['value']:.6f}")
        print(f"      U_DE={U_DE_param:.6e}, U_IT={U_IT_param:.6e}, "
              f"U_RO={U_RO:.6e}")
        print(f"      U_NUM = {U_NUM_param:.6e}")

    return summary


# ═════════════════════════════════════════════════════════════════════════════
# Figure Generation
# ═════════════════════════════════════════════════════════════════════════════
def generate_figures(de_results, de_analysis, it_analysis, ro_analysis, total):
    """Generate all HW4 figures."""
    os.makedirs(OUT_DIR, exist_ok=True)

    plt.rcParams.update({
        "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
        "legend.fontsize": 9, "figure.dpi": 150,
    })

    # ── Fig 1: Surface temperature evolution (finest mesh) ──────────────
    print("\n  Generating Figure 1: Surface temperature evolution ...")
    res_fine = solve_heat_1d_CN_neumann(
        256, 2048, L, TAU, RHO, CP, K_MAT,
        T_init, T_right, q_net, dq_net_dTs,
        newton_tol=1e-12, store_history=True,
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    t_hrs = res_fine["t"] / 3600.0
    ax.plot(t_hrs, res_fine["T_surface_history"], "b-", lw=1.5,
            label=r"$T_s(t)$ (Nz=256)")
    ax.axhline(T_FREEZE, color="r", ls="--", lw=1, label="273.15 K (freezing)")
    ax.set_xlabel("Time [hours]")
    ax.set_ylabel("Surface Temperature [K]")
    ax.set_title("Surface Temperature Evolution — Clear Winter Night")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig01_surface_temp_evolution.png")
    plt.close(fig)

    # ── Fig 2: Temperature profiles at selected times ───────────────────
    print("  Generating Figure 2: Temperature profiles ...")
    T_hist = res_fine["T_history"]
    z_cm = res_fine["z"] * 100.0
    fig, ax = plt.subplots(figsize=(6, 5))
    times_hr = [0, 1, 2, 3, 4, 5, 6]
    for th in times_hr:
        idx = int(th * 2048 / 6)
        if idx >= T_hist.shape[0]:
            idx = T_hist.shape[0] - 1
        ax.plot(T_hist[idx, :], z_cm, label=f"t = {th} hr")
    ax.set_xlabel("Temperature [K]")
    ax.set_ylabel("Depth z [cm]")
    ax.set_title("Road Temperature Profiles")
    ax.invert_yaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig02_temperature_profiles.png")
    plt.close(fig)

    # ── Fig 3: SRQ convergence (log-log) ────────────────────────────────
    print("  Generating Figure 3: SRQ convergence ...")
    srq_keys = ["Ts_tau", "qs_tau", "t_ice"]
    srq_titles = [r"$T_s(\tau)$", r"$q_s(\tau)$", r"$t_{ice}$"]
    srq_units = ["K", "W/m$^2$", "s"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, skey, stitle, sunit in zip(axes, srq_keys, srq_titles, srq_units):
        vals = de_analysis[skey]["values"]
        f_ext = de_analysis[skey]["f_exact"]
        if np.isnan(f_ext):
            f_ext = vals[-1]  # fallback
        errors = [abs(v - f_ext) for v in vals]
        h_vals = [2**(4 - i) for i in range(len(vals))]
        ax.loglog(h_vals, errors, "o-", color="tab:blue", lw=1.5, ms=6)
        # 2nd-order reference
        ref = errors[-1] * np.array([(h / h_vals[-1])**2 for h in h_vals])
        ax.loglog(h_vals, ref, "--", color="gray", alpha=0.6, label="$O(h^2)$")
        ax.set_xlabel("Normalised Grid Spacing, $h$")
        ax.set_ylabel(f"SRQ Error [{sunit}]")
        ax.set_title(stitle)
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("SRQ Convergence with Grid Refinement", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig03_srq_convergence.png", bbox_inches="tight")
    plt.close(fig)

    # ── Fig 4: Observed order of accuracy ───────────────────────────────
    print("  Generating Figure 4: Observed order ...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, skey, stitle in zip(axes, srq_keys, srq_titles):
        orders = de_analysis[skey]["orders"]
        h_trip = [2**(5 - i - 1) for i in range(len(orders))]
        valid = [(h, p) for h, p in zip(h_trip, orders) if not np.isnan(p)]
        if valid:
            hs, ps = zip(*valid)
            ax.semilogx(hs, ps, "s-", color="tab:red", ms=7, lw=1.5)
        ax.axhline(2.0, color="k", ls="--", lw=1, label="$p = 2$")
        ax.set_xlabel("$h$")
        ax.set_ylabel(r"$\hat{p}$")
        ax.set_title(stitle)
        ax.set_ylim([0, 4])
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.suptitle("Observed Order of Accuracy", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig04_observed_order.png", bbox_inches="tight")
    plt.close(fig)

    # ── Fig 5: Iterative error vs tolerance ─────────────────────────────
    print("  Generating Figure 5: Iterative error ...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, skey, stitle, sunit in zip(axes, srq_keys, srq_titles, srq_units):
        tols = it_analysis[skey]["tolerances"]
        errs = it_analysis[skey]["errors"]
        # Skip zero errors for log scale
        valid = [(t, e) for t, e in zip(tols, errs) if e > 0]
        if valid:
            ts, es = zip(*valid)
            ax.loglog(ts, es, "D-", color="tab:green", ms=6, lw=1.5)
        ax.set_xlabel("Newton Tolerance")
        ax.set_ylabel(f"Iterative Error [{sunit}]")
        ax.set_title(stitle)
        ax.grid(True, which="both", alpha=0.3)
    fig.suptitle("Iterative Error vs Newton Tolerance (Nz=256)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig05_iterative_error.png", bbox_inches="tight")
    plt.close(fig)

    # ── Fig 6: Numerical uncertainty budget (bar chart) ─────────────────
    print("  Generating Figure 6: Uncertainty budget ...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, scenario, title in zip(axes, ["fine", "param"],
                                    ["Fine Grid (Nz=256)",
                                     "Parametric Grid (Nz=32)"]):
        labels = [r"$T_s(\tau)$", r"$q_s(\tau)$", r"$t_{ice}$"]
        U_DE_vals = [total[sk][scenario]["U_DE"] for sk in srq_keys]
        U_IT_vals = [total[sk][scenario]["U_IT"] for sk in srq_keys]
        U_RO_vals = [abs(total[sk][scenario]["U_RO"]) for sk in srq_keys]

        x = np.arange(len(labels))
        w = 0.25
        ax.bar(x - w, U_DE_vals, w, label=r"$U_{DE}$", color="tab:blue")
        ax.bar(x, U_IT_vals, w, label=r"$U_{IT}$", color="tab:orange")
        ax.bar(x + w, U_RO_vals, w, label=r"$U_{RO}$", color="tab:green")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Uncertainty")
        ax.set_title(title)
        ax.set_yscale("log")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.suptitle(r"Numerical Uncertainty Budget: $U_{NUM} = U_{DE} + U_{IT} + U_{RO}$",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig06_uncertainty_budget.png", bbox_inches="tight")
    plt.close(fig)

    # ── Fig 7: GCI error bars on fine-grid solution ─────────────────────
    print("  Generating Figure 7: GCI error bars ...")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, skey, stitle, sunit in zip(axes, srq_keys, srq_titles, srq_units):
        vals = de_analysis[skey]["values"]
        DE = de_analysis[skey]["DE_fine"]
        GCI = de_analysis[skey]["GCI_fine"]
        f_ext = de_analysis[skey]["f_exact"]
        h_vals = [2**(4 - i) for i in range(len(vals))]

        ax.plot(h_vals, vals, "o-", color="tab:blue", ms=6, label="Computed")
        if not np.isnan(f_ext):
            ax.axhline(f_ext, color="tab:red", ls="--", lw=1,
                       label=f"Richardson extrap.")
            # GCI band on finest mesh
            ax.errorbar(h_vals[-2], vals[-2], yerr=GCI,
                        fmt="s", color="tab:orange", ms=8, capsize=5,
                        label=f"GCI = {GCI:.4e}")
        ax.set_xlabel("$h$")
        ax.set_ylabel(f"{stitle} [{sunit}]")
        ax.set_title(f"{stitle} vs Grid Spacing")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle("Grid Convergence with GCI Error Bars", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/fig07_gci_error_bars.png", bbox_inches="tight")
    plt.close(fig)

    print(f"  All figures saved to {OUT_DIR}/")


# ═════════════════════════════════════════════════════════════════════════════
# Main
# ═════════════════════════════════════════════════════════════════════════════
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # A. Discretization error
    de_results = run_de_study()
    de_analysis = analyze_de(de_results)

    # B. Iterative error
    it_results = run_it_study()
    it_analysis = analyze_it(it_results)

    # C. Round-off error
    ro_analysis = run_ro_study()

    # D. Total numerical uncertainty
    total = compute_total_uncertainty(de_analysis, it_analysis,
                                     ro_analysis, de_results)

    # Generate figures
    generate_figures(de_results, de_analysis, it_analysis, ro_analysis, total)

    # Export summary JSON
    summary = {
        "scenario": {
            "T_air_start": T_AIR_START, "T_air_end": T_AIR_END,
            "V_wind": V_WIND, "T_deep": T_DEEP,
            "epsilon": EPSILON, "epsilon_atm": EPSILON_ATM,
            "h_conv": H_CONV, "L": L, "tau": TAU,
            "rho": RHO, "cp": CP, "k": K_MAT,
        },
        "meshes": de_results,
        "de_analysis": {},
        "it_analysis": {},
        "ro_analysis": {},
        "total_uncertainty": {},
    }
    for skey in ["Ts_tau", "qs_tau", "t_ice"]:
        summary["de_analysis"][skey] = {
            "values": de_analysis[skey]["values"],
            "orders": [x if not np.isnan(x) else None
                       for x in de_analysis[skey]["orders"]],
            "p_hat": de_analysis[skey]["p_hat"] if not np.isnan(
                de_analysis[skey]["p_hat"]) else None,
            "f_exact": de_analysis[skey]["f_exact"] if not np.isnan(
                de_analysis[skey]["f_exact"]) else None,
            "DE_fine": de_analysis[skey]["DE_fine"] if not np.isnan(
                de_analysis[skey]["DE_fine"]) else None,
            "GCI_fine": de_analysis[skey]["GCI_fine"] if not np.isnan(
                de_analysis[skey]["GCI_fine"]) else None,
            "Fs": de_analysis[skey]["Fs"],
        }
        summary["it_analysis"][skey] = {
            "tolerances": it_analysis[skey]["tolerances"],
            "errors": it_analysis[skey]["errors"],
        }
        summary["ro_analysis"][skey] = ro_analysis[skey]
        summary["total_uncertainty"][skey] = {
            "fine": total[skey]["fine"],
            "param": total[skey]["param"],
            "DE_sign": total[skey]["DE_sign"] if not np.isnan(
                total[skey]["DE_sign"]) else None,
            "p_hat": total[skey]["p_hat"] if not np.isnan(
                total[skey]["p_hat"]) else None,
        }

    with open(f"{OUT_DIR}/hw4_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Summary exported to {OUT_DIR}/hw4_summary.json")
    print("\n" + "=" * 70)
    print("HW4 SOLUTION VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()