#!/usr/bin/env python3
"""
===============================================================================
VVSC Homework 5: Validation Metric Computation (AVM / MAVM)
===============================================================================
Application:  Black-Ice Risk Prediction - 1D Transient Heat Conduction
Course:       AOE/CS/ME 6444 - Verification and Validation in Scientific Computing
Author:       Dawood Wasif
===============================================================================

Physical scenario (same as HW4): Clear winter night in Blacksburg, VA
  - 6-hour simulation (sunset to midnight)
  - Asphalt road column, depth 0.5 m
  - Nonlinear surface-energy-balance Neumann BC, Newton iteration

Aleatory uncertain model input:
  - V_wind ~ Normal(mu = 2.5 m/s, sigma = 1.0 m/s)
    (nighttime 10-m wind has ~40% CV in operational RWIS data)

Synthetic experimental data: Option #2 (per HW5 specification)
  - Dataset 1 (N = 5):  chi_1 = [0.55, 0.95, 1.0, 1.1, 1.5]
  - Dataset 2 (N = 10): chi_2 = [0.1, 0.4, 0.6, 0.75, 0.8, 0.9, 0.91, 0.97, 1.3, 1.6]
  - SRQ_syn = alpha + chi * (beta - alpha), with alpha/beta from V = mu +/- sigma

SRQs tracked:
  1. t_ice   [s]    - ice-onset time  (PRIMARY - safety-critical)
  2. Ts_tau  [K]    - surface temperature at tau (secondary)
  3. qs_tau  [W/m2] - surface heat flux at tau (secondary)

Validation metrics:
  - AVM  (Ferson et al. 2008; Oberkampf & Roy 2nd ed. Ch. 12)
  - MAVM (Voyles & Roy 2015) - gives asymmetric (d+, d-) bounds

Sample sizes: n = 10, 25, 100  (plus n = 500 reference for convergence study)
Sampling: Latin Hypercube Sampling from N(mu, sigma^2)

Computational mesh: parametric-study grid from HW4
  - Nz = 32, Nt = 256  (U_NUM(t_ice) ~ 0.76 s << expected aleatory spread)
===============================================================================
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import json, os, time
from scipy.stats import qmc, norm

from vvsc_heat1d_solver import solve_heat_1d_CN_neumann

# =============================================================================
# 1. Physical scenario (frozen from HW4)
# =============================================================================
RHO, CP, K_MAT = 2100.0, 920.0, 1.2
L, TAU         = 0.5, 21600.0
EPSILON        = 0.93
SIGMA_SB       = 5.670374419e-8
EPSILON_ATM    = 0.74
T_AIR_START    = 275.0
T_AIR_END      = 268.0
T_DEEP         = 283.0
T_SURFACE_INIT = 278.0
T_FREEZE       = 273.15

# Parametric-study mesh from HW4 (U_NUM(t_ice) ~ 0.76 s)
NZ, NT       = 32, 256
NEWTON_TOL   = 1e-10            # tight enough that iterative error is negligible

# Aleatory uncertain input: wind speed
V_WIND_MU    = 2.5              # [m/s]
V_WIND_SIGMA = 1.0              # [m/s]   sigma/mu = 0.4 (physically reasonable for nighttime wind)

# Assignment-specified sample sizes (plus a reference at n=500)
SAMPLE_SIZES = [10, 25, 100]
N_REF        = 500

# Synthetic data chi vectors (as specified in HW5 prompt)
CHI_1 = np.array([0.55, 0.95, 1.00, 1.10, 1.50])
CHI_2 = np.array([0.10, 0.40, 0.60, 0.75, 0.80, 0.90, 0.91, 0.97, 1.30, 1.60])

OUT_DIR = "hw5_output"
os.makedirs(OUT_DIR, exist_ok=True)

# Reproducibility
RNG_SEED = 2026


# =============================================================================
# 2. Forward model: wrap the HW4 solver with V_wind as the varying input
# =============================================================================
def T_air(t):
    return T_AIR_START + (T_AIR_END - T_AIR_START) * (t / TAU)


def T_init(z):
    return T_SURFACE_INIT + (T_DEEP - T_SURFACE_INIT) * (z / L)


def T_right(t):
    return T_DEEP


def run_model(V_wind, Nz=NZ, Nt=NT, newton_tol=NEWTON_TOL):
    """Run the 1D thermal solver for a given wind speed and return the 3 SRQs."""
    h_conv = 5.7 + 3.8 * V_wind       # Palyvos (2008) wind-speed correlation

    def q_net(Ts, t):
        Ta = T_air(t)
        q_lw   = EPSILON * SIGMA_SB * (EPSILON_ATM * Ta**4 - Ts**4)
        q_conv = -h_conv * (Ts - Ta)
        return q_lw + q_conv

    def dq_net_dTs(Ts, t):
        return -4.0 * EPSILON * SIGMA_SB * Ts**3 - h_conv

    res = solve_heat_1d_CN_neumann(
        Nz, Nt, L, TAU, RHO, CP, K_MAT,
        T_init, T_right, q_net, dq_net_dTs,
        newton_tol=newton_tol, newton_maxiter=100, dtype=np.float64,
    )

    T_final = res["T_final"]
    dz      = res["dz"]

    # SRQ1: surface temperature at tau
    Ts_tau = float(T_final[0])

    # SRQ2: surface heat flux at tau (2nd-order one-sided FD)
    dTdz   = (-3.0 * T_final[0] + 4.0 * T_final[1] - T_final[2]) / (2.0 * dz)
    qs_tau = float(-K_MAT * dTdz)

    # SRQ3: ice-onset time (linear interpolation of T_surface_history crossing)
    Ts_hist = res["T_surface_history"]
    t_arr   = res["t"]
    t_ice   = np.nan
    for i in range(1, len(Ts_hist)):
        if Ts_hist[i] < T_FREEZE and Ts_hist[i - 1] >= T_FREEZE:
            frac  = (T_FREEZE - Ts_hist[i - 1]) / (Ts_hist[i] - Ts_hist[i - 1])
            t_ice = float(t_arr[i - 1] + frac * (t_arr[i] - t_arr[i - 1]))
            break

    return {"Ts_tau": Ts_tau, "qs_tau": qs_tau, "t_ice": t_ice}


# =============================================================================
# 3. LHS sampling from a 1D Normal distribution
# =============================================================================
def lhs_normal(mu, sigma, n, seed=0):
    """Latin Hypercube sample of size n from N(mu, sigma^2).

    Uses scipy.stats.qmc.LatinHypercube to draw stratified uniform samples,
    then maps through the inverse Normal CDF.
    """
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n).flatten()
    return norm.ppf(u, loc=mu, scale=sigma)


# =============================================================================
# 4. AVM and MAVM via exact step-function integration
# =============================================================================
def ecdf_stepwise(samples):
    """Return (x, F) such that the empirical CDF has jumps at x with values F.
    F = k / n at the k-th sorted sample.
    """
    s = np.sort(np.asarray(samples, dtype=float))
    n = s.size
    F = np.arange(1, n + 1) / n
    return s, F


def avm_mavm(sim_samples, exp_samples):
    """Exact AVM and MAVM via step-function integration.

    Convention used here (consistent with Voyles & Roy 2015 and
    Oberkampf & Roy 2nd ed., Ch. 12):

        d_plus  = integral of max(F_sim(y) - F_exp(y), 0) dy
                  [F_sim above F_exp => model UNDERpredicts y
                   => positive correction on prediction]

        d_minus = integral of max(F_exp(y) - F_sim(y), 0) dy
                  [F_sim below F_exp => model OVERpredicts y
                   => negative correction on prediction]

        AVM d = d_plus + d_minus

    Both CDFs are right-continuous step functions; integration is exact.
    """
    sim = np.sort(np.asarray(sim_samples, dtype=float))
    exp = np.sort(np.asarray(exp_samples, dtype=float))
    n_s, n_e = sim.size, exp.size

    # Union of all jump points; pad with endpoints just beyond the data
    y_min = min(sim.min(), exp.min())
    y_max = max(sim.max(), exp.max())
    span  = y_max - y_min
    pad   = max(1e-12, 0.01 * span)
    y_all = np.unique(np.concatenate([[y_min - pad],
                                      sim, exp,
                                      [y_max + pad]]))

    # Evaluate right-continuous CDFs at each y_all:  F(y) = #{x_i <= y} / n
    F_sim = np.searchsorted(sim, y_all, side="right") / n_s
    F_exp = np.searchsorted(exp, y_all, side="right") / n_e

    # Integrate using piecewise-constant values on (y_k, y_{k+1}]
    widths = np.diff(y_all)
    diff   = F_sim[:-1] - F_exp[:-1]       # value on the open interval
    d_plus  = float(np.sum(np.maximum(diff,  0.0) * widths))
    d_minus = float(np.sum(np.maximum(-diff, 0.0) * widths))
    d_total = d_plus + d_minus

    return {"AVM": d_total, "d_plus": d_plus, "d_minus": d_minus}


# =============================================================================
# 5. Build synthetic experimental data at the two specified sample sizes
# =============================================================================
def build_synthetic_data(srq_at_mu_minus_sigma, srq_at_mu_plus_sigma, chi):
    """Synthetic experimental data per HW5 spec:  SRQ = alpha + chi * (beta - alpha)."""
    alpha = min(srq_at_mu_minus_sigma, srq_at_mu_plus_sigma)
    beta  = max(srq_at_mu_minus_sigma, srq_at_mu_plus_sigma)
    return alpha + chi * (beta - alpha), alpha, beta


# =============================================================================
# 6. Driver
# =============================================================================
def main():
    t_start = time.time()
    print("=" * 72)
    print("VVSC Homework 5: Validation Metric Computation")
    print("=" * 72)

    SRQ_LIST = ["t_ice", "Ts_tau", "qs_tau"]
    SRQ_UNITS = {"t_ice": "s", "Ts_tau": "K", "qs_tau": "W/m^2"}
    SRQ_LABEL = {"t_ice": r"$t_{ice}$", "Ts_tau": r"$T_s(\tau)$",
                 "qs_tau": r"$q_s(\tau)$"}

    # -------------------------------------------------------------------
    # 6.1  Generate synthetic experimental data from V = mu +/- sigma
    # -------------------------------------------------------------------
    print("\n[6.1] Running model at V = mu - sigma and V = mu + sigma "
          "for synthetic-data endpoints ...")
    V_lo  = V_WIND_MU - V_WIND_SIGMA
    V_hi  = V_WIND_MU + V_WIND_SIGMA
    srq_lo = run_model(V_lo)
    srq_hi = run_model(V_hi)
    srq_nom = run_model(V_WIND_MU)
    print(f"   V = {V_lo:.2f} m/s  -> "
          f"Ts={srq_lo['Ts_tau']:.4f} K, qs={srq_lo['qs_tau']:.3f} W/m2, "
          f"t_ice={srq_lo['t_ice']:.2f} s")
    print(f"   V = {V_hi:.2f} m/s  -> "
          f"Ts={srq_hi['Ts_tau']:.4f} K, qs={srq_hi['qs_tau']:.3f} W/m2, "
          f"t_ice={srq_hi['t_ice']:.2f} s")
    print(f"   V = {V_WIND_MU:.2f} m/s  -> "
          f"Ts={srq_nom['Ts_tau']:.4f} K, qs={srq_nom['qs_tau']:.3f} W/m2, "
          f"t_ice={srq_nom['t_ice']:.2f} s   (nominal)")

    synthetic = {}
    for srq in SRQ_LIST:
        D1, a1, b1 = build_synthetic_data(srq_lo[srq], srq_hi[srq], CHI_1)
        D2, a2, b2 = build_synthetic_data(srq_lo[srq], srq_hi[srq], CHI_2)
        synthetic[srq] = {
            "alpha": float(a1),  "beta": float(b1),
            "D1": D1.tolist(),   "D2": D2.tolist(),
            "D1_mean": float(D1.mean()), "D1_std": float(D1.std(ddof=1)),
            "D2_mean": float(D2.mean()), "D2_std": float(D2.std(ddof=1)),
        }

    print("\n[6.1] Synthetic data endpoints (alpha, beta):")
    for srq in SRQ_LIST:
        a, b = synthetic[srq]["alpha"], synthetic[srq]["beta"]
        print(f"   {srq:8s}: alpha = {a:12.4f}, beta = {b:12.4f}, "
              f"(beta - alpha) = {b - a:12.4f} {SRQ_UNITS[srq]}")

    # -------------------------------------------------------------------
    # 6.2  LHS sampling and model propagation
    # -------------------------------------------------------------------
    print("\n[6.2] Latin Hypercube Sampling and model propagation ...")
    lhs_samples = {}
    sim_srq     = {}
    for n in SAMPLE_SIZES + [N_REF]:
        V_samples = lhs_normal(V_WIND_MU, V_WIND_SIGMA, n, seed=RNG_SEED + n)
        # Guard against (very unlikely) negative wind speeds from the tails
        V_samples = np.maximum(V_samples, 0.05)
        lhs_samples[n] = V_samples

        srqs = {s: np.zeros(n) for s in SRQ_LIST}
        t0 = time.time()
        for i, V in enumerate(V_samples):
            out = run_model(float(V))
            for s in SRQ_LIST:
                srqs[s][i] = out[s]
        dt = time.time() - t0
        sim_srq[n] = srqs
        print(f"   n = {n:4d} :  {dt:6.2f} s  "
              f"({dt/n*1000:5.1f} ms/run)  "
              f"t_ice range [{srqs['t_ice'].min():.1f}, "
              f"{srqs['t_ice'].max():.1f}] s")

    # -------------------------------------------------------------------
    # 6.3  AVM and MAVM for each (SRQ, dataset, sample size)
    # -------------------------------------------------------------------
    print("\n[6.3] Computing AVM and MAVM ...")
    metrics = {s: {"D1": {}, "D2": {}} for s in SRQ_LIST}
    for srq in SRQ_LIST:
        for dset_name, chi in [("D1", CHI_1), ("D2", CHI_2)]:
            exp_data, _, _ = build_synthetic_data(
                srq_lo[srq], srq_hi[srq], chi)
            for n in SAMPLE_SIZES + [N_REF]:
                m = avm_mavm(sim_srq[n][srq], exp_data)
                metrics[srq][dset_name][n] = m

    # Print a neat table for the primary SRQ
    print("\n   --- PRIMARY SRQ: t_ice [s] ---")
    print(f"   {'n':>5}  {'dataset':>7}  {'AVM':>10}  "
          f"{'d+':>10}  {'d-':>10}")
    for n in SAMPLE_SIZES + [N_REF]:
        for dset in ["D1", "D2"]:
            m = metrics["t_ice"][dset][n]
            print(f"   {n:5d}  {dset:>7s}  {m['AVM']:10.3f}  "
                  f"{m['d_plus']:10.3f}  {m['d_minus']:10.3f}")

    # -------------------------------------------------------------------
    # 6.4  Save all results
    # -------------------------------------------------------------------
    summary = {
        "scenario": {
            "description": "Clear winter night in Blacksburg, VA; 6 hr nocturnal cooling",
            "rho": RHO, "cp": CP, "k": K_MAT, "L": L, "tau": TAU,
            "epsilon": EPSILON, "epsilon_atm": EPSILON_ATM,
            "T_air_start": T_AIR_START, "T_air_end": T_AIR_END,
            "T_deep": T_DEEP, "T_surface_init": T_SURFACE_INIT,
            "T_freeze": T_FREEZE,
            "mesh": {"Nz": NZ, "Nt": NT, "newton_tol": NEWTON_TOL},
        },
        "aleatory_input": {
            "name": "V_wind", "units": "m/s",
            "distribution": "Normal",
            "mu": V_WIND_MU, "sigma": V_WIND_SIGMA,
            "CV": V_WIND_SIGMA / V_WIND_MU,
            "justification": (
                "Nighttime 10-m wind speed has CV ~ 0.3-0.5 in operational "
                "RWIS data; 0.4 is a representative value."),
        },
        "chi_vectors": {"D1": CHI_1.tolist(), "D2": CHI_2.tolist()},
        "srq_endpoints": {
            "V_mu_minus_sigma": {k: srq_lo[k] for k in SRQ_LIST},
            "V_mu_plus_sigma":  {k: srq_hi[k] for k in SRQ_LIST},
            "V_mu_nominal":     {k: srq_nom[k] for k in SRQ_LIST},
            "V_mu_minus_sigma_value": V_lo,
            "V_mu_plus_sigma_value":  V_hi,
        },
        "synthetic_data": synthetic,
        "lhs_samples": {str(n): V.tolist() for n, V in lhs_samples.items()},
        "sim_srq":     {str(n): {s: sim_srq[n][s].tolist() for s in SRQ_LIST}
                        for n in SAMPLE_SIZES + [N_REF]},
        "metrics":     {s: {d: {str(n): metrics[s][d][n]
                                for n in SAMPLE_SIZES + [N_REF]}
                            for d in ["D1", "D2"]}
                        for s in SRQ_LIST},
        "sample_sizes": SAMPLE_SIZES,
        "n_reference":  N_REF,
        "runtime_s":    time.time() - t_start,
    }
    with open(f"{OUT_DIR}/hw5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n   Summary written to {OUT_DIR}/hw5_summary.json")

    # -------------------------------------------------------------------
    # 6.5  Figures
    # -------------------------------------------------------------------
    print("\n[6.5] Generating figures ...")
    make_all_figures(summary, sim_srq, lhs_samples, metrics,
                     srq_lo, srq_hi, srq_nom)

    print(f"\nTotal runtime: {time.time() - t_start:.1f} s")
    print(f"All outputs in {OUT_DIR}/")
    return summary


# =============================================================================
# 7. Plotting
# =============================================================================
def make_all_figures(summary, sim_srq, lhs_samples, metrics,
                     srq_lo, srq_hi, srq_nom):
    SRQ_LIST = ["t_ice", "Ts_tau", "qs_tau"]
    SRQ_UNITS = {"t_ice": "s", "Ts_tau": "K", "qs_tau": "W/m$^2$"}
    SRQ_LABEL = {"t_ice": r"$t_{\mathrm{ice}}$",
                 "Ts_tau": r"$T_s(\tau)$", "qs_tau": r"$q_s(\tau)$"}

    # Common style
    plt.rcParams.update({"font.size": 11, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 140})

    # ---------------------------------------------------------------
    # Fig 1: aleatory input PDF and V_wind -> SRQ mapping
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))

    # Left: PDF of V_wind with mu, mu+/-sigma marked
    V_grid = np.linspace(V_WIND_MU - 4 * V_WIND_SIGMA,
                         V_WIND_MU + 4 * V_WIND_SIGMA, 400)
    pdf = norm.pdf(V_grid, V_WIND_MU, V_WIND_SIGMA)
    axes[0].fill_between(V_grid, pdf, alpha=0.25, color="tab:blue")
    axes[0].plot(V_grid, pdf, color="tab:blue", lw=2)
    for V, lbl, c in [(V_WIND_MU - V_WIND_SIGMA, r"$\mu-\sigma$", "tab:red"),
                      (V_WIND_MU,               r"$\mu$",        "k"),
                      (V_WIND_MU + V_WIND_SIGMA, r"$\mu+\sigma$", "tab:red")]:
        axes[0].axvline(V, color=c, ls="--", lw=1.2)
        axes[0].text(V, pdf.max() * 1.05, lbl, ha="center",
                     fontsize=10, color=c)
    axes[0].set_xlabel(r"Wind speed $V_{\mathrm{wind}}$ [m/s]")
    axes[0].set_ylabel("Probability density")
    axes[0].set_title(rf"Aleatory input: $V_{{\mathrm{{wind}}}} \sim "
                      rf"\mathcal{{N}}({V_WIND_MU},\,{V_WIND_SIGMA}^2)$")

    # Right: V_wind -> t_ice mapping with alpha, beta endpoints
    V_scan = np.linspace(0.5, 5.0, 19)
    t_ice_scan = np.array([run_model(float(V))["t_ice"] for V in V_scan])
    axes[1].plot(V_scan, t_ice_scan, "o-", color="tab:purple", lw=1.6,
                 ms=5, label=r"model $V \mapsto t_{\mathrm{ice}}$")
    axes[1].axvline(V_WIND_MU - V_WIND_SIGMA, color="tab:red",
                    ls="--", lw=1, alpha=0.7)
    axes[1].axvline(V_WIND_MU + V_WIND_SIGMA, color="tab:red",
                    ls="--", lw=1, alpha=0.7)
    axes[1].axvline(V_WIND_MU, color="k", ls="--", lw=1, alpha=0.7)
    axes[1].scatter([V_WIND_MU - V_WIND_SIGMA, V_WIND_MU + V_WIND_SIGMA],
                    [srq_lo["t_ice"], srq_hi["t_ice"]],
                    s=90, color="tab:red", zorder=5,
                    label=r"synthetic-data endpoints")
    axes[1].set_xlabel(r"Wind speed $V_{\mathrm{wind}}$ [m/s]")
    axes[1].set_ylabel(r"$t_{\mathrm{ice}}$ [s]")
    axes[1].set_title(r"Model mapping $V_{\mathrm{wind}} \mapsto t_{\mathrm{ice}}$")
    axes[1].legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig01_aleatory_setup.png", bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------
    # Fig 2: LHS samples of V_wind for all n
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, len(SAMPLE_SIZES), figsize=(13, 3.6),
                             sharex=True, sharey=True)
    V_grid2 = np.linspace(V_WIND_MU - 4 * V_WIND_SIGMA,
                          V_WIND_MU + 4 * V_WIND_SIGMA, 400)
    pdf2 = norm.pdf(V_grid2, V_WIND_MU, V_WIND_SIGMA)
    for ax, n in zip(axes, SAMPLE_SIZES):
        ax.fill_between(V_grid2, pdf2, alpha=0.2, color="tab:blue")
        ax.plot(V_grid2, pdf2, color="tab:blue", lw=1.5)
        Vn = lhs_samples[n]
        ax.scatter(Vn, np.full_like(Vn, pdf2.max() * 0.05),
                   marker="|", s=120, color="tab:red", lw=1.5,
                   label=f"LHS n = {n}")
        ax.set_xlabel(r"$V_{\mathrm{wind}}$ [m/s]")
        ax.set_title(f"n = {n}")
        ax.legend(loc="upper right", fontsize=9)
    axes[0].set_ylabel("Probability density")
    plt.suptitle(r"LHS samples of $V_{\mathrm{wind}} \sim "
                 rf"\mathcal{{N}}({V_WIND_MU},\,{V_WIND_SIGMA}^2)$",
                 y=1.02)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig02_lhs_samples.png", bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------
    # Fig 3: synthetic experimental data (D1, D2) for all 3 SRQs
    # ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8))
    for ax, srq in zip(axes, SRQ_LIST):
        D1 = np.array(summary["synthetic_data"][srq]["D1"])
        D2 = np.array(summary["synthetic_data"][srq]["D2"])
        a  = summary["synthetic_data"][srq]["alpha"]
        b  = summary["synthetic_data"][srq]["beta"]
        ax.axvspan(a, b, color="tab:gray", alpha=0.15,
                   label=r"$[\alpha, \beta]$")
        ax.axvline(a, color="tab:gray", ls=":", lw=1)
        ax.axvline(b, color="tab:gray", ls=":", lw=1)
        ax.scatter(D1, np.ones_like(D1) * 1.0, color="tab:red",
                   s=70, marker="o", label=f"D1 (N=5)")
        ax.scatter(D2, np.ones_like(D2) * 0.5, color="tab:blue",
                   s=70, marker="s", label=f"D2 (N=10)")
        ax.set_ylim(0.0, 1.5)
        ax.set_yticks([])
        ax.set_xlabel(f"{SRQ_LABEL[srq]} [{SRQ_UNITS[srq]}]")
        ax.set_title(f"Synthetic experimental data  ({SRQ_LABEL[srq]})")
        ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig03_synthetic_data.png", bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------
    # Fig 4 (primary): CDF-comparison AVM panel for t_ice
    #   rows = datasets D1, D2;  cols = sample sizes 10, 25, 100
    # ---------------------------------------------------------------
    make_cdf_grid(summary, sim_srq, "t_ice", "AVM",
                  f"{OUT_DIR}/fig04_avm_tice.png",
                  r"$t_{\mathrm{ice}}$ [s]")

    # ---------------------------------------------------------------
    # Fig 5 (primary): CDF-comparison MAVM panel for t_ice (d+ / d-)
    # ---------------------------------------------------------------
    make_cdf_grid(summary, sim_srq, "t_ice", "MAVM",
                  f"{OUT_DIR}/fig05_mavm_tice.png",
                  r"$t_{\mathrm{ice}}$ [s]")

    # ---------------------------------------------------------------
    # Fig 6: convergence of AVM and MAVM with sample size  (all SRQs)
    # ---------------------------------------------------------------
    # Compute metrics at MANY intermediate n to show convergence clearly
    n_conv = [10, 15, 20, 25, 35, 50, 75, 100, 150, 200, 300, 500]
    print("   Convergence study: extra LHS runs ...")
    extra_sim = {}
    for n in n_conv:
        if n in sim_srq:
            extra_sim[n] = sim_srq[n]
            continue
        Vn = lhs_normal(V_WIND_MU, V_WIND_SIGMA, n, seed=RNG_SEED + n)
        Vn = np.maximum(Vn, 0.05)
        srqs = {s: np.zeros(n) for s in SRQ_LIST}
        for i, V in enumerate(Vn):
            out = run_model(float(V))
            for s in SRQ_LIST:
                srqs[s][i] = out[s]
        extra_sim[n] = srqs

    conv_metrics = {s: {"D1": {"AVM": [], "d+": [], "d-": []},
                        "D2": {"AVM": [], "d+": [], "d-": []}}
                    for s in SRQ_LIST}
    for srq in SRQ_LIST:
        for dset, chi in [("D1", CHI_1), ("D2", CHI_2)]:
            exp_d, _, _ = build_synthetic_data(
                srq_lo[srq], srq_hi[srq], chi)
            for n in n_conv:
                m = avm_mavm(extra_sim[n][srq], exp_d)
                conv_metrics[srq][dset]["AVM"].append(m["AVM"])
                conv_metrics[srq][dset]["d+"].append(m["d_plus"])
                conv_metrics[srq][dset]["d-"].append(m["d_minus"])

    fig, axes = plt.subplots(2, 3, figsize=(14, 7.5), sharex=True)
    for col, srq in enumerate(SRQ_LIST):
        for row, dset in enumerate(["D1", "D2"]):
            ax = axes[row, col]
            ax.semilogx(n_conv, conv_metrics[srq][dset]["AVM"],
                        "o-", color="k", lw=1.8, ms=5,
                        label="AVM (=$d^+ + d^-$)")
            ax.semilogx(n_conv, conv_metrics[srq][dset]["d+"],
                        "^-", color="tab:red", lw=1.4, ms=5,
                        label=r"$d^+$ (model underpredicts)")
            ax.semilogx(n_conv, conv_metrics[srq][dset]["d-"],
                        "v-", color="tab:blue", lw=1.4, ms=5,
                        label=r"$d^-$ (model overpredicts)")
            for n in SAMPLE_SIZES:
                ax.axvline(n, color="tab:gray", ls=":", lw=0.8, alpha=0.7)
            ax.set_title(f"{SRQ_LABEL[srq]}, dataset {dset}")
            if row == 1:
                ax.set_xlabel("LHS sample size $n$")
            if col == 0:
                ax.set_ylabel(f"Validation metric [{SRQ_UNITS[srq]}]")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig06_convergence.png", bbox_inches="tight")
    plt.close()

    # Save convergence data too
    with open(f"{OUT_DIR}/hw5_convergence.json", "w") as f:
        json.dump({"n_grid": n_conv, "metrics": conv_metrics}, f, indent=2)

    # ---------------------------------------------------------------
    # Fig 7: MAVM uncertainty bands on mean prediction (across n)
    #        (primary SRQ t_ice, D2 dataset)
    # ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    means = [np.mean(extra_sim[n]["t_ice"]) for n in n_conv]
    d_plus  = np.array(conv_metrics["t_ice"]["D2"]["d+"])
    d_minus = np.array(conv_metrics["t_ice"]["D2"]["d-"])
    ax.plot(n_conv, means, "o-", color="tab:purple", lw=1.8,
            ms=5, label=r"$\overline{y}_{\mathrm{sim}}$ (LHS mean)")
    ax.fill_between(n_conv, np.array(means) - d_minus,
                    np.array(means) + d_plus,
                    color="tab:orange", alpha=0.35,
                    label=r"MAVM band $[\overline{y}-d^-,\ \overline{y}+d^+]$")
    # Experimental mean (D2) as reference
    D2_mean = summary["synthetic_data"]["t_ice"]["D2_mean"]
    ax.axhline(D2_mean, color="tab:red", ls="--", lw=1.4,
               label=r"experimental mean $\overline{y}_{\mathrm{exp}}$ (D2)")
    ax.set_xscale("log")
    ax.set_xlabel("LHS sample size $n$")
    ax.set_ylabel(r"$t_{\mathrm{ice}}$ [s]")
    ax.set_title(r"Asymmetric MAVM band around LHS mean prediction "
                 r"($t_{\mathrm{ice}}$, D2)")
    for n in SAMPLE_SIZES:
        ax.axvline(n, color="tab:gray", ls=":", lw=0.8, alpha=0.7)
    ax.legend(loc="best", fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig07_mavm_band.png", bbox_inches="tight")
    plt.close()

    # ---------------------------------------------------------------
    # Fig 8: supplementary CDF grids for Ts and qs (primary vs primary)
    # ---------------------------------------------------------------
    make_cdf_grid(summary, sim_srq, "Ts_tau", "MAVM",
                  f"{OUT_DIR}/fig08_mavm_Ts.png",
                  r"$T_s(\tau)$ [K]")
    make_cdf_grid(summary, sim_srq, "qs_tau", "MAVM",
                  f"{OUT_DIR}/fig09_mavm_qs.png",
                  r"$q_s(\tau)$ [W/m$^2$]")

    print(f"   Figures saved to {OUT_DIR}/")


def make_cdf_grid(summary, sim_srq, srq, metric_kind, outpath, srq_label):
    """2 x 3 grid: rows = D1, D2;  cols = n = 10, 25, 100.

    metric_kind: "AVM" shades the total area;
                 "MAVM" shades d+ and d- separately in different colors.
    """
    fig, axes = plt.subplots(2, len(SAMPLE_SIZES), figsize=(14, 7.3),
                             sharex=True)
    for row, dset_name in enumerate(["D1", "D2"]):
        exp_data = np.array(summary["synthetic_data"][srq][dset_name])
        chi = CHI_1 if dset_name == "D1" else CHI_2
        for col, n in enumerate(SAMPLE_SIZES):
            ax = axes[row, col]
            sim = np.sort(sim_srq[n][srq])
            exp = np.sort(exp_data)

            # Build a fine y-grid covering both
            y_min = min(sim.min(), exp.min())
            y_max = max(sim.max(), exp.max())
            pad = 0.03 * (y_max - y_min)
            y_grid = np.linspace(y_min - pad, y_max + pad, 600)
            F_sim = np.searchsorted(sim, y_grid, side="right") / sim.size
            F_exp = np.searchsorted(exp, y_grid, side="right") / exp.size

            # Shade AVM or MAVM regions
            if metric_kind == "AVM":
                ax.fill_between(y_grid, F_sim, F_exp,
                                color="tab:purple", alpha=0.35,
                                label="AVM area")
            else:
                # d+: F_sim > F_exp (model underpredicts)
                ax.fill_between(y_grid, F_sim, F_exp,
                                where=(F_sim >= F_exp),
                                color="tab:red", alpha=0.38,
                                label=r"$d^+$ (underpred.)")
                # d-: F_sim < F_exp (model overpredicts)
                ax.fill_between(y_grid, F_sim, F_exp,
                                where=(F_sim < F_exp),
                                color="tab:blue", alpha=0.38,
                                label=r"$d^-$ (overpred.)")

            ax.plot(y_grid, F_sim, color="tab:green", lw=2,
                    label=r"$F_{\mathrm{sim}}$")
            ax.plot(y_grid, F_exp, color="tab:orange", lw=2,
                    label=r"$S_n$ (exp)")

            # Annotation
            m = avm_mavm(sim, exp)
            if metric_kind == "AVM":
                text = rf"AVM $= {m['AVM']:.3g}$"
            else:
                text = (rf"$d^+ = {m['d_plus']:.3g}$" + "\n"
                        rf"$d^- = {m['d_minus']:.3g}$" + "\n"
                        rf"AVM $= {m['AVM']:.3g}$")
            ax.text(0.04, 0.92, text, transform=ax.transAxes,
                    fontsize=10, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", edgecolor="0.7",
                              alpha=0.92))

            ax.set_ylim(-0.02, 1.05)
            ax.set_title(f"{dset_name}, n = {n}")
            if row == 1:
                ax.set_xlabel(srq_label)
            if col == 0:
                ax.set_ylabel("CDF")
            if row == 0 and col == 0:
                ax.legend(loc="lower right", fontsize=8)
    plt.suptitle(f"{metric_kind}: simulation CDF vs. experimental EDF",
                 y=1.00, fontsize=12)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()