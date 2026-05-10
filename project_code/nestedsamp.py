"""
Project requirement — uncertainty propagation via nested sampling.

Outer loop: equal-partition sampling on the epistemic interval
            eps_atm in [0.70, 0.78] with Ne+1 endpoint samples.
Inner loop: Latin Hypercube Sampling of V_wind ~ N(2.5, 1.0^2) (n inner = 50).

Sweep Ne in {5, 10, 25}, build the p-box envelope at the prediction
condition T_air_end = 263 K.  Compare to a probabilistic counterfactual
with eps_atm ~ Uniform[0.70, 0.78] (joint LHS over both inputs).
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc, norm

from solver import PhysicalScenario, solve_physical


MU_V = 2.5
SIG_V = 1.0
EPS_LO, EPS_HI = 0.70, 0.78
T_AIR_END_PRED = 263.0
N_INNER = 50
NE_LIST = [5, 10, 25]


def lhs_V_wind(n: int, seed: int = 0) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    return norm.ppf(sampler.random(n=n).ravel(), loc=MU_V, scale=SIG_V)


def joint_lhs(n: int, seed: int = 1) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=2, seed=seed)
    u = sampler.random(n=n)
    V = norm.ppf(u[:, 0], loc=MU_V, scale=SIG_V)
    eps = EPS_LO + (EPS_HI - EPS_LO) * u[:, 1]
    return V, eps


def t_ice_at(V: float, eps: float, T_end: float = T_AIR_END_PRED,
             Nz: int = 32, Nt: int = 256) -> float:
    return solve_physical(Nz, Nt, PhysicalScenario(V_wind=float(V),
                                                    eps_atm=float(eps),
                                                    T_air_end=T_end))["t_ice"]


def run_nested(Ne: int, seed: int = 42):
    """Outer: Ne+1 endpoint samples on [EPS_LO, EPS_HI].  Inner: N_INNER LHS over V."""
    eps_endpoints = np.linspace(EPS_LO, EPS_HI, Ne + 1)
    V_arr = lhs_V_wind(N_INNER, seed=seed)
    cdfs = []  # list of sorted samples per outer fix
    for eps in eps_endpoints:
        samples = np.array([t_ice_at(V, eps) for V in V_arr])
        cdfs.append(np.sort(samples))
    cdfs = np.array(cdfs)  # (Ne+1, N_INNER)
    return eps_endpoints, V_arr, cdfs


def pbox_envelope(cdfs: np.ndarray, n_grid: int = 500):
    """Pointwise envelope across the family of CDFs at common probability levels.

    For each cumulative-probability level, the lower bound is the minimum
    value of t_ice across the family at that level; the upper bound is the
    maximum.  Equivalently, project each CDF onto a common probability axis.
    """
    # cdfs[i] is a sorted N_INNER vector of t_ice samples for outer-loop sample i.
    # Empirical CDF of cdfs[i] takes value k/N_INNER at the k-th sorted sample.
    Ne_plus_1, N = cdfs.shape
    p_levels = (np.arange(1, N + 1)) / N  # the discrete CDF heights
    # Each row of `cdfs` already gives x-values at these p-levels.  To get
    # left/right envelope at each p-level, take min/max across rows.
    x_lower = cdfs.min(axis=0)  # leftmost CDF — most aggressive cooling
    x_upper = cdfs.max(axis=0)  # rightmost CDF — slowest cooling
    return p_levels, x_lower, x_upper, cdfs


def median_pbox_width(p_levels, x_lower, x_upper):
    """Width of p-box at p=0.5 (linear interpolation in p)."""
    p_med = 0.5
    x_lo_med = float(np.interp(p_med, p_levels, x_lower))
    x_up_med = float(np.interp(p_med, p_levels, x_upper))
    return x_up_med - x_lo_med, x_lo_med, x_up_med


def main(fig_dir: str = "../figs", results_dir: str = "../results"):
    print("=== Nested-sampling p-box at T_air,end = 263 K ===")
    pbox_results = {}
    for Ne in NE_LIST:
        eps_endpts, V_arr, cdfs = run_nested(Ne)
        p_levels, x_lo, x_up, _ = pbox_envelope(cdfs)
        width, x_lo_med, x_up_med = median_pbox_width(p_levels, x_lo, x_up)
        pbox_results[Ne] = {
            "eps_endpoints": eps_endpts.tolist(),
            "cdfs": cdfs.tolist(),
            "p_levels": p_levels.tolist(),
            "x_lower": x_lo.tolist(),
            "x_upper": x_up.tolist(),
            "median_width": width,
            "x_lo_med": x_lo_med,
            "x_up_med": x_up_med,
        }
        print(f"  Ne = {Ne:3d}  median p-box width = {width:7.2f} s  "
              f"(p=0.5: [{x_lo_med:.1f}, {x_up_med:.1f}])")

    # ----- Probabilistic counterfactual -----
    print("\n=== Probabilistic-uniform counterfactual ===")
    N_uniform = 1000
    V_u, eps_u = joint_lhs(N_uniform, seed=1)
    t_uniform = np.array([t_ice_at(V, eps) for V, eps in zip(V_u, eps_u)])
    t_uniform_sorted = np.sort(t_uniform)
    p_uniform = (np.arange(1, N_uniform + 1)) / N_uniform
    spread = float(np.percentile(t_uniform_sorted, 95) - np.percentile(t_uniform_sorted, 5))
    print(f"  N = {N_uniform}, p5..p95 spread = {spread:.1f} s")

    # ----- Plots -----
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    for ax, Ne in zip(axes, NE_LIST):
        pr = pbox_results[Ne]
        cdfs = np.array(pr["cdfs"])
        p_levels = np.array(pr["p_levels"])
        # Plot the family
        for cdf in cdfs:
            ax.step(np.concatenate([[cdf[0]], cdf]),
                    np.concatenate([[0], p_levels]), where="post",
                    color="grey", alpha=0.45, lw=0.7)
        # Envelope
        x_lo = np.array(pr["x_lower"])
        x_up = np.array(pr["x_upper"])
        ax.step(np.concatenate([[x_lo[0]], x_lo]),
                np.concatenate([[0], p_levels]), where="post",
                color="#c44", lw=1.8, label=r"leftmost CDF ($\varepsilon_\mathrm{atm}=0.78$)")
        ax.step(np.concatenate([[x_up[0]], x_up]),
                np.concatenate([[0], p_levels]), where="post",
                color="#44c", lw=1.8, label=r"rightmost CDF ($\varepsilon_\mathrm{atm}=0.70$)")
        ax.set_title(fr"$N_e = {Ne}$ (median width = {pr['median_width']:.0f} s)")
        ax.set_xlabel(r"$t_\mathrm{ice}$ [s]")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Cumulative probability")
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(r"Nested-sampling p-box convergence (prediction $T_\mathrm{air,end}=263$ K)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_pbox_panel.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(f"{fig_dir}/fig_pbox_panel.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # P-box vs uniform CDF
    pr = pbox_results[NE_LIST[-1]]
    p_levels = np.array(pr["p_levels"])
    x_lo = np.array(pr["x_lower"])
    x_up = np.array(pr["x_upper"])

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.fill_betweenx(p_levels, x_up, x_lo, color="#f8d8a0", alpha=0.6,
                     label=r"P-box envelope ($N_e=25$)")
    ax.step(np.concatenate([[x_lo[0]], x_lo]),
            np.concatenate([[0], p_levels]), where="post",
            color="#c44", lw=1.8)
    ax.step(np.concatenate([[x_up[0]], x_up]),
            np.concatenate([[0], p_levels]), where="post",
            color="#44c", lw=1.8)
    ax.step(np.concatenate([[t_uniform_sorted[0]], t_uniform_sorted]),
            np.concatenate([[0], p_uniform]), where="post",
            color="black", lw=1.5,
            label=r"Probabilistic CDF ($\varepsilon_\mathrm{atm}\sim$ Unif)")
    ax.set_xlabel(r"$t_\mathrm{ice}$ [s]")
    ax.set_ylabel("Cumulative probability")
    ax.set_title("P-box vs. probabilistic-uniform counterfactual")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_pbox_vs_uniform.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_pbox_vs_uniform.png", dpi=200)
    plt.close(fig)

    summary = {
        "Ne_list": NE_LIST,
        "pbox": pbox_results,
        "uniform_t_sorted": t_uniform_sorted.tolist(),
        "uniform_p": p_uniform.tolist(),
        "uniform_p5_p95_spread": spread,
    }
    with open(f"{results_dir}/pbox.json", "w") as f:
        json.dump(summary, f, default=float)
    return summary


if __name__ == "__main__":
    main()
