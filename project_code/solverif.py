"""
HW4 redux at the prediction-domain bounding box.

Project requirement: redo solution verification at multiple points in the
input-uncertainty space — the four corners (V_wind = mu ± 2 sigma; eps_atm at
its lower and upper interval bound) and the centre.  Three meshes per point;
Richardson extrapolation with factor of safety F_s = 1.25 (since observed
order is within 10% of formal p=2 at every point).

The conservative predictive numerical uncertainty is the corner maximum.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from solver import PhysicalScenario, solve_physical


# Aleatory: V_wind ~ N(2.5, 1.0^2)
MU_V = 2.5
SIG_V = 1.0
EPS_LO, EPS_HI = 0.70, 0.78

T_AIR_END_PRED = 263.0  # prediction condition
NEWTON_TOL = 1e-12

POINTS = {
    "centre":               (MU_V,             0.5 * (EPS_LO + EPS_HI)),
    "(mu-2s, eps_lo)":      (MU_V - 2 * SIG_V, EPS_LO),
    "(mu-2s, eps_hi)":      (MU_V - 2 * SIG_V, EPS_HI),
    "(mu+2s, eps_lo)":      (MU_V + 2 * SIG_V, EPS_LO),
    "(mu+2s, eps_hi)":      (MU_V + 2 * SIG_V, EPS_HI),
}

MESHES = [(16, 128), (32, 256), (64, 512)]   # coarse / medium / fine


def t_ice_at(Nz: int, Nt: int, V_wind: float, eps_atm: float, T_air_end: float) -> float:
    scn = PhysicalScenario(V_wind=V_wind, eps_atm=eps_atm, T_air_end=T_air_end)
    res = solve_physical(Nz, Nt, scn, newton_tol=NEWTON_TOL)
    return res["t_ice"]


def gci_three_mesh(f_coarse, f_med, f_fine, r=2.0, Fs=1.25):
    """Roache GCI with observed order p_hat from a 3-mesh triplet."""
    num = (f_coarse - f_med)
    den = (f_med - f_fine)
    if abs(den) < 1e-14:
        return 0.0, np.nan, 0.0
    p_hat = float(np.log(num / den) / np.log(r)) if num * den > 0 else float("nan")
    DE1 = (f_fine - f_med) / (r ** p_hat - 1) if np.isfinite(p_hat) else 0.0
    GCI = Fs * abs(DE1)
    return p_hat, DE1, GCI


def main(fig_dir: str = "../figs", results_dir: str = "../results"):
    print("=== Solution verification at prediction-domain corners ===")
    print(f"  Prediction T_air,end = {T_AIR_END_PRED} K")
    out = {}
    for name, (V, eps) in POINTS.items():
        f_coarse = t_ice_at(*MESHES[0], V, eps, T_AIR_END_PRED)
        f_med    = t_ice_at(*MESHES[1], V, eps, T_AIR_END_PRED)
        f_fine   = t_ice_at(*MESHES[2], V, eps, T_AIR_END_PRED)
        p_hat, DE1, GCI = gci_three_mesh(f_coarse, f_med, f_fine)
        out[name] = {
            "V_wind": V, "eps_atm": eps,
            "f_coarse": f_coarse, "f_med": f_med, "f_fine": f_fine,
            "p_hat": p_hat, "DE1": DE1, "GCI": GCI,
        }
        print(f"  {name:25s} V={V:.2f}  eps={eps:.2f}  "
              f"p_hat={p_hat:.3f}  DE1={DE1:7.3f} s  GCI={GCI:7.3f} s")

    U_NUM = max(o["GCI"] for o in out.values())
    worst_corner = max(out, key=lambda k: out[k]["GCI"])
    print(f"\n  Conservative U_NUM = {U_NUM:.3f} s (at {worst_corner})")

    # ----------- Bar chart of GCI at each location -----------
    fig, ax = plt.subplots(figsize=(7, 4.4))
    names = list(out.keys())
    gcis = [out[n]["GCI"] for n in names]
    colors = ["#4c72b0"] + ["#dd8452"] * 4
    bars = ax.bar(range(len(names)), gcis, color=colors, edgecolor="black", linewidth=0.5)
    for i, (bar, g) in enumerate(zip(bars, gcis)):
        ax.text(bar.get_x() + bar.get_width() / 2, g + 0.15, f"{g:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace("(mu", r"($\mu$").replace("eps_lo", r"$\varepsilon_{atm}$ lo")
                          .replace("eps_hi", r"$\varepsilon_{atm}$ hi").replace("2s", r"$2\sigma$")
                       for n in names], rotation=20, ha="right", fontsize=9)
    ax.axhline(U_NUM, color="red", linestyle="--", lw=1.0,
               label=fr"$U_\mathrm{{NUM}} = {U_NUM:.2f}$ s (corner max)")
    ax.set_ylim(0, U_NUM * 1.30)
    ax.set_ylabel(r"GCI$_{12}$ for $t_\mathrm{ice}$ [s]")
    ax.set_title("Numerical uncertainty across prediction-domain corners")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_corners_GCI.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_corners_GCI.png", dpi=200)
    plt.close(fig)

    # ----------- Observed order at each location -----------
    fig, ax = plt.subplots(figsize=(7, 4.0))
    ps = [out[n]["p_hat"] for n in names]
    bars = ax.bar(range(len(names)), ps, color="#55a868", edgecolor="black", linewidth=0.5)
    for bar, p in zip(bars, ps):
        ax.text(bar.get_x() + bar.get_width() / 2, p + 0.03, f"{p:.2f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(2.0, color="k", linestyle="--", lw=1.0, label="formal $p=2$")
    ax.axhspan(1.8, 2.2, alpha=0.1, color="grey", label="±10% of formal")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n for n in names], rotation=20, ha="right", fontsize=9)
    ax.set_ylim(1.5, 2.8)
    ax.set_ylabel(r"Observed order $\hat p$")
    ax.set_title("Observed order of accuracy across prediction-domain corners")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_corners_orders.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_corners_orders.png", dpi=200)
    plt.close(fig)

    summary = {"U_NUM": U_NUM, "worst_corner": worst_corner, "points": out}
    with open(f"{results_dir}/corners.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    main()
