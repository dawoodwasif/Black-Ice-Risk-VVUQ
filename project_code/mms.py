"""
HW3 redux — code verification via the Method of Manufactured Solutions.

Manufactured solution (HW3, eq. 8):
    T_ms(z, t) = T0 + Tz sin(az pi z/L) + Tt cos(at pi t/tau)
                + Tzt cos(azt pi z/L) sin(azt2 pi t/tau)

Analytic source term derived symbolically with SymPy and verified at four
test points to machine precision.  Six systematically-refined meshes with
refinement ratio r=2; observed order computed from consecutive pairs.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sympy as sp

from solver import Material, solve_mms

L_DOMAIN = 0.5
TAU = 3600.0  # 1 hour for MMS run

T0 = 280.0
Tz = 5.0
az = 1.0
Tt = 3.0
at = 1.0
Tzt = 2.0
azt = 2.0
azt2 = 1.5

MAT = Material(rho=2100.0, cp=920.0, k=1.2)


def ensure_dir(path: str) -> Path:
    out_dir = Path(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def T_ms(z, t):
    z = np.asarray(z, dtype=float)
    return (
        T0
        + Tz * np.sin(az * np.pi * z / L_DOMAIN)
        + Tt * np.cos(at * np.pi * t / TAU)
        + Tzt * np.cos(azt * np.pi * z / L_DOMAIN) * np.sin(azt2 * np.pi * t / TAU)
    )


def s_ms(z, t):
    """rho cp dT/dt - k d2T/dz2."""
    z = np.asarray(z, dtype=float)
    dTdt = (
        -Tt * (at * np.pi / TAU) * np.sin(at * np.pi * t / TAU)
        + Tzt * (azt2 * np.pi / TAU)
        * np.cos(azt * np.pi * z / L_DOMAIN)
        * np.cos(azt2 * np.pi * t / TAU)
    )
    d2Tdz2 = (
        -Tz * (az * np.pi / L_DOMAIN) ** 2 * np.sin(az * np.pi * z / L_DOMAIN)
        - Tzt * (azt * np.pi / L_DOMAIN) ** 2
        * np.cos(azt * np.pi * z / L_DOMAIN)
        * np.sin(azt2 * np.pi * t / TAU)
    )
    return MAT.rho * MAT.cp * dTdt - MAT.k * d2Tdz2


# ---------------------------- SymPy verification -----------------------------


def verify_source_with_sympy() -> dict:
    z, t = sp.symbols("z t", real=True)
    Tms = (
        T0
        + Tz * sp.sin(az * sp.pi * z / L_DOMAIN)
        + Tt * sp.cos(at * sp.pi * t / TAU)
        + Tzt * sp.cos(azt * sp.pi * z / L_DOMAIN) * sp.sin(azt2 * sp.pi * t / TAU)
    )
    s_expr = MAT.rho * MAT.cp * sp.diff(Tms, t) - MAT.k * sp.diff(Tms, z, 2)
    s_func = sp.lambdify((z, t), s_expr, "numpy")
    test_pts = [(0.1, 600.0), (0.25, 1800.0), (0.4, 3000.0), (0.05, 900.0)]
    rows = []
    max_rel = 0.0
    for zp, tp in test_pts:
        s_sym = float(s_func(zp, tp))
        s_code = float(s_ms(np.array([zp]), tp)[0])
        rel = abs(s_sym - s_code) / max(abs(s_sym), 1.0)
        max_rel = max(max_rel, rel)
        rows.append((zp, tp, s_sym, s_code, rel))
    return {"rows": rows, "max_rel": max_rel}


# ---------------------------- grid refinement --------------------------------


def run_refinement(levels: int = 6):
    Nz_list = [8 * 2 ** ell for ell in range(levels)]
    Nt_list = [16 * 2 ** ell for ell in range(levels)]
    rows = []
    for Nz, Nt in zip(Nz_list, Nt_list):
        z, field = solve_mms(Nz, Nt, L_DOMAIN, TAU, MAT, T_ms, s_ms)
        T_num = field[-1]
        T_ex = T_ms(z, TAU)
        e = T_num[1:-1] - T_ex[1:-1]  # interior nodes only
        L1 = float(np.mean(np.abs(e)))
        L2 = float(np.sqrt(np.mean(e ** 2)))
        Linf = float(np.max(np.abs(e)))
        Fo = MAT.alpha * (TAU / Nt) / (L_DOMAIN / Nz) ** 2
        rows.append({"Nz": Nz, "Nt": Nt, "Fo": Fo, "L1": L1, "L2": L2, "Linf": Linf})
    return rows


def observed_orders(rows):
    pairs = []
    for i in range(1, len(rows)):
        coarse = rows[i - 1]
        fine = rows[i]
        pL1 = float(np.log(coarse["L1"] / fine["L1"]) / np.log(2))
        pL2 = float(np.log(coarse["L2"] / fine["L2"]) / np.log(2))
        pLinf = float(np.log(coarse["Linf"] / fine["Linf"]) / np.log(2))
        pairs.append({"pair": f"{i}->{i+1}", "pL1": pL1, "pL2": pL2, "pLinf": pLinf})
    return pairs


# ------------------------------ plotting -------------------------------------


def make_figures(rows, fig_dir: str):
    fig_dir = ensure_dir(fig_dir)
    h = np.array([rows[-1]["Nz"] / r["Nz"] for r in rows])  # normalised spacing
    L1 = np.array([r["L1"] for r in rows])
    L2 = np.array([r["L2"] for r in rows])
    Linf = np.array([r["Linf"] for r in rows])

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    ax.loglog(h, L1, "o-", label=r"$L_1$ norm", color="#1f77b4")
    ax.loglog(h, L2, "s-", label=r"$L_2$ norm", color="#d62728")
    ax.loglog(h, Linf, "^-", label=r"$L_\infty$ norm", color="#2ca02c")
    # Reference 2nd-order slope
    ref = L2[-1] * (h / h[-1]) ** 2
    ax.loglog(h, ref, "k--", lw=1.0, label=r"slope $p=2$")
    ax.set_xlabel(r"Normalised grid spacing, $h$")
    ax.set_ylabel(r"Discretisation error norm")
    ax.set_title("MMS code verification: error norms vs grid spacing")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_mms_norms.pdf", dpi=200)
    fig.savefig(fig_dir / "fig_mms_norms.png", dpi=200)
    plt.close(fig)

    pairs = observed_orders(rows)
    p = np.array([row["pL2"] for row in pairs])
    h_pairs = h[1:]

    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.plot(h_pairs, [row["pL1"] for row in pairs], "o-", label=r"$L_1$")
    ax.plot(h_pairs, [row["pL2"] for row in pairs], "s-", label=r"$L_2$")
    ax.plot(h_pairs, [row["pLinf"] for row in pairs], "^-", label=r"$L_\infty$")
    ax.axhline(2.0, color="k", linestyle="--", lw=1.0, label="formal $p=2$")
    ax.set_xscale("log")
    ax.set_xlabel(r"Normalised grid spacing, $h$")
    ax.set_ylabel(r"Observed order $\hat p$")
    ax.set_ylim(1.5, 2.5)
    ax.set_title("MMS: observed order of accuracy")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_mms_orders.pdf", dpi=200)
    fig.savefig(fig_dir / "fig_mms_orders.png", dpi=200)
    plt.close(fig)


def main(fig_dir: str = "../figs", results_dir: str = "../results"):
    fig_dir = ensure_dir(fig_dir)
    results_dir = ensure_dir(results_dir)
    print("=== HW3 redux: MMS code verification ===")
    sym = verify_source_with_sympy()
    print(f"  SymPy source verification — max relative diff: {sym['max_rel']:.2e}")

    rows = run_refinement()
    print(f"  Mesh        Fo        |e|_L1       |e|_L2       |e|_Linf")
    for r in rows:
        print(f"  Nz={r['Nz']:4d} Nt={r['Nt']:5d} {r['Fo']:.4f} "
              f"{r['L1']:.4e}  {r['L2']:.4e}  {r['Linf']:.4e}")

    pairs = observed_orders(rows)
    print("\n  Observed order between consecutive pairs:")
    print("  pair          p(L1)    p(L2)    p(Linf)")
    for p in pairs:
        print(f"  {p['pair']:8s}  {p['pL1']:6.4f}  {p['pL2']:6.4f}  {p['pLinf']:6.4f}")

    make_figures(rows, fig_dir)

    summary = {"sym_max_rel": sym["max_rel"], "rows": rows, "pairs": pairs}
    with open(results_dir / "mms.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("  Saved figures and JSON.")
    return summary


if __name__ == "__main__":
    main()
