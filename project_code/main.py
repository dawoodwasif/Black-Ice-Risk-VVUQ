"""Orchestrator: run the full pipeline end-to-end.

Order:
  1. Sanity baseline                (one solver call)
  2. MMS code verification          (HW3 redux)
  3. Solution verif. at corners     (HW4 redux)
  4. MAVM at five points + extrap.  (HW5 + project extension)
  5. Nested-sampling p-box          (Ne in {5,10,25}) and prob-uniform
  6. Total predictive uncertainty
  7. Story figures (response surface, surface evolution, workflow, T-field)

All numerical results are dumped to ../results/*.json; figures land in ../figs.
"""

from __future__ import annotations

import json
import time

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "savefig.bbox": "tight",
    "figure.dpi": 110,
})


def main():
    t0 = time.time()
    print("=" * 70)
    print("VVUQ — Black Ice Risk Model — Final Project Driver")
    print("=" * 70)

    import solver  # noqa: F401
    import mms
    import solverif
    import validation
    import nestedsamp
    import total
    import extras

    print("\n" + "-" * 70)
    print("[1/7] Sanity baseline")
    from solver import PhysicalScenario, solve_physical
    res = solve_physical(32, 256, PhysicalScenario())
    assert abs(res["Ts_tau"] - 269.32) < 0.05, f"baseline Ts_tau = {res['Ts_tau']}"
    assert abs(res["t_ice"] - 7258.92) < 1.0, f"baseline t_ice = {res['t_ice']}"
    print(f"      OK: Ts(tau)={res['Ts_tau']:.4f} K, t_ice={res['t_ice']:.2f} s")

    print("\n" + "-" * 70)
    print("[2/7] MMS code verification")
    mms_summary = mms.main()

    print("\n" + "-" * 70)
    print("[3/7] Solution verification at corners")
    corners_summary = solverif.main()

    print("\n" + "-" * 70)
    print("[4/7] MAVM extrapolation + Devore PI verification")
    mavm_summary = validation.main()

    print("\n" + "-" * 70)
    print("[5/7] Nested-sampling p-box + probabilistic counterfactual")
    pbox_summary = nestedsamp.main()

    print("\n" + "-" * 70)
    print("[6/7] Total predictive uncertainty")
    total_summary = total.main(corners_summary, mavm_summary, pbox_summary)

    print("\n" + "-" * 70)
    print("[7/7] Story figures")
    extras.main()

    print("\n" + "=" * 70)
    print(f"All done in {time.time() - t0:.1f} s")
    print("=" * 70)

    # Write a top-level summary
    top = {
        "U_NUM": corners_summary["U_NUM"],
        "U_MAVM_plus": mavm_summary["U_MAVM_plus"],
        "U_MAVM_minus": mavm_summary["U_MAVM_minus"],
        "median_pbox_width_Ne25":
            pbox_summary["pbox"][25]["median_width"]
            if 25 in pbox_summary["pbox"]
            else pbox_summary["pbox"]["25"]["median_width"],
        "total_median_width": total_summary["median_width_total"],
        "budget_shares": total_summary["budget"]["shares"],
        "elapsed_s": time.time() - t0,
    }
    with open("../results/top_summary.json", "w") as f:
        json.dump(top, f, indent=2, default=float)


if __name__ == "__main__":
    main()
