"""
Total predictive uncertainty: stack the bands.

  yellow band : input p-box (epistemic + aleatory)
  blue band   : input p-box widened by MAVM (model form)
  red band    : further widened by ±U_NUM (numerical)
  black curve : probabilistic-uniform reference

Also compute the four-component budget at the median:
  aleatory wind, epistemic emissivity, model form, numerical.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from solver import PhysicalScenario, solve_physical


def _np_array(d, key):
    return np.array(d[key])


def stack_bands(pbox_p, pbox_lo, pbox_up, U_MAVM_minus, U_MAVM_plus, U_NUM):
    """Successively widen the input p-box."""
    # Input
    inp_lo = pbox_lo.copy()
    inp_up = pbox_up.copy()
    # Model-form widening (asymmetric per MAVM convention)
    mf_lo = inp_lo - U_MAVM_minus
    mf_up = inp_up + U_MAVM_plus
    # Numerical widening (symmetric)
    num_lo = mf_lo - U_NUM
    num_up = mf_up + U_NUM
    return inp_lo, inp_up, mf_lo, mf_up, num_lo, num_up


def median_widths(p, lo, up):
    """Width of an envelope at p = 0.5 by linear interpolation."""
    lo_med = float(np.interp(0.5, p, lo))
    up_med = float(np.interp(0.5, p, up))
    return up_med - lo_med, lo_med, up_med


def aleatory_spread_from_centre(p_levels, cdfs_array, eps_endpoints,
                                 eps_target=0.74, p_low=0.05, p_high=0.95):
    """Spread of the inner CDF closest to eps_atm = eps_target between p_low and p_high."""
    idx = int(np.argmin(np.abs(np.asarray(eps_endpoints) - eps_target)))
    cdf = np.asarray(cdfs_array[idx])  # sorted samples
    p = np.asarray(p_levels)
    x_low = float(np.interp(p_low, p, cdf))
    x_high = float(np.interp(p_high, p, cdf))
    return x_high - x_low


def main(corners_summary, mavm_summary, pbox_summary,
         fig_dir: str = "../figs", results_dir: str = "../results"):
    print("=== Total predictive uncertainty ===")
    U_NUM = corners_summary["U_NUM"]
    U_MAVM_plus = mavm_summary["U_MAVM_plus"]
    U_MAVM_minus = max(mavm_summary["U_MAVM_minus"], 0.0)
    print(f"  U_NUM         = {U_NUM:.2f} s")
    print(f"  U_MAVM+       = {U_MAVM_plus:.2f} s")
    print(f"  U_MAVM-       = {U_MAVM_minus:.2f} s")

    # Use Ne = 25 for total stack
    pr_25 = pbox_summary["pbox"][str(25)] if str(25) in pbox_summary["pbox"] else pbox_summary["pbox"][25]
    p_levels = np.asarray(pr_25["p_levels"])
    pbox_lo = np.asarray(pr_25["x_lower"])
    pbox_up = np.asarray(pr_25["x_upper"])
    eps_endpts = pr_25["eps_endpoints"]
    cdfs = pr_25["cdfs"]

    inp_lo, inp_up, mf_lo, mf_up, num_lo, num_up = stack_bands(
        p_levels, pbox_lo, pbox_up, U_MAVM_minus, U_MAVM_plus, U_NUM
    )

    # Median widths
    w_inp, _, _ = median_widths(p_levels, inp_lo, inp_up)
    w_mf, _, _ = median_widths(p_levels, mf_lo, mf_up)
    w_total, lo_total, up_total = median_widths(p_levels, num_lo, num_up)
    print(f"  Median widths: input = {w_inp:.1f}, +model-form = {w_mf:.1f}, "
          f"+numerical = {w_total:.1f} s")

    # Budget
    aleatory_spread = aleatory_spread_from_centre(p_levels, cdfs, eps_endpts,
                                                   eps_target=0.74)
    epistemic_spread = max(w_inp - aleatory_spread, 0.0)
    model_form_spread = U_MAVM_plus + U_MAVM_minus
    numerical_spread = 2 * U_NUM
    total = aleatory_spread + epistemic_spread + model_form_spread + numerical_spread
    budget = {
        "aleatory_V_wind":   aleatory_spread,
        "epistemic_eps_atm": epistemic_spread,
        "model_form":        model_form_spread,
        "numerical":         numerical_spread,
        "total":             total,
        "shares": {
            "aleatory":  aleatory_spread / total,
            "epistemic": epistemic_spread / total,
            "model":     model_form_spread / total,
            "numerical": numerical_spread / total,
        },
    }
    print(f"\n  Budget breakdown (additive, at median):")
    print(f"    Aleatory  V_wind       : {aleatory_spread:7.2f} s ({100 * aleatory_spread / total:5.1f}%)")
    print(f"    Epistemic eps_atm      : {epistemic_spread:7.2f} s ({100 * epistemic_spread / total:5.1f}%)")
    print(f"    Model-form (d+ + d-)   : {model_form_spread:7.2f} s ({100 * model_form_spread / total:5.1f}%)")
    print(f"    Numerical (2 U_NUM)    : {numerical_spread:7.2f} s ({100 * numerical_spread / total:5.1f}%)")
    print(f"    TOTAL                  : {total:7.2f} s")

    # ----------- total uncertainty figure -----------
    uniform_t = np.asarray(pbox_summary["uniform_t_sorted"])
    uniform_p = np.asarray(pbox_summary["uniform_p"])

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.fill_betweenx(p_levels, num_up, num_lo, color="#e34a4a", alpha=0.85,
                     label=fr"Numerical ($\pm U_\mathrm{{NUM}}$)")
    ax.fill_betweenx(p_levels, mf_up, mf_lo, color="#4c72b0", alpha=0.85,
                     label=fr"Model-form (MAVM PI)")
    ax.fill_betweenx(p_levels, inp_up, inp_lo, color="#f5d76e", alpha=0.95,
                     label=fr"Input p-box ($N_e=25$)")
    ax.step(np.concatenate([[uniform_t[0]], uniform_t]),
            np.concatenate([[0], uniform_p]), where="post",
            color="black", lw=1.6,
            label=r"Probabilistic-uniform reference")
    ax.set_xlabel(r"$t_\mathrm{ice}$ [s]")
    ax.set_ylabel("Cumulative probability")
    ax.set_title(r"Total predictive uncertainty in $t_\mathrm{ice}$ at $T_\mathrm{air,end}=263$ K")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    # Annotation
    ax.text(0.02, 0.97,
            f"Total median width: {w_total:.0f} s\n"
            f"  ≈ {w_total/60:.1f} min",
            transform=ax.transAxes, fontsize=10, va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_total.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_total.png", dpi=200)
    plt.close(fig)

    # Budget bar chart + pie
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    labels = ["Aleatory\n($V_\\mathrm{wind}$)", "Epistemic\n($\\varepsilon_\\mathrm{atm}$)",
              "Model form\n($d^++d^-$)", "Numerical\n($2\\, U_\\mathrm{NUM}$)"]
    values = [aleatory_spread, epistemic_spread, model_form_spread, numerical_spread]
    colors = ["#f5d76e", "#dd8452", "#4c72b0", "#e34a4a"]
    bars = axes[0].bar(labels, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, v + 30, f"{v:.0f} s\n({100*v/total:.0f}%)",
                     ha="center", va="bottom", fontsize=9)
    axes[0].set_ylim(0, max(values) * 1.20)  # headroom so labels don't touch the title
    axes[0].set_ylabel("Contribution to $t_\\mathrm{ice}$ uncertainty [s]")
    axes[0].set_title(f"Predictive-uncertainty budget (total: {total:.0f} s ≈ {total/60:.0f} min)")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].pie(values, labels=labels, colors=colors, autopct="%1.1f%%",
                startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    axes[1].set_title("Share of total uncertainty")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_budget.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_budget.png", dpi=200)
    plt.close(fig)

    summary = {
        "U_NUM": U_NUM, "U_MAVM_plus": U_MAVM_plus, "U_MAVM_minus": U_MAVM_minus,
        "median_width_input": w_inp,
        "median_width_with_modelform": w_mf,
        "median_width_total": w_total,
        "median_total_lower": lo_total,
        "median_total_upper": up_total,
        "budget": budget,
    }
    with open(f"{results_dir}/total.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    with open("../results/corners.json") as f:
        corners = json.load(f)
    with open("../results/mavm.json") as f:
        mavm = json.load(f)
    with open("../results/pbox.json") as f:
        pbox = json.load(f)
    main(corners, mavm, pbox)
