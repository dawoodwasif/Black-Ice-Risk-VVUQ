"""Extra figures that tell the story end-to-end."""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from solver import PhysicalScenario, solve_physical


def fig_surface_evolution(fig_dir: str = "../figs"):
    """Surface temperature time series at validation (T_end=268) and prediction (T_end=263) cases."""
    Nz, Nt = 64, 1024
    res_val = solve_physical(Nz, Nt, PhysicalScenario(T_air_end=268.0))
    res_pred = solve_physical(Nz, Nt, PhysicalScenario(T_air_end=263.0))

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    t_hr = res_val["t"] / 3600.0
    ax.plot(t_hr, res_val["Ts_t"], color="#4c72b0", lw=2.0,
            label=fr"Validation ($T_\mathrm{{air,end}}=268$ K)")
    ax.plot(t_hr, res_pred["Ts_t"], color="#dd8452", lw=2.0,
            label=fr"Prediction ($T_\mathrm{{air,end}}=263$ K)")
    ax.axhline(273.15, color="red", linestyle="--", lw=1.0, label="freezing point")

    for res, color, label_prefix in (
        (res_val, "#4c72b0", "validation"),
        (res_pred, "#dd8452", "prediction"),
    ):
        if np.isfinite(res["t_ice"]):
            ax.axvline(res["t_ice"] / 3600.0, color=color, lw=0.8, ls=":")
            ax.text(res["t_ice"] / 3600.0 + 0.05, 274.5,
                    fr"$t_\mathrm{{ice}}={res['t_ice']/60:.0f}\,$min",
                    color=color, fontsize=9, rotation=90, va="bottom")

    ax.set_xlabel("Time after sunset [hr]")
    ax.set_ylabel(r"Surface temperature $T_s$ [K]")
    ax.set_title("Nocturnal cooling: validation vs. prediction conditions")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_surface_evolution.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_surface_evolution.png", dpi=200)
    plt.close(fig)


def fig_response_surface(fig_dir: str = "../figs"):
    """t_ice as a function of (V_wind, eps_atm) at the prediction T_air_end."""
    V_grid = np.linspace(0.5, 4.5, 9)
    eps_grid = np.linspace(0.70, 0.78, 9)
    Z = np.empty((len(eps_grid), len(V_grid)))
    for i, eps in enumerate(eps_grid):
        for j, V in enumerate(V_grid):
            res = solve_physical(32, 256, PhysicalScenario(V_wind=V, eps_atm=eps,
                                                            T_air_end=263.0))
            Z[i, j] = res["t_ice"] / 60.0  # minutes

    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    cs = ax.contourf(V_grid, eps_grid, Z, levels=20, cmap="viridis")
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label(r"$t_\mathrm{ice}$ [min]")
    contours = ax.contour(V_grid, eps_grid, Z, levels=8, colors="white", linewidths=0.8)
    ax.clabel(contours, inline=True, fontsize=8, fmt="%.0f")

    # Mark the input box and centre
    ax.plot(2.5, 0.74, "r*", markersize=14, label="centre")
    box_V = [0.5, 4.5, 4.5, 0.5, 0.5]
    box_e = [0.70, 0.70, 0.78, 0.78, 0.70]
    ax.plot(box_V, box_e, "r-", lw=1.5, label="±2σ × interval")
    ax.set_xlabel(r"$V_\mathrm{wind}$ [m/s]")
    ax.set_ylabel(r"$\varepsilon_\mathrm{atm}$ [-]")
    ax.set_title(r"$t_\mathrm{ice}$ response surface at $T_\mathrm{air,end}=263$ K")
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_response_surface.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_response_surface.png", dpi=200)
    plt.close(fig)
    return V_grid.tolist(), eps_grid.tolist(), Z.tolist()


def fig_temperature_field(fig_dir: str = "../figs"):
    """Spatio-temporal temperature evolution at the prediction condition."""
    Nz, Nt = 64, 512
    res = solve_physical(Nz, Nt, PhysicalScenario(T_air_end=263.0), return_full=True)
    field = res["T_field"]
    z = res["z"]
    t = res["t"]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    cs = ax.contourf(t / 3600.0, z * 100, field.T, levels=30, cmap="RdBu_r",
                      vmin=263, vmax=283)
    cb = fig.colorbar(cs, ax=ax)
    cb.set_label("Temperature [K]")
    ax.contour(t / 3600.0, z * 100, field.T, levels=[273.15], colors="black", linewidths=1.2)
    ax.invert_yaxis()
    ax.set_xlabel("Time after sunset [hr]")
    ax.set_ylabel(r"Depth $z$ [cm]")
    ax.set_title(r"Road temperature field (prediction $T_\mathrm{air,end}=263$ K). "
                 r"Black contour = 0 °C")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_temperature_field.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_temperature_field.png", dpi=200)
    plt.close(fig)


def fig_workflow(fig_dir: str = "../figs"):
    """Polished, publication-quality workflow figure."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")

    # Refined palette (low-saturation, publication-quality)
    NAVY      = "#2A4A6E"
    NAVY_FILL = "#DEE7F0"
    TEAL      = "#3F6F70"
    TEAL_FILL = "#E0EAEA"
    BURGUNDY  = "#7A2E3F"
    BURGUNDY_FILL = "#F1E2E5"
    SHADOW    = "#999999"
    GREY_TEXT = "#3A3A3A"

    stages = [
        ("I",   "Code Verification",        "Method of Manufactured\nSolutions; second-order\naccuracy verified",  NAVY,     NAVY_FILL),
        ("II",  "Solution Verification",    "GCI at 4 corners + centre\nof input-uncertainty box",                NAVY,     NAVY_FILL),
        ("III", "Validation",               "MAVM at 5 control values\n+ 95% PI extrapolation",                   TEAL,     TEAL_FILL),
        ("IV",  "Uncertainty Propagation",  "Nested LHS p-box;\n$N_e \\in \\{5,10,25\\}$",                         TEAL,     TEAL_FILL),
    ]

    box_w = 2.55
    box_h = 1.55
    y_top = 4.0
    x_starts = [0.20, 3.05, 5.90, 8.75]
    header_frac = 0.32

    def draw_card(x, y, label, title, body, edge, fill):
        # Drop shadow
        ax.add_patch(FancyBboxPatch(
            (x + 0.08, y - 0.12), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=SHADOW, edgecolor="none", alpha=0.18, zorder=1,
        ))
        # Body card
        ax.add_patch(FancyBboxPatch(
            (x, y), box_w, box_h,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=fill, edgecolor=edge, linewidth=1.2, zorder=2,
        ))
        # Header band (rounded top, square bottom via overlay)
        header_y = y + box_h * (1 - header_frac)
        ax.add_patch(FancyBboxPatch(
            (x, header_y), box_w, box_h * header_frac,
            boxstyle="round,pad=0.02,rounding_size=0.10",
            facecolor=edge, edgecolor=edge, linewidth=0, zorder=3,
        ))
        ax.add_patch(plt.Rectangle(
            (x, header_y - 0.005), box_w, 0.16,
            facecolor=edge, edgecolor="none", zorder=3,
        ))
        # Title centered in the full header width
        title_y = header_y + box_h * header_frac / 2 + 0.02
        ax.text(x + box_w / 2, title_y, title,
                ha="center", va="center", fontsize=9.5, fontweight="bold",
                color="white", family="serif", zorder=5)
        # Body text
        ax.text(x + box_w / 2, y + box_h * (1 - header_frac) / 2,
                body, ha="center", va="center", fontsize=8.8,
                color=GREY_TEXT, family="serif", zorder=5)

    for (label, title, body, edge, fill), x0 in zip(stages, x_starts):
        draw_card(x0, y_top - box_h, label, title, body, edge, fill)

    def group_bracket(x_left, x_right, y, label, color):
        ax.plot([x_left, x_right], [y, y], color=color, lw=1.2)
        for xv in (x_left, x_right):
            ax.plot([xv, xv], [y, y - 0.12], color=color, lw=1.2)
        ax.text((x_left + x_right) / 2, y + 0.18, label,
                ha="center", va="bottom", fontsize=10.5, color=color,
                family="serif", style="italic")

    bracket_y = y_top + 0.42
    group_bracket(x_starts[0] + 0.05, x_starts[1] + box_w - 0.05,
                  bracket_y, "Verification", NAVY)
    group_bracket(x_starts[2] + 0.05, x_starts[3] + box_w - 0.05,
                  bracket_y, "Validation & Uncertainty Propagation", TEAL)

    for i in range(3):
        x_a = x_starts[i] + box_w
        x_b = x_starts[i + 1]
        y_arrow = y_top - box_h / 2
        ax.add_patch(FancyArrowPatch(
            (x_a + 0.05, y_arrow), (x_b - 0.05, y_arrow),
            arrowstyle="-|>", mutation_scale=14, lw=1.4,
            color=GREY_TEXT, zorder=6,
        ))

    out_w = 5.4
    out_h = 1.35
    out_x = (12 - out_w) / 2
    out_y = 0.55
    ax.add_patch(FancyBboxPatch(
        (out_x + 0.08, out_y - 0.12), out_w, out_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=SHADOW, edgecolor="none", alpha=0.18, zorder=1,
    ))
    ax.add_patch(FancyBboxPatch(
        (out_x, out_y), out_w, out_h,
        boxstyle="round,pad=0.02,rounding_size=0.12",
        facecolor=BURGUNDY_FILL, edgecolor=BURGUNDY, linewidth=1.4, zorder=2,
    ))
    ax.text(out_x + out_w / 2, out_y + out_h - 0.30,
            "Total Predictive Uncertainty",
            ha="center", va="center", fontsize=11.5, fontweight="bold",
            color=BURGUNDY, family="serif", zorder=5)
    ax.text(out_x + out_w / 2, out_y + 0.42,
            r"input p-box $\;\oplus\;$ MAVM $\;\oplus\;$ $U_\mathrm{NUM}$",
            ha="center", va="center", fontsize=10.5,
            color=GREY_TEXT, family="serif", zorder=5)

    # Aggregation arrows: each stage to the output node.
    # Straight lines for true mirror symmetry; destinations spread across the
    # top edge of the output box so the lines fan in cleanly without overlap.
    out_center_x = out_x + out_w / 2
    fan_offsets = [-1.7, -0.6, 0.6, 1.7]   # symmetric about centre
    for x0, dx in zip(x_starts, fan_offsets):
        x_src = x0 + box_w / 2
        y_src = y_top - box_h
        x_dst = out_center_x + dx
        y_dst = out_y + out_h
        ax.add_patch(FancyArrowPatch(
            (x_src, y_src - 0.05), (x_dst, y_dst + 0.05),
            arrowstyle="-|>", mutation_scale=12, lw=1.1,
            color=GREY_TEXT, alpha=0.6, zorder=1,
        ))

    ax.text(6, 5.55, "Verification, Validation, and Predictive UQ Workflow",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=NAVY, family="serif")

    fig.savefig(f"{fig_dir}/fig_workflow.pdf", dpi=200, bbox_inches="tight")
    fig.savefig(f"{fig_dir}/fig_workflow.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main(fig_dir: str = "../figs", results_dir: str = "../results"):
    print("=== Story figures ===")
    print("  → fig_surface_evolution")
    fig_surface_evolution(fig_dir)
    print("  → fig_response_surface")
    V, eps, Z = fig_response_surface(fig_dir)
    print("  → fig_temperature_field")
    fig_temperature_field(fig_dir)
    print("  → fig_workflow")
    fig_workflow(fig_dir)

    with open(f"{results_dir}/extra.json", "w") as f:
        json.dump({"V_grid": V, "eps_grid": eps, "t_ice_min": Z}, f)


if __name__ == "__main__":
    main()
