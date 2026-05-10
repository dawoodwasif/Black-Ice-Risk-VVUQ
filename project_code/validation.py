"""
HW5 redux + project extension — MAVM regression and prediction-interval
extrapolation to the prediction location.

Modified Area Validation Metric (Voyles & Roy 2015):
    d+ = ∫ max(F_sim - S_n, 0) dy
    d- = ∫ max(S_n - F_sim, 0) dy
Both integrals evaluated exactly as a sum of step-function rectangles
on the union of jump points.

Synthetic experimental data follow the project's Option #2:
    SRQ_exp = alpha + chi * (beta - alpha)
with chi vector D2 = [0.10, 0.40, 0.60, 0.75, 0.80, 0.90, 0.91, 0.97, 1.30, 1.60].
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import qmc, norm, t as student_t

from solver import PhysicalScenario, solve_physical


MU_V = 2.5
SIG_V = 1.0
EPS_CENTER = 0.74

T_AIR_END_VALIDATION_GRID = [264.0, 266.0, 268.0, 270.0, 272.0]
T_AIR_END_PRED = 263.0

CHI_D2 = np.array([0.10, 0.40, 0.60, 0.75, 0.80, 0.90, 0.91, 0.97, 1.30, 1.60])

N_LHS = 50  # inner-loop aleatory samples


# --------------------------- LHS over V_wind ---------------------------------


def lhs_V_wind(n: int, seed: int = 0) -> np.ndarray:
    sampler = qmc.LatinHypercube(d=1, seed=seed)
    u = sampler.random(n=n).ravel()
    return norm.ppf(u, loc=MU_V, scale=SIG_V)


def t_ice_array(V_arr: np.ndarray, eps_atm: float, T_air_end: float,
                Nz: int = 32, Nt: int = 256) -> np.ndarray:
    out = np.empty(len(V_arr))
    for i, V in enumerate(V_arr):
        scn = PhysicalScenario(V_wind=float(V), eps_atm=eps_atm, T_air_end=T_air_end)
        out[i] = solve_physical(Nz, Nt, scn)["t_ice"]
    return out


# --------------------------- MAVM integration --------------------------------


def empirical_cdf_at(samples: np.ndarray, y: np.ndarray) -> np.ndarray:
    s = np.sort(samples)
    return np.searchsorted(s, y, side="right") / len(s)


def mavm(sim_samples: np.ndarray, exp_samples: np.ndarray):
    """Return (d_plus, d_minus, AVM) computed exactly as step-function integrals."""
    pad = 0.05 * (max(sim_samples.max(), exp_samples.max()) - min(sim_samples.min(), exp_samples.min()))
    breakpoints = np.unique(np.concatenate([sim_samples, exp_samples,
                                            [sim_samples.min() - pad, sim_samples.max() + pad,
                                             exp_samples.min() - pad, exp_samples.max() + pad]]))
    breakpoints = np.sort(breakpoints)
    # Evaluate F_sim and S_n at each breakpoint+ (just to the right)
    eps = 1e-9
    y_eval = breakpoints[:-1] + eps
    F_sim = empirical_cdf_at(sim_samples, y_eval)
    S_n = empirical_cdf_at(exp_samples, y_eval)
    widths = np.diff(breakpoints)
    d_plus = float(np.sum(np.maximum(F_sim - S_n, 0.0) * widths))
    d_minus = float(np.sum(np.maximum(S_n - F_sim, 0.0) * widths))
    return d_plus, d_minus, d_plus + d_minus


# --------------------------- Devore textbook test ----------------------------


def devore_verification():
    """Reproduce J. Devore, Probability and Statistics, 7th ed., p. 446.

    Linear regression of y on x with n=13 data points.  At x=500, the textbook
    reports y_hat = 0.4046 with 95% prediction half-width 0.2992.
    """
    x = np.array([398, 292, 352, 575, 568, 450, 550, 408, 484, 350, 503, 600, 600], float)
    y = np.array([0.15, 0.05, 0.23, 0.43, 0.23, 0.40, 0.44, 0.44, 0.45, 0.09,
                  0.59, 0.63, 0.60], float)
    n = len(x)
    xbar = x.mean()
    Sxx = np.sum((x - xbar) ** 2)
    slope = np.sum((x - xbar) * (y - y.mean())) / Sxx
    intercept = y.mean() - slope * xbar
    y_pred = intercept + slope * x
    s = np.sqrt(np.sum((y - y_pred) ** 2) / (n - 2))

    x_star = 500.0
    y_star = intercept + slope * x_star
    se_pi = s * np.sqrt(1.0 + 1.0 / n + (x_star - xbar) ** 2 / Sxx)
    t_crit = student_t.ppf(0.975, n - 2)
    half = t_crit * se_pi
    return {"y_hat": float(y_star), "half_width_PI": float(half),
            "intercept": float(intercept), "slope": float(slope), "s": float(s),
            "n": n}


# --------------------------- MAVM at five points -----------------------------


def run_mavm_grid(seed: int = 0):
    rows = []
    for T_end in T_AIR_END_VALIDATION_GRID:
        V = lhs_V_wind(N_LHS, seed=seed + int(T_end))
        sim = t_ice_array(V, EPS_CENTER, T_end)
        # Synthetic experimental data: alpha = SRQ at V = mu+sig, beta = SRQ at V = mu-sig
        # (sensitivity at +- sig; chi=0..1 maps the synthetic range)
        alpha = solve_physical(32, 256, PhysicalScenario(V_wind=MU_V + SIG_V,
                                                          eps_atm=EPS_CENTER,
                                                          T_air_end=T_end))["t_ice"]
        beta = solve_physical(32, 256, PhysicalScenario(V_wind=MU_V - SIG_V,
                                                         eps_atm=EPS_CENTER,
                                                         T_air_end=T_end))["t_ice"]
        exp = alpha + CHI_D2 * (beta - alpha)
        d_plus, d_minus, AVM = mavm(sim, exp)
        rows.append({"T_air_end": T_end, "alpha": alpha, "beta": beta,
                     "d_plus": d_plus, "d_minus": d_minus, "AVM": AVM,
                     "sim_mean": float(np.mean(sim)), "sim_std": float(np.std(sim)),
                     "exp_mean": float(np.mean(exp)), "sim": sim.tolist(), "exp": exp.tolist()})
    return rows


def regress_and_extrapolate(T_arr: np.ndarray, y_arr: np.ndarray, x_star: float,
                            confidence: float = 0.95):
    n = len(T_arr)
    xbar = T_arr.mean()
    Sxx = np.sum((T_arr - xbar) ** 2)
    slope = np.sum((T_arr - xbar) * (y_arr - y_arr.mean())) / Sxx
    intercept = y_arr.mean() - slope * xbar
    y_pred = intercept + slope * T_arr
    s = np.sqrt(np.sum((y_arr - y_pred) ** 2) / (n - 2))
    y_star = intercept + slope * x_star
    se_pi = s * np.sqrt(1.0 + 1.0 / n + (x_star - xbar) ** 2 / Sxx)
    t_crit = student_t.ppf(0.5 + confidence / 2, n - 2)
    half = t_crit * se_pi
    return {"slope": float(slope), "intercept": float(intercept), "s": float(s),
            "y_star": float(y_star), "half_PI": float(half),
            "upper_PI": float(y_star + half)}


# ------------------------------- plotting ------------------------------------


def plot_devore(devore: dict, fig_dir: str):
    x = np.array([398, 292, 352, 575, 568, 450, 550, 408, 484, 350, 503, 600, 600], float)
    y = np.array([0.15, 0.05, 0.23, 0.43, 0.23, 0.40, 0.44, 0.44, 0.45, 0.09,
                  0.59, 0.63, 0.60], float)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    xs = np.linspace(250, 650, 200)
    ys = devore["intercept"] + devore["slope"] * xs

    n = devore["n"]
    xbar = x.mean()
    Sxx = np.sum((x - xbar) ** 2)
    s = devore["s"]
    t_crit = student_t.ppf(0.975, n - 2)
    se_ci = s * np.sqrt(1.0 / n + (xs - xbar) ** 2 / Sxx)
    se_pi = s * np.sqrt(1.0 + 1.0 / n + (xs - xbar) ** 2 / Sxx)

    ax.plot(xs, ys, "k-", lw=1.5, label="Linear regression")
    ax.fill_between(xs, ys - t_crit * se_ci, ys + t_crit * se_ci,
                    color="#4c72b0", alpha=0.25, label="95% confidence interval")
    ax.plot(xs, ys - t_crit * se_pi, "r--", lw=1.0, label="95% prediction interval")
    ax.plot(xs, ys + t_crit * se_pi, "r--", lw=1.0)
    ax.plot(x, y, "ko", markersize=6, label="Devore data")
    y_hat = devore["y_hat"]
    half = devore["half_width_PI"]
    ax.errorbar(500, y_hat, yerr=half, fmt="ro", markersize=8, capsize=4,
                label=f"PI at $x=500$: {y_hat:.4f} ± {half:.4f}")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title("Devore (2009): PI code verification")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_devore.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_devore.png", dpi=200)
    plt.close(fig)


def plot_mavm_extrap(rows, plus_fit, minus_fit, fig_dir: str):
    T_arr = np.array([r["T_air_end"] for r in rows])
    d_plus = np.array([r["d_plus"] for r in rows])
    d_minus = np.array([r["d_minus"] for r in rows])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, fit, data, name in (
        (axes[0], plus_fit, d_plus, "$d^{+}$"),
        (axes[1], minus_fit, d_minus, "$d^{-}$"),
    ):
        ax.plot(T_arr, data, "ko", markersize=8, label=f"{name} (5 validation pts)")
        Ts = np.linspace(min(T_AIR_END_PRED, T_arr.min()) - 1,
                         T_arr.max() + 1, 200)
        ys = fit["intercept"] + fit["slope"] * Ts
        ax.plot(Ts, ys, "k-", lw=1.4, label="Linear regression")

        n = len(T_arr)
        xbar = T_arr.mean()
        Sxx = np.sum((T_arr - xbar) ** 2)
        s = fit["s"]
        t_crit = student_t.ppf(0.975, n - 2)
        se_ci = s * np.sqrt(1.0 / n + (Ts - xbar) ** 2 / Sxx)
        se_pi = s * np.sqrt(1.0 + 1.0 / n + (Ts - xbar) ** 2 / Sxx)

        ax.fill_between(Ts, ys - t_crit * se_ci, ys + t_crit * se_ci,
                        color="#4c72b0", alpha=0.30, label="95% confidence interval")
        ax.fill_between(Ts, ys - t_crit * se_pi, ys + t_crit * se_pi,
                        color="#dd8452", alpha=0.20, label="95% prediction interval")

        # Prediction location
        ax.axvline(T_AIR_END_PRED, color="red", linestyle="--", lw=1.0,
                   label=f"prediction $T_\\mathrm{{air,end}} = {T_AIR_END_PRED}$ K")
        ax.errorbar(T_AIR_END_PRED, fit["y_star"], yerr=fit["half_PI"], fmt="rs",
                    markersize=8, capsize=4,
                    label=f"upper PI = {fit['upper_PI']:.1f} s")
        ax.set_xlabel(r"$T_\mathrm{air,end}$ [K]")
        ax.set_ylabel(f"{name} [s]")
        ax.set_title(f"{name} regression vs. control parameter")
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_mavm_extrap.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_mavm_extrap.png", dpi=200)
    plt.close(fig)


def plot_mavm_cdfs(rows, fig_dir: str):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex="col")
    axes = axes.ravel()
    rows_for_plot = rows[:5]
    for i, r in enumerate(rows_for_plot):
        ax = axes[i]
        sim = np.sort(np.array(r["sim"]))
        exp = np.sort(np.array(r["exp"]))

        # Step CDFs
        sim_y = np.arange(1, len(sim) + 1) / len(sim)
        exp_y = np.arange(1, len(exp) + 1) / len(exp)
        ax.step(np.concatenate([[sim[0]], sim]),
                np.concatenate([[0], sim_y]), where="post",
                color="#4c72b0", lw=1.6, label=r"$F_\mathrm{sim}$ (LHS, $n=50$)")
        ax.step(np.concatenate([[exp[0]], exp]),
                np.concatenate([[0], exp_y]), where="post",
                color="#dd8452", lw=1.6, label=r"$S_n$ (synthetic exp.)")
        ax.set_title(fr"$T_\mathrm{{air,end}} = {r['T_air_end']:.0f}$ K  "
                     fr"($d^{{+}}={r['d_plus']:.1f}, d^{{-}}={r['d_minus']:.1f}$ s)")
        ax.set_xlabel(r"$t_\mathrm{ice}$ [s]")
        ax.set_ylabel("CDF")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(frameon=False, loc="lower right", fontsize=9)
    axes[-1].axis("off")
    axes[-1].text(0.05, 0.55,
                  "MAVM decomposition at each control-parameter value.\n"
                  "$F_\\mathrm{sim}$ consistently lies left of $S_n$:\n"
                  "the model systematically under-predicts $t_\\mathrm{ice}$.\n"
                  "Magnitudes regressed against $T_\\mathrm{air,end}$\n"
                  "in the next figure for extrapolation to 263 K.",
                  fontsize=10, va="top", transform=axes[-1].transAxes)
    fig.tight_layout()
    fig.savefig(f"{fig_dir}/fig_mavm_cdfs.pdf", dpi=200)
    fig.savefig(f"{fig_dir}/fig_mavm_cdfs.png", dpi=200)
    plt.close(fig)


def main(fig_dir: str = "../figs", results_dir: str = "../results"):
    print("=== Devore (2009) PI verification ===")
    devore = devore_verification()
    print(f"  y_hat(500)    = {devore['y_hat']:.4f}  (textbook: 0.4046)")
    print(f"  PI half-width = {devore['half_width_PI']:.4f}  (textbook: 0.2992)")
    plot_devore(devore, fig_dir)

    print("\n=== MAVM at 5 control-parameter values ===")
    rows = run_mavm_grid()
    print(f"  {'T_end':>6}  {'d+':>9}  {'d-':>9}  {'AVM':>9}  {'sim_mu':>9}  {'exp_mu':>9}")
    for r in rows:
        print(f"  {r['T_air_end']:6.1f}  {r['d_plus']:9.3f}  {r['d_minus']:9.3f}  "
              f"{r['AVM']:9.3f}  {r['sim_mean']:9.2f}  {r['exp_mean']:9.2f}")

    T_arr = np.array([r["T_air_end"] for r in rows])
    d_plus = np.array([r["d_plus"] for r in rows])
    d_minus = np.array([r["d_minus"] for r in rows])
    plus_fit = regress_and_extrapolate(T_arr, d_plus, T_AIR_END_PRED)
    minus_fit = regress_and_extrapolate(T_arr, d_minus, T_AIR_END_PRED)

    print(f"\n  d+ regression at T={T_AIR_END_PRED} K: y_hat = {plus_fit['y_star']:.2f}, "
          f"upper PI = {plus_fit['upper_PI']:.2f} s")
    print(f"  d- regression at T={T_AIR_END_PRED} K: y_hat = {minus_fit['y_star']:.3f}, "
          f"upper PI = {minus_fit['upper_PI']:.3f} s")

    plot_mavm_extrap(rows, plus_fit, minus_fit, fig_dir)
    plot_mavm_cdfs(rows, fig_dir)

    summary = {
        "devore": devore, "rows": rows,
        "plus_fit": plus_fit, "minus_fit": minus_fit,
        "U_MAVM_plus": plus_fit["upper_PI"],
        "U_MAVM_minus": minus_fit["upper_PI"],
    }
    with open(f"{results_dir}/mavm.json", "w") as f:
        json.dump(summary, f, indent=2, default=float)
    return summary


if __name__ == "__main__":
    main()
