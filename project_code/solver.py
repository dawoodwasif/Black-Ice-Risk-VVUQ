"""
1D transient heat conduction solver for the black-ice VVUQ project.

Two modes:

  mode='mms'      — Dirichlet at both ends, manufactured-solution source term.
                    Used for code verification (HW3 redux).

  mode='physical' — Nonlinear Neumann at z=0 (surface energy balance),
                    Dirichlet at z=L (deep-ground temperature).
                    Solved with Crank-Nicolson + Newton iteration.

Numerical scheme:
  - Crank-Nicolson in time (formal O(dt^2))
  - 2nd-order central FD in space (formal O(dz^2))
  - Surface BC handled via ghost-cell elimination plus Newton iteration on Ts.
  - Tridiagonal solve via the Thomas algorithm.

Material defaults (asphalt, HW2/HW4 baseline):
    rho = 2100 kg/m^3, cp = 920 J/(kg K), k = 1.2 W/(m K).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

# --------------------------- material / constants ----------------------------

SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann constant, W/(m^2 K^4)


@dataclass
class Material:
    rho: float = 2100.0
    cp: float = 920.0
    k: float = 1.2

    @property
    def alpha(self) -> float:
        return self.k / (self.rho * self.cp)


@dataclass
class PhysicalScenario:
    """One nocturnal-cooling scenario."""
    L: float = 0.5
    tau: float = 6 * 3600.0
    T_air_start: float = 275.0
    T_air_end: float = 268.0  # validation default; vary for control-parameter sweep
    T_deep: float = 283.0
    T_s_init: float = 278.0
    eps_surface: float = 0.93
    eps_atm: float = 0.74
    V_wind: float = 2.5

    def T_air(self, t: float) -> float:
        return self.T_air_start + (self.T_air_end - self.T_air_start) * (t / self.tau)

    def h_conv(self) -> float:
        # Palyvos (2008) linear correlation for windward surfaces
        return 5.7 + 3.8 * self.V_wind


# ------------------------------ Thomas solver --------------------------------


def thomas(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Solve a tridiagonal system A x = d in O(N) time."""
    n = len(d)
    cp = np.empty(n - 1)
    dp = np.empty(n)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, n - 1):
        m = b[i] - a[i - 1] * cp[i - 1]
        cp[i] = c[i] / m
        dp[i] = (d[i] - a[i - 1] * dp[i - 1]) / m
    dp[n - 1] = (d[n - 1] - a[n - 2] * dp[n - 2]) / (b[n - 1] - a[n - 2] * cp[n - 2])
    x = np.empty(n)
    x[-1] = dp[-1]
    for i in range(n - 2, -1, -1):
        x[i] = dp[i] - cp[i] * x[i + 1]
    return x


# ------------------------------ MMS mode -------------------------------------


def solve_mms(
    Nz: int,
    Nt: int,
    L: float,
    tau: float,
    mat: Material,
    T_ms: Callable[[np.ndarray, float], np.ndarray],
    s_ms: Callable[[np.ndarray, float], np.ndarray],
):
    """Solve the heat eqn with Dirichlet BCs and manufactured source.

    Returns the final-time profile and the full (Nt+1, Nz+1) field.
    """
    z = np.linspace(0.0, L, Nz + 1)
    dz = z[1] - z[0]
    dt = tau / Nt
    Fo = mat.alpha * dt / dz ** 2

    T = T_ms(z, 0.0).astype(float).copy()
    field = np.empty((Nt + 1, Nz + 1))
    field[0] = T

    Nint = Nz - 1  # number of interior unknowns
    a = np.full(Nint - 1, -Fo / 2)
    b = np.full(Nint, 1.0 + Fo)
    c = np.full(Nint - 1, -Fo / 2)

    for n in range(Nt):
        t_n = n * dt
        t_np1 = (n + 1) * dt
        Tnew = T.copy()
        Tnew[0] = T_ms(z[0:1], t_np1)[0]
        Tnew[-1] = T_ms(z[-1:], t_np1)[0]

        s_n = s_ms(z[1:-1], t_n)
        s_np1 = s_ms(z[1:-1], t_np1)
        rhs = (
            (Fo / 2) * T[:-2]
            + (1.0 - Fo) * T[1:-1]
            + (Fo / 2) * T[2:]
            + (dt / (2 * mat.rho * mat.cp)) * (s_n + s_np1)
        )
        rhs[0] += (Fo / 2) * Tnew[0]
        rhs[-1] += (Fo / 2) * Tnew[-1]
        Tnew[1:-1] = thomas(a.copy(), b.copy(), c.copy(), rhs)
        T = Tnew
        field[n + 1] = T

    return z, field


# ------------------------------ physical mode --------------------------------


def _surface_flux(Ts: float, Tair: float, eps_s: float, eps_a: float, h: float) -> tuple[float, float]:
    """G(Ts) = eps_s sigma (eps_a Tair^4 - Ts^4) - h (Ts - Tair).

    Returns (G, dG/dTs).
    """
    G = eps_s * SIGMA_SB * (eps_a * Tair ** 4 - Ts ** 4) - h * (Ts - Tair)
    dG = -4.0 * eps_s * SIGMA_SB * Ts ** 3 - h
    return G, dG


def solve_physical(
    Nz: int,
    Nt: int,
    scn: PhysicalScenario,
    mat: Material = Material(),
    newton_tol: float = 1e-10,
    max_newton: int = 30,
    return_full: bool = False,
):
    """Solve the cooling problem with the nonlinear Neumann surface BC.

    Returns a dict containing:
        Ts_t        : surface temperature time series (length Nt+1)
        T_final     : final-time profile (length Nz+1)
        z           : depth grid
        t           : time grid
        t_ice       : first time Ts < 273.15 K (np.inf if never crossed)
        Ts_tau      : final-time surface temperature
        qs_tau      : final-time surface heat flux (W/m^2, 2nd-order one-sided FD)
        T_field     : (Nt+1, Nz+1) field, only if return_full=True
    """
    z = np.linspace(0.0, scn.L, Nz + 1)
    dz = z[1] - z[0]
    dt = scn.tau / Nt
    Fo = mat.alpha * dt / dz ** 2
    h = scn.h_conv()

    # IC: linear profile from T_s_init at z=0 to T_deep at z=L.
    T = scn.T_s_init + (scn.T_deep - scn.T_s_init) * z / scn.L
    Ts_t = np.empty(Nt + 1)
    Ts_t[0] = T[0]

    if return_full:
        field = np.empty((Nt + 1, Nz + 1))
        field[0] = T

    # Unknowns: T_0, T_1, ..., T_{Nz-1}.  T_{Nz} = T_deep (Dirichlet, known).
    Nun = Nz
    a_const = np.full(Nun - 1, -Fo / 2)  # subdiagonal
    c_const = np.full(Nun - 1, -Fo / 2)  # superdiagonal
    b = np.empty(Nun)

    # Surface BC (Crank-Nicolson + ghost-cell elimination):
    #     (1 + Fo) T_0^{n+1} - Fo T_1^{n+1}
    #         = T_0^n + Fo (T_1^n - T_0^n) + (dt/(rho cp dz)) [G^n + G^{n+1}]
    # Linearize G^{n+1}(T_0) ≈ G(T0*) + G'(T0*) (T_0 - T0*) and Newton-iterate.

    t_ice = np.inf
    C = dt / (mat.rho * mat.cp * dz)  # constant for the surface flux term
    for n in range(Nt):
        t_n = n * dt
        t_np1 = (n + 1) * dt
        Tair_n = scn.T_air(t_n)
        Tair_np1 = scn.T_air(t_np1)

        Gn, _ = _surface_flux(T[0], Tair_n, scn.eps_surface, scn.eps_atm, h)

        # Build RHS vector of length Nun = Nz.
        # Index 0 (surface): explicit (n-step) part
        rhs_surf_explicit = T[0] + Fo * (T[1] - T[0]) + C * Gn

        # Indices 1..Nz-2 (purely interior): standard CN explicit RHS.
        # T has length Nz+1; interior unknown indices in T are 1..Nz-1, i.e. T[1:-1].
        # For unknowns i = 1..Nz-2, we need T[0..Nz-3], T[1..Nz-2], T[2..Nz-1].
        # That is T[0:Nz-2], T[1:Nz-1], T[2:Nz].
        rhs_mid = (
            (Fo / 2) * T[0 : Nz - 2]
            + (1.0 - Fo) * T[1 : Nz - 1]
            + (Fo / 2) * T[2 : Nz]
        )

        # Index Nz-1 (last unknown, adjacent to Dirichlet T[Nz] = T_deep):
        # Standard CN explicit + (Fo/2) T_deep moved from LHS to RHS.
        rhs_last = (
            (Fo / 2) * T[Nz - 2]
            + (1.0 - Fo) * T[Nz - 1]
            + (Fo / 2) * T[Nz]
        ) + (Fo / 2) * scn.T_deep

        rhs = np.empty(Nun)
        rhs[1:-1] = rhs_mid
        rhs[-1] = rhs_last

        # Newton iteration on T_0^{n+1}.
        T0_iter = T[0]
        Tnew_full = T.copy()
        for it in range(max_newton):
            Gnp1, dGdT = _surface_flux(T0_iter, Tair_np1, scn.eps_surface, scn.eps_atm, h)
            b[0] = 1.0 + Fo - C * dGdT
            rhs0 = rhs_surf_explicit + C * (Gnp1 - dGdT * T0_iter)
            b[1:] = 1.0 + Fo

            rhs[0] = rhs0
            # Build c with proper surface coefficient (-Fo, not -Fo/2)
            c = c_const.copy()
            c[0] = -Fo  # surface row: (1+Fo) T_0 - Fo T_1
            Tnew = thomas(a_const.copy(), b.copy(), c, rhs)
            new_T0 = Tnew[0]
            if abs(new_T0 - T0_iter) < newton_tol:
                T0_iter = new_T0
                Tnew_full[:Nun] = Tnew
                Tnew_full[-1] = scn.T_deep
                break
            T0_iter = new_T0
            Tnew_full[:Nun] = Tnew
            Tnew_full[-1] = scn.T_deep

        T = Tnew_full
        Ts_t[n + 1] = T[0]

        if t_ice == np.inf and T[0] < 273.15:
            # Linear interpolation in t between previous and current.
            T0_prev = Ts_t[n]
            T0_curr = Ts_t[n + 1]
            if T0_prev > 273.15 >= T0_curr:
                frac = (273.15 - T0_prev) / (T0_curr - T0_prev) if T0_curr != T0_prev else 0.0
                t_ice = t_n + frac * dt

        if return_full:
            field[n + 1] = T

    qs_tau = -mat.k * (-3 * T[0] + 4 * T[1] - T[2]) / (2 * dz)

    out = {
        "Ts_t": Ts_t,
        "T_final": T,
        "z": z,
        "t": np.linspace(0.0, scn.tau, Nt + 1),
        "t_ice": t_ice,
        "Ts_tau": T[0],
        "qs_tau": qs_tau,
    }
    if return_full:
        out["T_field"] = field
    return out


# Convenient wrapper for sweeps over (V_wind, eps_atm, T_air_end).

def t_ice(
    V_wind: float,
    eps_atm: float,
    T_air_end: float,
    Nz: int = 32,
    Nt: int = 256,
    newton_tol: float = 1e-10,
) -> float:
    scn = PhysicalScenario(V_wind=V_wind, eps_atm=eps_atm, T_air_end=T_air_end)
    res = solve_physical(Nz, Nt, scn, newton_tol=newton_tol, return_full=False)
    return res["t_ice"]


if __name__ == "__main__":
    # Quick sanity check at HW4 baseline
    res = solve_physical(32, 256, PhysicalScenario())
    print(f"Nz=32 Nt=256: Ts(tau) = {res['Ts_tau']:.4f} K, "
          f"qs(tau) = {res['qs_tau']:.4f} W/m^2, t_ice = {res['t_ice']:.2f} s")
