# Black-Ice-Risk-VVUQ

Verification and Validation of a 1D Transient Heat Conduction Solver for Black-Ice Risk Prediction in Autonomous Vehicle Braking Safety.

**Course:** AOE/CS/ME 6444 — Verification and Validation in Scientific Computing, Spring 2026

**Instructor:** Dr. Chris Roy, Virginia Tech

**Author:** Dawood Wasif

---

## Project Overview

This project implements and verifies a one-dimensional transient heat conduction solver that predicts road surface temperature to assess black-ice formation risk. The thermal model solves:

$$\rho \, c_p \, \frac{\partial T}{\partial t} = k \, \frac{\partial^2 T}{\partial z^2}$$

through a vertical road column (asphalt + subgrade) and couples with the CARLA autonomous driving simulator to evaluate emergency braking distances under icy conditions.

**Numerical method:**
- Spatial: 2nd-order central finite differences $$O(\Delta z^2)$$
- Temporal: Crank–Nicolson (implicit trapezoidal) $$O(\Delta t^2)$$
- Linear solver: Thomas algorithm (tridiagonal)
- Formal order of accuracy: $$p = 2$$

---

## Repository Structure

```
Black-Ice-Risk-VVUQ/
├── vvsc_heat1d_solver.py   # Reusable 1D heat equation solver module
├── vvsc_hw3_mms.py         # HW3: MMS code verification driver
├── requirements.txt        # Python dependencies
├── hw3_output/             # Generated figures and results
│   ├── fig01_manufactured_solution.png
│   ├── fig02_source_term.png
│   ├── fig03_error_norms.png
│   ├── fig04_observed_order.png
│   ├── fig05_local_error.png
│   ├── fig06_local_error_spacetime.png
│   ├── fig07_solution_comparison.png
│   ├── fig08_srq_convergence.png
│   ├── fig09_roundoff_assessment.png
│   └── verification_summary.json
└── README.md
```

---

## Setup

### 1. Create Conda Environment

```bash
conda create -n vvuq python=3.11 -y
conda activate vvuq
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy matplotlib sympy
```

---

## Usage

### Run MMS Code Verification (HW3)

```bash
conda activate vvuq
python vvsc_hw3_mms.py
```

This will:

1. Verify the MMS source term against SymPy symbolic differentiation
2. Run the grid convergence study on 6 systematically refined meshes ($$r = 2$$)
3. Perform SRQ (System Response Quantity) convergence analysis
4. Generate 9 publication-quality figures in `hw3_output/`
5. Export a `verification_summary.json` with all numerical results

### Use the Solver Module Independently

```python
from vvsc_heat1d_solver import solve_heat_1d_CN
import numpy as np

result = solve_heat_1d_CN(
    Nz=128, Nt=256,
    L=0.5, t_final=3600.0,
    rho=2100.0, cp=920.0, k=1.2,
    T_init_fn=lambda z: 280.0 * np.ones_like(z),
    T_left_fn=lambda t: 280.0,
    T_right_fn=lambda t: 285.0,
    source_fn=None,
    store_history=False,
)

print(result['T_final'])  # Temperature profile at t = 3600 s
```

---

## HW3 Results Summary

The code **passes** the order-of-accuracy test. Observed orders on the finest mesh pair:

| Norm | Observed Order $$\hat{p}$$ | Formal Order |
|------|---------------------------|--------------|
| $$L_1$$ | 2.0054 | 2 |
| $$L_2$$ | 2.0027 | 2 |
| $$L_\infty$$ | 1.9997 | 2 |

Round-off error is negligible (discretisation error dominates by $$\sim 10^6 \times$$).

---

## References

1. Oberkampf, W. L. & Roy, C. J. (2025). *Verification and Validation in Scientific Computing*, 2nd ed., Cambridge University Press.
2. Roy, C. J. (2005). Review of code and solution verification procedures for computational simulation. *J. Comput. Phys.*, 205(1), 131–156.
3. Roache, P. J. (2002). Code verification by the method of manufactured solutions. *ASME J. Fluids Eng.*, 124(1), 4–10.
