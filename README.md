# soliton-lab

An interactive simulator and pseudo-spectral solver for the **Korteweg–de Vries (KdV) equation** — the canonical model of nonlinear dispersive waves and the birthplace of the *soliton*.

Build an initial condition from solitons and Gaussian packets, preview it, then watch it evolve: solitons travel at amplitude-dependent speeds, pass *through* each other unchanged, and a smooth hump shatters into a rank-ordered train of solitons plus a dispersive tail.

$$u_t + 6\,u\,u_x + u_{xxx} = 0$$

---

## What it does

- **Single solitons** — `u(x,t) = 2\kappa^2\,\mathrm{sech}^2\!\big(\kappa(x - x_0 - 4\kappa^2 t)\big)`, where one parameter `κ` fixes amplitude (`2κ²`), speed (`4κ²`), and width (`∝1/κ`).
- **Multi-soliton interaction** — add several solitons; taller (faster) ones overtake shorter ones and re-emerge with their original shapes (only a phase shift), the defining property of solitons.
- **Gaussian packets** — a Gaussian is *not* a KdV eigenstate, so it disperses into a train of solitons (the leading one can be taller than the original hump) plus an oscillatory tail.
- **Superpositions** — mix any number of solitons and Gaussians in the same domain and watch them interact.
- **Periodic domain** — waves that leave one side re-enter the other; the simulator auto-sizes the final time to one full lap of the slowest component.

### Interactive web app

- Live **preview of the initial condition** at `t=0` before you run.
- Add/remove solitons and Gaussians, each with stepper arrows.
- Amplitude-aware time window, fixed y-axis (no jitter), clean axis labels.
- Smooth playback of the time evolution.

---

## The physics, briefly — why this is interesting

In 1953–55 at Los Alamos, **Fermi, Pasta, Ulam and Tsingou** simulated a chain of nonlinearly coupled masses, expecting the energy to thermalize across all modes. Instead it nearly returned to its initial state — the **FPUT recurrence**, a famous "paradox." In 1965 **Zabusky and Kruskal** showed the continuum limit of that chain is the KdV equation, that its localized waves pass through each other intact, and coined the term **soliton**. The recurrence is the solitons re-aligning on the periodic domain. KdV is *integrable* — it has infinitely many conserved quantities — which is why it refuses to thermalize.

This project lets you watch those phenomena directly.

---

## Numerical method

The solver is **pseudo-spectral in space** (FFT-based derivatives) and uses **ETDRK4** (exponential time-differencing Runge–Kutta, 4th order; Cox & Matthews 2002, Kassam & Trefethen 2005) in time.

Two points are essential and easy to get wrong:

- **The stiff linear term `u_xxx` is integrated exactly in Fourier space.** A naive explicit RK4 on `u_xxx` has eigenvalues `∝ i k³` and is violently unstable, forcing absurdly small time steps. ETDRK4 treats the linear part exactly and only the nonlinear term explicitly, removing the `dt ∝ dx³` barrier.
- **The ETD coefficient (φ-) functions are evaluated by a complex contour integral.** Because `L = i k³` is purely *imaginary*, the φ-functions are complex-valued, so the contour spans the full circle (the real-axis symmetry shortcut used for dissipative PDEs gives the wrong, only first-order, answer here).

The result is clean 4th-order-in-time, spectral-in-space accuracy: a single soliton matches the analytical solution to `~1e-10` when resolved.

---

## Install

Requires Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Run the web app

```bash
python simple_web_app.py
# open http://localhost:5001
```

Choose your initial condition (add solitons and/or Gaussians), check the live preview, then **Run Simulation**.

## Use as a library

```python
import numpy as np
from kdv_solver.solver import (
    Grid, KdVProblem, PseudoSpectralSolver,
    soliton_profile, multi_soliton_field, gaussian_packet,
)

grid = Grid(N=512, L=100.0)
problem = KdVProblem(grid, kappa=0.5)
solver = PseudoSpectralSolver(problem, dt=1e-3)

# Single soliton -> (u_final, times)
u_final, times = solver.solve(t_final=10.0)

# Custom initial field (multi-soliton or Gaussian) with snapshots for animation
u0 = multi_soliton_field(grid, [(1.0, 15.0), (0.6, 30.0)])   # tall/fast behind short/slow
# u0 = gaussian_packet(grid, amplitude=1.0, x0=30.0, width=3.0)
times, frames = solver.solve_with_history(t_final=12.0, n_snapshots=50, u0=u0)
```

---

## Project layout

```
src/kdv_solver/solver.py   # Grid, KdVProblem, PseudoSpectralSolver + field builders
templates/index.html       # interactive web UI (Chart.js)
simple_web_app.py          # Flask app exposing /api/simulate
tests/                     # unit + feature (validation) tests
specs/                     # project constitution: mission, architecture, validation contract
```

## Development

```bash
pytest                 # run tests (100% coverage enforced)
ruff check --fix .     # lint
ruff format .          # format
mypy src/              # type-check
```

The test suite validates the solver against the analytical soliton (shape, speed, amplitude conservation, L2 error, spectral convergence) and the nonlinear physics (two-soliton resilience, Gaussian breakup, mass conservation).

---

## References

- N. J. Zabusky & M. D. Kruskal, *Interaction of "solitons" in a collisionless plasma and the recurrence of initial states*, Phys. Rev. Lett. **15**, 240 (1965).
- A.-K. Kassam & L. N. Trefethen, *Fourth-order time-stepping for stiff PDEs*, SIAM J. Sci. Comput. **26**, 1214 (2005).
- S. M. Cox & P. C. Matthews, *Exponential time differencing for stiff systems*, J. Comput. Phys. **176**, 430 (2002).
- E. Fermi, J. Pasta, S. Ulam (& M. Tsingou), *Studies of nonlinear problems*, Los Alamos report LA-1940 (1955).
