# Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     kdv_solver.py                           │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               KdVProblem                                │  │
│  │  - Grid setup (N, L, dx, k_wave)                       │  │
│  │  - Initial condition (soliton, multi-soliton, etc.)    │  │
│  │  - Analytical solution evaluator                       │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               PseudoSpectralSolver                      │  │
│  │  - FFT-based spatial derivative computation            │  │
│  │  - Exponential time differencing (ETD)                 │  │
│  │  - Time stepping loop                                  │  │
│  └───────────────────────────────────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────┐  │
│  │               Validation                                │  │
│  │  - Error norms (L2, L∞)                                │  │
│  │  - Convergence analysis                                │  │
│  │  - Visualization                                       │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Data Flow

1. **Initialization**: Create uniform grid with periodic boundary conditions
2. **Fourier space**: Compute wave numbers for FFT-based derivatives
3. **Initial condition**: Evaluate analytical soliton at t=0
4. **Time stepping**: For each time step:
   - FFT: transform u → û (Fourier space)
   - Apply linear operator e^(AΔt) for u_xxx term
   - Compute nonlinear term 6uu_x in physical space
   - Apply integrating factor for nonlinear contribution
   - FFT back: û → u (physical space)
5. **Validation**: Compare with analytical solution at each output time

## Core Components

### Grid (KdVProblem)
- Uniform grid: x_j = j·Δx, j = 0, ..., N-1
- Periodic: u_N = u_0
- Wave numbers: k_m = 2πm/L for m = 0, ..., N/2

### Spatial Derivatives (PseudoSpectralSolver)
- u_x → i·k·û(k) in Fourier space
- u_xxx → (i·k)³·û(k) = -i·k³·û(k)
- FFT/IFFT for transform pair

### Time Integration
- Exponential Time Differencing (ETD) or
- Runge-Kutta4 with FFT-based derivatives
- Adaptive or fixed time step based on CFL condition

## Scaling Plan

- **Weak scaling**: Fix grid spacing, increase domain size
- **Strong scaling**: Fix domain size, refine grid
- Expected: spectral convergence for smooth solutions
