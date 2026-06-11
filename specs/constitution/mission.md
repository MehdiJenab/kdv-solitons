# Mission

## What

A pseudo-spectral solver for the Korteweg-de Vries (KdV) equation:

```
u_t + 6 u u_x + u_xxx = 0
```

This nonlinear PDE describes weakly nonlinear long waves in shallow water, plasma physics, and other dispersive media. The solver uses pseudo-spectral methods: FFT for spatial derivatives and exponential time differencing for time integration.

## Why

The KdV equation has exact soliton solutions - stable, localized wave packets that maintain their shape while propagating. The single-soliton solution is:

```
u(x,t) = 2κ² sech²(κ(x - 4κ²t))
```

This provides a rigorous validation test for any numerical solver. The pseudo-spectral method is particularly well-suited for this problem due to its excellent phase accuracy for smooth solutions.

## Who

Researchers and students in applied mathematics, physics, and engineering who need to simulate shallow water waves, plasma oscillations, or other dispersive wave phenomena.

## Scale

- 1D domain with periodic boundary conditions
- Grid sizes: 64 to 4096 points
- Time integration up to t_max where multiple solitons may interact
- Single-precision acceptable for validation; double for production

## Goals

1. Implement a working pseudo-spectral KdV solver
2. Validate against single-soliton analytical solution
3. Achieve spectral accuracy (error decreases exponentially with grid resolution)
4. Provide clean, maintainable code for educational and research use
