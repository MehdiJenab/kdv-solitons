#!/usr/bin/env python3
"""Test script to validate the KdV solver implementation with better parameters."""

import numpy as np
from kdv_solver.solver import Grid, KdVProblem, PseudoSpectralSolver

def main():
    print("Testing KdV solver with adjusted parameters...")

    # Use parameters that are more suitable for the test
    N = 256
    L = 100.0
    kappa = 0.5
    dt = 0.0001  # Much smaller time step
    t_final = 0.5

    print(f"N={N}, L={L}, kappa={kappa}, dt={dt}, t_final={t_final}")
    print(f"dx = {L/N}")

    # Setup
    grid = Grid(N, L)
    problem = KdVProblem(grid, kappa)

    # Check initial condition
    u_initial = problem.initial_condition()
    u_analytical = problem.soliton(0.0)

    print(f"Initial condition matches analytical at t=0: {np.allclose(u_initial, u_analytical)}")
    print(f"Max initial amplitude: {np.max(u_initial)}")
    print(f"Expected amplitude: {2 * kappa**2}")

    # Try to run solver with a smaller time step
    try:
        solver = PseudoSpectralSolver(problem, dt)
        print(f"Max dt allowed: {solver.max_dt:.2e}")

        u_final, times = solver.solve(t_final)

        print(f"Final time reached: {times[-1]}")
        print(f"Final max amplitude: {np.max(u_final)}")
        print(f"Analytical max amplitude: {2 * kappa**2}")

        # Check center position
        x_max_numerical = grid.x[np.argmax(u_final)]
        c = 4 * kappa**2
        x_max_analytical = c * t_final

        print(f"Numerical center position: {x_max_numerical}")
        print(f"Analytical center position: {x_max_analytical}")

        # Phase difference
        phase_diff = (x_max_numerical - x_max_analytical) % L
        if phase_diff > L / 2:
            phase_diff -= L

        print(f"Phase difference: {phase_diff}")
        print(f"Expected displacement: {c * t_final}")

        # Calculate L2 error
        u_analytical_final = problem.soliton(t_final)
        dx = grid.dx
        error = np.sqrt(dx * np.sum((u_final - u_analytical_final) ** 2))
        norm = np.sqrt(dx * np.sum(u_analytical_final ** 2))
        l2_error = error / norm
        print(f"L2 error: {l2_error}")

    except Exception as e:
        print(f"Error during solving: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()