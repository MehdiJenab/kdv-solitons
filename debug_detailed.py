#!/usr/bin/env python3
"""Detailed debug of KdV solver."""

import numpy as np
from kdv_solver.solver import Grid, KdVProblem, PseudoSpectralSolver

def main():
    print("Detailed KdV solver debugging...")

    # Use the exact parameters from the failing test
    N = 256
    L = 100.0
    kappa = 0.5
    dt = 0.1 * (L/N)**3  # This is what the test uses: 0.1 * dx**3
    t_final = 0.5

    print(f"Parameters:")
    print(f"  N={N}, L={L}, kappa={kappa}")
    print(f"  dx = {L/N:.6f}")
    print(f"  dt = {dt:.6f}")
    print(f"  max_dt = {0.05 * (L/N)**3:.6f}")
    print(f"  t_final = {t_final}")

    # Setup
    grid = Grid(N, L)
    problem = KdVProblem(grid, kappa)

    # Check initial condition
    u_initial = problem.initial_condition()
    u_analytical = problem.soliton(0.0)

    print(f"\nInitial condition verification:")
    print(f"  Matches analytical at t=0: {np.allclose(u_initial, u_analytical)}")
    print(f"  Max initial amplitude: {np.max(u_initial):.6f}")
    print(f"  Expected amplitude: {2 * kappa**2:.6f}")

    # Try to run solver step by step
    try:
        solver = PseudoSpectralSolver(problem, dt)
        print(f"\nSolver created successfully")
        print(f"  max_dt = {solver.max_dt:.6f}")

        # Check the first few steps manually
        u_n = problem.initial_condition()
        print(f"\nStep-by-step testing:")
        print(f"  Initial u max: {np.max(u_n):.6f}")

        # Test the RHS function
        rhs_result = solver.rhs(u_n)
        print(f"  RHS at t=0: max={np.max(rhs_result):.6f}, min={np.min(rhs_result):.6f}")

        # Test one RK4 step
        u_next = solver.step_rk4(u_n)
        print(f"  After one RK4 step: max={np.max(u_next):.6f}")

        # Try a few more steps
        for i in range(5):
            u_n = solver.step_rk4(u_n)
            print(f"  Step {i+1}: max={np.max(u_n):.6f}")

        print("\nFull solve...")
        u_final, times = solver.solve(t_final)

        print(f"Final result:")
        print(f"  Final time reached: {times[-1]:.6f}")
        print(f"  Final max amplitude: {np.max(u_final):.6f}")
        print(f"  Expected amplitude: {2 * kappa**2:.6f}")

        # Check center position
        x_max_numerical = grid.x[np.argmax(u_final)]
        c = 4 * kappa**2
        x_max_analytical = c * t_final

        print(f"  Numerical center position: {x_max_numerical:.6f}")
        print(f"  Analytical center position: {x_max_analytical:.6f}")

        # Phase difference
        phase_diff = (x_max_numerical - x_max_analytical) % L
        if phase_diff > L / 2:
            phase_diff -= L

        print(f"  Phase difference: {phase_diff:.6f}")

        # Calculate L2 error
        u_analytical_final = problem.soliton(t_final)
        dx = grid.dx
        error = np.sqrt(dx * np.sum((u_final - u_analytical_final) ** 2))
        norm = np.sqrt(dx * np.sum(u_analytical_final ** 2))
        l2_error = error / norm
        print(f"  L2 error: {l2_error:.6f}")

    except Exception as e:
        print(f"Error during solving: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()