#!/usr/bin/env python3
"""Direct test of solver to make sure it works."""

import numpy as np
from kdv_solver.solver import Grid, KdVProblem, PseudoSpectralSolver

def test_solver():
    """Test the solver with very conservative parameters."""
    print("Testing solver with conservative parameters...")

    # Very conservative parameters that should work
    N = 64
    L = 50.0
    kappa = 0.5
    dt = 0.00001  # Very small dt
    t_final = 0.01  # Very short time

    print(f"Using: N={N}, L={L}, kappa={kappa}, dt={dt}, t_final={t_final}")

    try:
        grid = Grid(N, L)
        problem = KdVProblem(grid, kappa)

        print("Created problem successfully")

        u_initial = problem.initial_condition()
        print(f"Initial condition max: {np.max(u_initial):.6f}")

        solver = PseudoSpectralSolver(problem, dt)
        print(f"Solver created, max_dt: {solver.max_dt:.6e}")

        # Test one step
        u_step = solver.step_rk4(u_initial)
        print(f"After one step max: {np.max(u_step):.6f}")

        # Test full solve
        u_final, times = solver.solve(t_final)
        print(f"Solve completed. Final max: {np.max(u_final):.6f}")
        print(f"Final time: {times[-1]:.6f}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_solver()