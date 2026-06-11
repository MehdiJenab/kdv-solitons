#!/usr/bin/env python3
"""Test that the solver works with web interface parameters."""

import numpy as np
from kdv_solver.solver import Grid, KdVProblem, PseudoSpectralSolver

def test_web_parameters():
    """Test with parameters that should work for web interface."""
    print("Testing solver with web interface parameters...")

    # Parameters matching typical web interface settings
    N = 256
    L = 100.0
    kappa = 0.5
    dt = 0.0001  # Very small time step for stability
    t_final = 0.1  # Short time for testing

    print(f"Parameters: N={N}, L={L}, kappa={kappa}, dt={dt}, t_final={t_final}")

    try:
        # Setup
        grid = Grid(N, L)
        problem = KdVProblem(grid, kappa)

        # Test initial condition
        u_initial = problem.initial_condition()
        print(f"Initial condition max: {np.max(u_initial):.6f}")

        # Create solver with very conservative dt
        solver = PseudoSpectralSolver(problem, dt)
        print(f"Max dt allowed: {solver.max_dt:.6e}")

        # Test the RHS function
        rhs = solver.rhs(u_initial)
        print(f"RHS max: {np.max(rhs):.6f}")

        # Run one step to make sure it works
        u_next = solver.step_rk4(u_initial)
        print(f"After one step max: {np.max(u_next):.6f}")

        # Run full solve
        u_final, times = solver.solve(t_final)
        print(f"Solved successfully. Final max: {np.max(u_final):.6f}")
        print(f"Final time: {times[-1]:.6f}")

        return True

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_web_parameters()
    if success:
        print("✓ Solver test passed!")
    else:
        print("✗ Solver test failed!")