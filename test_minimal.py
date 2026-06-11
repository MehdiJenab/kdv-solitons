#!/usr/bin/env python3
"""Simple test of the solver with the exact same parameters as tests."""

import numpy as np
from kdv_solver.solver import Grid, KdVProblem, PseudoSpectralSolver

def test_simple_case():
    """Test with minimal parameters to isolate the problem."""
    print("Testing simple case...")

    # Exact parameters from test
    N = 256
    L = 100.0
    kappa = 0.5
    dt = 0.1 * (L/N)**3  # This is what test uses
    t_final = 0.5

    print(f"Using parameters: N={N}, L={L}, kappa={kappa}, dt={dt}, t_final={t_final}")

    # Create problem
    grid = Grid(N, L)
    problem = KdVProblem(grid, kappa)

    # Test initial condition
    u0 = problem.initial_condition()
    print(f"Initial condition max: {np.max(u0)}")

    # Test derivatives
    u_x = problem.compute_u_x(u0)
    u_xxx = problem.compute_u_xxx(u0)
    print(f"u_x max: {np.max(u_x)}, min: {np.min(u_x)}")
    print(f"u_xxx max: {np.max(u_xxx)}, min: {np.min(u_xxx)}")

    # Test RHS
    solver = PseudoSpectralSolver(problem, dt)
    rhs = solver.rhs(u0)
    print(f"RHS max: {np.max(rhs)}, min: {np.min(rhs)}")

    print("All basic tests passed!")

if __name__ == "__main__":
    test_simple_case()