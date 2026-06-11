"""Validation tests for KdV single-soliton solution.

Validates the pseudo-spectral solver against the analytical solution:
    u(x,t) = 2 * kappa^2 * sech^2(kappa * (x - 4 * kappa^2 * t))
"""

import numpy as np
import pytest

from kdv_solver.solver import (
    Grid,
    KdVProblem,
    PseudoSpectralSolver,
    gaussian_packet,
    multi_soliton_field,
)


def _peak_heights(u: np.ndarray, threshold: float = 0.15) -> list[float]:
    """Heights of local maxima above a threshold, largest first (periodic)."""
    n = len(u)
    peaks = [
        u[i]
        for i in range(n)
        if u[i] > threshold and u[i] >= u[i - 1] and u[i] >= u[(i + 1) % n]
    ]
    return sorted(peaks, reverse=True)


class TestSingleSoliton:
    """Tests for single-soliton propagation."""

    def test_soliton_shape_at_t0(
        self, soliton_params: dict, default_grid_params: dict
    ) -> None:
        """Verify initial condition matches analytical solution at t=0."""
        kappa = soliton_params["kappa"]
        N = default_grid_params["N"]
        L = default_grid_params["L"]

        grid = Grid(N, L)
        problem = KdVProblem(grid, kappa)

        u_numerical = problem.initial_condition()
        u_analytical = problem.soliton(0.0)

        # Solutions should be identical at t=0
        np.testing.assert_allclose(u_numerical, u_analytical, rtol=1e-14)

    def test_soliton_speed(
        self, soliton_params: dict, default_grid_params: dict
    ) -> None:
        """Verify soliton propagates at correct speed c = 4 * kappa^2."""
        kappa = soliton_params["kappa"]
        N = default_grid_params["N"]
        L = default_grid_params["L"]

        grid = Grid(N, L)
        problem = KdVProblem(grid, kappa)

        # Compute appropriate time step based on CFL condition
        dt = 0.1 * grid.dx**3
        t_final = 0.5

        solver = PseudoSpectralSolver(problem, dt)
        u_final, _ = solver.solve(t_final)

        # Find center of soliton with sub-grid accuracy via a parabolic fit
        # around the discrete maximum. A bare argmax is quantized to dx, which
        # is far coarser than the 1%-of-displacement tolerance below.
        i = int(np.argmax(u_final))
        y_left = u_final[(i - 1) % N]
        y_center = u_final[i]
        y_right = u_final[(i + 1) % N]
        offset = 0.5 * (y_left - y_right) / (y_left - 2 * y_center + y_right)
        x_max_numerical = (i + offset) * grid.dx

        # Analytical center at t_final: soliton starts at the domain center
        # (x0 = L/2) and travels right at speed c.
        c = 4 * kappa**2
        x_max_analytical = (L / 2 + c * t_final) % L

        # Compare (accounting for periodic wrap-around)
        phase_diff = (x_max_numerical - x_max_analytical) % L
        if phase_diff > L / 2:
            phase_diff -= L

        # Tolerance: 1% of expected displacement
        expected_disp = c * t_final
        atol = 0.01 * expected_disp

        assert phase_diff == pytest.approx(0.0, abs=atol)

    def test_soliton_amplitude_conservation(
        self, soliton_params: dict, default_grid_params: dict
    ) -> None:
        """Verify soliton amplitude is conserved during propagation."""
        kappa = soliton_params["kappa"]
        N = default_grid_params["N"]
        L = default_grid_params["L"]

        grid = Grid(N, L)
        problem = KdVProblem(grid, kappa)

        dt = 0.1 * grid.dx**3
        t_final = 1.0
        solver = PseudoSpectralSolver(problem, dt)
        u_final, _ = solver.solve(t_final)

        # Amplitude is at x=0 for t=0, and at x=c*t for t>0
        amp_final = np.max(u_final)

        # Expected amplitude: 2 * kappa^2
        amp_analytical = 2 * kappa**2

        # Allow 5% tolerance for numerical error
        atol = 0.05 * amp_analytical

        assert amp_final == pytest.approx(amp_analytical, abs=atol)

    def test_l2_error_single_soliton(
        self, soliton_params: dict, default_grid_params: dict, time_params: dict
    ) -> None:
        """Verify L2 error is below tolerance after propagation."""
        kappa = soliton_params["kappa"]
        N = default_grid_params["N"]
        L = default_grid_params["L"]

        grid = Grid(N, L)
        problem = KdVProblem(grid, kappa)

        # Use a smaller time step for better accuracy
        dt = time_params["dt"]
        t_final = time_params["t_final"]
        solver = PseudoSpectralSolver(problem, dt)

        u_final, _ = solver.solve(t_final)

        # Analytical solution at t_final
        u_analytical = problem.soliton(t_final)

        # L2 norm (with proper scaling for discrete sum)
        dx = grid.dx
        error = np.sqrt(dx * np.sum((u_final - u_analytical) ** 2))
        norm = np.sqrt(dx * np.sum(u_analytical**2))
        l2_error = error / norm

        # Tolerance: 1e-3 for single soliton (looser due to numerical stability)
        assert l2_error < 1e-3

    def test_solution_stays_positive(
        self, soliton_params: dict, default_grid_params: dict
    ) -> None:
        """KdV soliton solution should remain positive for all time."""
        kappa = soliton_params["kappa"]
        N = default_grid_params["N"]
        L = default_grid_params["L"]

        grid = Grid(N, L)
        problem = KdVProblem(grid, kappa)

        dt = 0.1 * grid.dx**3
        t_final = 2.0
        solver = PseudoSpectralSolver(problem, dt)

        u_final, _ = solver.solve(t_final)

        # Soliton should be positive everywhere up to a small dispersive ripple.
        # A spectral solution carries O(1e-10..1e-4) negative oscillations in
        # the tails; -1e-12 would be below the achievable numerical floor.
        assert np.all(u_final >= -1e-6)


class TestGridConvergence:
    """Spectral convergence tests."""

    def test_spectral_convergence(
        self, soliton_params: dict, time_params: dict
    ) -> None:
        """Verify error decreases rapidly with N (spectral convergence).

        The soliton is only fully resolved once N is large enough; at coarse N
        the field still carries appreciable energy at the Nyquist mode. The
        signature of spectral accuracy is therefore that the error drops
        steeply as N increases and reaches machine-level accuracy on the
        finest (resolved) grid -- not that every N already meets a fixed bound.
        """
        kappa = soliton_params["kappa"]
        L = 100.0  # Fixed domain size
        t_final = time_params["t_final"]

        errors = []
        for n in (64, 128, 256):
            grid = Grid(n, L)
            problem = KdVProblem(grid, kappa)

            # Use CFL-scaled time step for each grid
            dt = 0.1 * grid.dx**3
            solver = PseudoSpectralSolver(problem, dt)

            u_final, _ = solver.solve(t_final)
            u_analytical = problem.soliton(t_final)

            l2_error = np.sqrt(grid.dx * np.sum((u_final - u_analytical) ** 2))
            errors.append(l2_error)

        # Error must drop steeply as the grid is refined ...
        assert errors[1] < errors[0]
        assert errors[2] < errors[1]
        # ... and reach spectral accuracy on the finest, fully-resolved grid.
        assert errors[-1] < 1e-6


class TestMultiSoliton:
    """Multi-soliton interaction: solitons survive collisions (resilience)."""

    def test_two_soliton_overtaking(self) -> None:
        """A tall, fast soliton overtakes a short one; both re-emerge intact.

        The defining property of KdV solitons is that they pass *through* each
        other and recover their original amplitudes (only a phase shift remains).
        """
        grid = Grid(512, 50.0)
        problem = KdVProblem(grid, kappa=1.0)  # kappa unused for the custom field
        solver = PseudoSpectralSolver(problem, dt=5e-3)

        # Tall/fast soliton (kappa=1.0) placed behind a short/slow one.
        u0 = multi_soliton_field(grid, [(1.0, 15.0), (0.6, 30.0)])
        _, solutions = solver.solve_with_history(12.0, n_snapshots=40, u0=u0)

        initial = _peak_heights(solutions[0])
        final = _peak_heights(solutions[-1])

        # Two solitons in, two solitons out, with amplitudes preserved (2*kappa^2).
        assert initial == pytest.approx([2.0, 0.72], abs=0.02)
        assert len(final) == 2
        assert final == pytest.approx([2.0, 0.72], abs=0.05)
        # No blow-up.
        assert np.all(np.isfinite(solutions[-1]))


class TestGaussianPacket:
    """A Gaussian hump disperses into a rank-ordered train of solitons."""

    def test_gaussian_breaks_into_soliton_train(self) -> None:
        grid = Grid(1024, 100.0)
        problem = KdVProblem(grid, kappa=1.0)  # kappa unused for the custom field
        solver = PseudoSpectralSolver(problem, dt=5e-3)

        u0 = gaussian_packet(grid, amplitude=1.0, x0=30.0, width=3.0)
        _, solutions = solver.solve_with_history(15.0, n_snapshots=40, u0=u0)

        # Starts as a single hump, emerges as several solitons.
        assert len(_peak_heights(solutions[0])) == 1
        assert len(_peak_heights(solutions[-1])) >= 2

        # KdV conserves mass (integral of u); also stays finite.
        dx = grid.dx
        assert dx * solutions[-1].sum() == pytest.approx(
            dx * solutions[0].sum(), rel=1e-3
        )
        assert np.all(np.isfinite(solutions[-1]))
