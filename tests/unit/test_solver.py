"""Unit tests for the KdV solver building blocks.

Covers grid validation, FFT-based spatial derivatives, and the snapshot-
recording integration helper used by the visualization layer.
"""

import numpy as np
import pytest

from kdv_solver.solver import (
    Grid,
    KdVProblem,
    PseudoSpectralSolver,
    conserved_quantities,
    cosine_wave,
    gaussian_packet,
    mode_energies,
    multi_soliton_field,
    predict_soliton_amplitudes,
    soliton_profile,
)


class TestGridValidation:
    """Grid parameter validation."""

    def test_rejects_nonpositive_n(self) -> None:
        with pytest.raises(ValueError, match="N must be positive"):
            Grid(0, 100.0)

    def test_rejects_nonpositive_l(self) -> None:
        with pytest.raises(ValueError, match="L must be positive"):
            Grid(64, 0.0)


class TestSpectralDerivatives:
    """FFT-based derivatives against an exact trigonometric reference."""

    def _problem(self) -> tuple[KdVProblem, np.ndarray]:
        # Domain [0, 2*pi) so integer wave numbers are represented exactly.
        grid = Grid(64, 2 * np.pi)
        problem = KdVProblem(grid, kappa=1.0)
        return problem, grid.x

    def test_compute_u_x(self) -> None:
        problem, x = self._problem()
        m = 3  # integer wave number -> exactly representable on the grid
        u = np.sin(m * x)
        expected = m * np.cos(m * x)
        np.testing.assert_allclose(problem.compute_u_x(u), expected, atol=1e-12)

    def test_compute_u_xxx(self) -> None:
        problem, x = self._problem()
        m = 3
        u = np.sin(m * x)
        expected = -(m**3) * np.cos(m * x)
        # The k**3 factor amplifies round-off, so use a looser absolute tol.
        np.testing.assert_allclose(problem.compute_u_xxx(u), expected, atol=1e-9)


class TestSolveWithHistory:
    """The snapshot-recording solve used for visualization."""

    def test_history_shape_and_endpoints(self) -> None:
        grid = Grid(256, 100.0)
        problem = KdVProblem(grid, kappa=0.5)
        solver = PseudoSpectralSolver(problem, dt=1e-3)

        times, solutions = solver.solve_with_history(1.0, n_snapshots=10)

        # times and solutions are aligned, monotonically increasing in time.
        assert len(times) == len(solutions)
        assert times[0] == 0.0
        assert times[-1] == pytest.approx(1.0)
        assert np.all(np.diff(times) > 0)

        # First snapshot is the initial condition.
        np.testing.assert_allclose(solutions[0], problem.initial_condition())

        # Amplitude is conserved by the (correct) integrator.
        assert solutions[-1].max() == pytest.approx(solutions[0].max(), rel=1e-2)

    def test_single_step_history(self) -> None:
        # t_final smaller than dt still yields at least one integration step.
        grid = Grid(128, 100.0)
        problem = KdVProblem(grid, kappa=0.5)
        solver = PseudoSpectralSolver(problem, dt=1.0)

        times, solutions = solver.solve_with_history(0.1, n_snapshots=50)

        assert len(solutions) == 2  # initial condition + single step
        assert times[-1] == pytest.approx(0.1)

    def test_history_accepts_custom_u0(self) -> None:
        # A custom initial field (e.g. multi-soliton) is used verbatim.
        grid = Grid(256, 100.0)
        problem = KdVProblem(grid, kappa=0.5)
        solver = PseudoSpectralSolver(problem, dt=1e-3)

        u0 = multi_soliton_field(grid, [(0.6, 30.0), (0.4, 70.0)])
        _, solutions = solver.solve_with_history(0.5, n_snapshots=5, u0=u0)

        np.testing.assert_allclose(solutions[0], u0)


class TestSolitonProfiles:
    """Single- and multi-soliton field builders."""

    def test_profile_position_and_amplitude(self) -> None:
        grid = Grid(512, 100.0)
        u = soliton_profile(grid, kappa=0.7, x0=40.0, t=0.0)
        # Amplitude is 2*kappa^2 (up to grid sampling) and the peak sits at x0.
        assert u.max() == pytest.approx(2 * 0.7**2, rel=1e-2)
        assert grid.x[int(np.argmax(u))] == pytest.approx(40.0, abs=grid.dx)

    def test_multi_soliton_is_superposition(self) -> None:
        grid = Grid(512, 100.0)
        specs = [(0.8, 25.0), (0.5, 60.0)]
        field = multi_soliton_field(grid, specs)
        expected = soliton_profile(grid, 0.8, 25.0) + soliton_profile(grid, 0.5, 60.0)
        np.testing.assert_allclose(field, expected)

    def test_gaussian_packet(self) -> None:
        grid = Grid(512, 100.0)
        u = gaussian_packet(grid, amplitude=1.5, x0=40.0, width=4.0)
        assert u.max() == pytest.approx(1.5, rel=1e-3)
        assert grid.x[int(np.argmax(u))] == pytest.approx(40.0, abs=grid.dx)
        assert np.all(u >= 0)

    def test_cosine_wave(self) -> None:
        grid = Grid(512, 20.0)
        u = cosine_wave(grid, amplitude=0.9, mode=1)
        assert u.max() == pytest.approx(0.9, rel=1e-3)
        assert u.mean() == pytest.approx(0.0, abs=1e-12)  # zero-mean
        assert u[0] == pytest.approx(0.9)  # cosine peaks at x=0 for mode 1


class TestConservedQuantities:
    """The first three KdV invariants."""

    def test_single_soliton_values(self) -> None:
        # For a single soliton: mass = integral 2k^2 sech^2(k x) dx = 4*kappa.
        grid = Grid(512, 100.0)
        problem = KdVProblem(grid, kappa=0.5)
        mass, momentum, energy = conserved_quantities(
            problem, problem.initial_condition()
        )
        assert mass == pytest.approx(4 * 0.5, rel=1e-4)
        assert momentum > 0  # integral u^2

    def test_invariants_are_conserved(self) -> None:
        grid = Grid(512, 50.0)
        problem = KdVProblem(grid, kappa=1.0)
        solver = PseudoSpectralSolver(problem, dt=2e-3)
        u0 = multi_soliton_field(grid, [(1.0, 15.0), (0.6, 30.0)])
        _, solutions = solver.solve_with_history(8.0, n_snapshots=20, u0=u0)

        first = conserved_quantities(problem, solutions[0])
        last = conserved_quantities(problem, solutions[-1])
        for i0, i1 in zip(first, last):
            assert i1 == pytest.approx(i0, rel=1e-4, abs=1e-6)


class TestInverseScattering:
    """Predicting emergent solitons from the initial condition."""

    def test_single_soliton_one_bound_state(self) -> None:
        grid = Grid(512, 80.0)
        problem = KdVProblem(grid, kappa=0.8)
        amps = predict_soliton_amplitudes(grid, problem.initial_condition())
        assert len(amps) == 1
        assert amps[0] == pytest.approx(2 * 0.8**2, rel=0.05)

    def test_two_solitons_predicted(self) -> None:
        grid = Grid(1024, 80.0)
        u0 = multi_soliton_field(grid, [(1.0, 20.0), (0.6, 50.0)])
        amps = predict_soliton_amplitudes(grid, u0)
        assert len(amps) == 2
        assert amps == pytest.approx([2.0, 0.72], abs=0.05)


class TestModeEnergies:
    """Fourier mode-energy decomposition."""

    def test_cosine_energy_in_first_mode(self) -> None:
        grid = Grid(256, 20.0)
        e = mode_energies(cosine_wave(grid, 0.9, mode=1), n_modes=6)
        assert len(e) == 6
        # All the energy is in mode 1 for a single cosine.
        assert e[0] > 0
        assert np.all(e[1:] < 1e-6 * e[0])
