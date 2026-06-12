"""KdV solver module.

Pseudo-spectral solver for the KdV equation:
    u_t + 6 u u_x + u_xxx = 0

The stiff linear term ``u_xxx`` is integrated *exactly* in Fourier space and
the nonlinear term ``6 u u_x`` is advanced with a fourth-order exponential
time-differencing Runge-Kutta scheme (ETDRK4, Cox & Matthews 2002; Kassam &
Trefethen 2005).  This removes the explicit ``dt ~ dx**3`` stability barrier
that makes a naive explicit RK4 integrator blow up for KdV.
"""

from dataclasses import dataclass

import numpy as np

# Number of points used for the complex contour integral that evaluates the
# ETDRK4 coefficient functions without cancellation error near LR = 0.
_CONTOUR_POINTS = 32


def soliton_profile(
    grid: "Grid", kappa: float, x0: float, t: float = 0.0
) -> np.ndarray:
    """
    Evaluate one periodic KdV soliton on the grid.

    u(x,t) = 2 * kappa^2 * sech^2(kappa * (x - x0 - 4 * kappa^2 * t))

    The argument is wrapped into ``[-L/2, L/2)`` about the (moving) center so
    the profile is smooth and periodic on the domain.

    Args:
        grid: Discretization grid
        kappa: Soliton parameter (amplitude 2*kappa^2, speed 4*kappa^2)
        x0: Center of the soliton at t=0
        t: Time

    Returns:
        u(x, t) evaluated on the grid
    """
    c = 4 * kappa**2  # soliton speed
    center = x0 + c * t
    x_shifted = (grid.x - center + grid.L / 2) % grid.L
    x_shifted -= grid.L / 2
    return 2 * kappa**2 / np.cosh(kappa * x_shifted) ** 2


def multi_soliton_field(
    grid: "Grid", solitons: list[tuple[float, float]]
) -> np.ndarray:
    """
    Build a t=0 initial condition by superposing several solitons.

    Each entry is a ``(kappa, x0)`` pair. The superposition is only an exact
    KdV solution when the solitons are well separated, but the solver then
    evolves the true nonlinear interaction (taller, faster solitons overtake
    and pass through shorter ones, re-emerging with their original shape).

    Args:
        grid: Discretization grid
        solitons: List of ``(kappa, x0)`` pairs

    Returns:
        Superposed field on the grid
    """
    field = np.zeros(grid.N)
    for kappa, x0 in solitons:
        field = field + soliton_profile(grid, kappa, x0, 0.0)
    return field


def gaussian_packet(
    grid: "Grid", amplitude: float, x0: float, width: float
) -> np.ndarray:
    """
    Build a Gaussian initial condition: ``A * exp(-(x - x0)^2 / (2 * width^2))``.

    A Gaussian is not a KdV eigenstate, so under the dynamics it breaks up into
    a rank-ordered train of solitons plus a dispersive (oscillatory) tail --
    a good demonstration of the inverse-scattering structure of KdV. The
    distance to the center is wrapped into ``[-L/2, L/2)`` to keep the bump
    periodic on the domain.

    Args:
        grid: Discretization grid
        amplitude: Peak height A
        x0: Center of the packet
        width: Gaussian standard deviation (sigma)

    Returns:
        Gaussian field on the grid
    """
    x_shifted = (grid.x - x0 + grid.L / 2) % grid.L
    x_shifted -= grid.L / 2
    return amplitude * np.exp(-(x_shifted**2) / (2 * width**2))


def conserved_quantities(
    problem: "KdVProblem", u: np.ndarray
) -> tuple[float, float, float]:
    """
    Evaluate the first three KdV invariants of the field ``u``.

    For ``u_t + 6 u u_x + u_xxx = 0`` these are conserved by the exact dynamics:

    - mass     ``= integral u dx``
    - momentum ``= integral u^2 dx``
    - energy   ``= integral (2 u^3 - u_x^2) dx``

    Tracking them over a run both demonstrates KdV's integrability and serves
    as an accuracy diagnostic (a well-resolved spectral run keeps them flat).

    Args:
        problem: KdV problem (provides the grid and the spectral derivative)
        u: Field on the grid

    Returns:
        ``(mass, momentum, energy)``
    """
    dx = problem.grid.dx
    u_x = problem.compute_u_x(u)
    mass = float(dx * np.sum(u))
    momentum = float(dx * np.sum(u**2))
    energy = float(dx * np.sum(2 * u**3 - u_x**2))
    return mass, momentum, energy


def predict_soliton_amplitudes(
    grid: "Grid", u: np.ndarray, max_points: int = 512, cutoff: float = 0.02
) -> list[float]:
    """
    Predict the amplitudes of the solitons that ``u`` will evolve into.

    By the inverse-scattering theory of KdV, the asymptotic solitons of an
    initial field ``u(x,0)`` are the bound states (negative eigenvalues) of the
    Schroedinger operator ``-d^2/dx^2 - u``. An eigenvalue ``-kappa^2`` yields a
    soliton of amplitude ``2*kappa^2``. This solves that eigenproblem on the
    (optionally down-sampled) grid and returns the predicted amplitudes,
    largest first -- before any time stepping.

    Args:
        grid: Discretization grid
        u: Initial field
        max_points: Down-sample to at most this many points for the eigensolve
        cutoff: Ignore predicted amplitudes below ``cutoff * max(u)`` (radiation)

    Returns:
        Predicted soliton amplitudes, largest first
    """
    n = len(u)
    if n > max_points:
        idx = np.linspace(0, n, max_points, endpoint=False).astype(int)
        u_s = u[idx]
    else:
        u_s = u
    m = len(u_s)
    dx = grid.L / m

    # Periodic finite-difference second derivative.
    d2 = (
        np.diag(-2.0 * np.ones(m))
        + np.diag(np.ones(m - 1), 1)
        + np.diag(np.ones(m - 1), -1)
    ) / dx**2
    d2[0, -1] += 1.0 / dx**2
    d2[-1, 0] += 1.0 / dx**2

    operator = -d2 - np.diag(u_s)
    eigenvalues = np.linalg.eigvalsh(operator)

    threshold = cutoff * max(float(np.max(u)), 1e-9)
    amplitudes = [
        2.0 * (-lam) for lam in eigenvalues if lam < 0 and 2.0 * (-lam) > threshold
    ]
    return sorted(amplitudes, reverse=True)


def mode_energies(u: np.ndarray, n_modes: int = 6) -> np.ndarray:
    """
    Energy in the first ``n_modes`` Fourier modes of ``u`` (k = 1..n_modes).

    Returns ``|u_hat_k|^2`` for the lowest non-constant modes. Tracking these
    over time visualizes the Fermi-Pasta-Ulam-Tsingou picture: energy that
    stays trapped in a few low modes (recurrence) versus spreading to many.

    Args:
        u: Field on the grid
        n_modes: Number of low modes to return

    Returns:
        Array of length ``n_modes`` with the per-mode energy.
    """
    u_hat = np.fft.rfft(u)
    energy = np.abs(u_hat) ** 2
    out = np.zeros(n_modes)
    available = min(n_modes, len(energy) - 1)
    out[:available] = energy[1 : 1 + available]
    return out


def cosine_wave(grid: "Grid", amplitude: float, mode: int = 1) -> np.ndarray:
    """
    Build a cosine initial condition: ``A * cos(2*pi*mode*x / L)``.

    This is the Zabusky-Kruskal (1965) setup. A smooth cosine is not a KdV
    eigenstate: it steepens, breaks into a rank-ordered train of solitons, and
    -- because KdV is integrable on the periodic domain -- the solitons later
    re-align to nearly reconstruct the initial cosine (the Fermi-Pasta-Ulam-
    Tsingou recurrence). ``mode`` must be an integer so the wave is periodic.

    Args:
        grid: Discretization grid
        amplitude: Peak height A
        mode: Number of full wavelengths across the domain

    Returns:
        Cosine field on the grid
    """
    return amplitude * np.cos(2 * np.pi * mode * grid.x / grid.L)


@dataclass
class Grid:
    """Uniform grid with periodic boundary conditions."""

    N: int  # number of grid points
    L: float  # domain length

    def __post_init__(self) -> None:
        """Validate grid parameters."""
        if self.N <= 0:
            raise ValueError("N must be positive")
        if self.L <= 0:
            raise ValueError("L must be positive")

    @property
    def dx(self) -> float:
        """Grid spacing."""
        return self.L / self.N

    @property
    def x(self) -> np.ndarray:
        """Grid points: x_j = j * dx for j = 0, ..., N-1."""
        return np.arange(self.N) * self.dx

    @property
    def k_wave(self) -> np.ndarray:
        """Wave numbers for FFT-based derivatives (rfft layout)."""
        k = np.fft.rfftfreq(self.N, d=self.dx)
        return k * 2 * np.pi


class KdVProblem:
    """KdV equation problem with single soliton initial condition."""

    def __init__(self, grid: Grid, kappa: float):
        """
        Initialize KdV problem.

        Args:
            grid: Discretization grid
            kappa: Soliton parameter (controls amplitude and speed)
        """
        self.grid = grid
        self.kappa = kappa

    def initial_condition(self) -> np.ndarray:
        """Evaluate single soliton at t=0."""
        return self.soliton(0.0)

    def soliton(self, t: float) -> np.ndarray:
        """
        Evaluate analytical single-soliton solution.

        u(x,t) = 2 * kappa^2 * sech^2(kappa * (x - x0 - 4 * kappa^2 * t))

        The soliton starts at the center of the domain, ``x0 = L/2``, and
        travels to the right at speed ``c = 4 * kappa^2`` (wrapping around the
        periodic boundary). The argument is wrapped into ``[-L/2, L/2)`` so the
        soliton is periodic on the domain; without this wrapping the profile
        would have a jump discontinuity across the periodic seam, which
        destroys the spectral accuracy of the FFT-based derivatives.

        Args:
            t: Time

        Returns:
            u(x, t) evaluated on the grid
        """
        # Single soliton starting at the center of the domain.
        return soliton_profile(self.grid, self.kappa, self.grid.L / 2, t)

    def compute_u_x(self, u: np.ndarray) -> np.ndarray:
        """Compute u_x using FFT."""
        u_hat = np.fft.rfft(u)
        u_x_hat = 1j * self.grid.k_wave * u_hat
        return np.fft.irfft(u_x_hat, n=self.grid.N)

    def compute_u_xxx(self, u: np.ndarray) -> np.ndarray:
        """Compute u_xxx using FFT."""
        u_hat = np.fft.rfft(u)
        u_xxx_hat = -1j * self.grid.k_wave**3 * u_hat
        return np.fft.irfft(u_xxx_hat, n=self.grid.N)


class PseudoSpectralSolver:
    """Pseudo-spectral KdV solver using ETDRK4 time integration."""

    def __init__(self, problem: KdVProblem, dt: float):
        """
        Initialize solver.

        Args:
            problem: KdV problem definition
            dt: Target time step size (the actual step is adjusted slightly so
                an integer number of steps lands exactly on ``t_final``).
        """
        self.problem = problem
        self.dt = dt
        self.grid = problem.grid
        self.t_current = 0.0

        # Wave numbers; zero the Nyquist mode for odd-order derivatives so the
        # transforms of real fields stay real (standard pseudo-spectral practice).
        k = problem.grid.k_wave.copy()
        if problem.grid.N % 2 == 0:
            k[-1] = 0.0
        self._k = k

        # Linear operator L of u_hat_t = L u_hat for the term -u_xxx.
        # u_xxx -> (i k)^3 = -i k^3, so -u_xxx -> +i k^3.
        self._linear = 1j * k**3

        # Fourier multiplier for the (conservative) nonlinear term
        #   -6 u u_x = -3 d/dx (u^2)  ->  -3 i k * fft(u^2)
        self._nonlinear_factor = -3j * k

    def reset(self) -> None:
        """Reset solver state for a new solve."""
        self.t_current = 0.0

    def _nonlinear(self, u_hat: np.ndarray) -> np.ndarray:
        """Evaluate the nonlinear term -3 d/dx(u^2) in Fourier space."""
        u = np.fft.irfft(u_hat, n=self.grid.N)
        return self._nonlinear_factor * np.fft.rfft(u * u)

    def _etdrk4_coefficients(
        self, h: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Precompute the ETDRK4 propagators and quadrature weights for step ``h``.

        The coefficient functions (phi-functions of ``h*L``) are evaluated by a
        contour integral over a circle in the complex plane to avoid the
        catastrophic cancellation that a direct formula suffers near ``L = 0``.
        Because the KdV linear operator ``L = i k^3`` is purely imaginary, the
        phi-functions are complex valued, so the contour spans the *full*
        circle and the (complex) mean is kept as is -- the real-axis symmetry
        shortcut used for dissipative operators does not apply here.
        """
        linear = self._linear
        e = np.exp(h * linear)
        e2 = np.exp(h * linear / 2)

        m = _CONTOUR_POINTS
        roots = np.exp(2j * np.pi * (np.arange(1, m + 1) - 0.5) / m)
        lr = h * linear[:, None] + roots[None, :]

        q = h * np.mean((np.exp(lr / 2) - 1) / lr, axis=1)
        f1 = h * np.mean((-4 - lr + np.exp(lr) * (4 - 3 * lr + lr**2)) / lr**3, axis=1)
        f2 = h * np.mean((2 + lr + np.exp(lr) * (-2 + lr)) / lr**3, axis=1)
        f3 = h * np.mean((-4 - 3 * lr - lr**2 + np.exp(lr) * (4 - lr)) / lr**3, axis=1)
        return e, e2, q, f1, f2, f3

    def _step(
        self,
        v: np.ndarray,
        e: np.ndarray,
        e2: np.ndarray,
        q: np.ndarray,
        f1: np.ndarray,
        f2: np.ndarray,
        f3: np.ndarray,
    ) -> np.ndarray:
        """Advance the Fourier-space solution ``v`` by one ETDRK4 step."""
        n_v = self._nonlinear(v)
        a = e2 * v + q * n_v
        n_a = self._nonlinear(a)
        b = e2 * v + q * n_a
        n_b = self._nonlinear(b)
        c = e2 * a + q * (2 * n_b - n_v)
        n_c = self._nonlinear(c)
        v_next: np.ndarray = e * v + n_v * f1 + 2 * (n_a + n_b) * f2 + n_c * f3
        return v_next

    def solve(self, t_final: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Solve KdV from t=0 to t=t_final.

        Args:
            t_final: Final time

        Returns:
            Tuple of (u_final, times) where u_final is the solution at t_final.
        """
        self.reset()
        n_steps = max(1, round(t_final / self.dt))
        h = t_final / n_steps
        times = np.linspace(0.0, t_final, n_steps + 1)

        coeffs = self._etdrk4_coefficients(h)
        v = np.fft.rfft(self.problem.initial_condition())
        for _ in range(n_steps):
            v = self._step(v, *coeffs)
            self.t_current += h

        u_final: np.ndarray = np.fft.irfft(v, n=self.grid.N)
        return u_final, times

    def solve_with_history(
        self,
        t_final: float,
        n_snapshots: int = 50,
        u0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Solve KdV and return evenly spaced snapshots for visualization.

        Args:
            t_final: Final time
            n_snapshots: Approximate number of intermediate frames to record
                (in addition to the initial condition).
            u0: Optional initial field. Defaults to the problem's single-soliton
                initial condition; pass a custom field (e.g. a multi-soliton
                superposition) to evolve an arbitrary state.

        Returns:
            Tuple of (times, solutions) where ``times`` is a 1-D array and
            ``solutions`` is a list of physical-space snapshots, the first of
            which is the initial condition.
        """
        self.reset()
        n_steps = max(1, round(t_final / self.dt))
        h = t_final / n_steps
        save_every = max(1, n_steps // max(1, n_snapshots))

        initial = self.problem.initial_condition() if u0 is None else np.asarray(u0)
        coeffs = self._etdrk4_coefficients(h)
        v = np.fft.rfft(initial)

        times = [0.0]
        solutions = [initial]
        for i in range(n_steps):
            v = self._step(v, *coeffs)
            self.t_current += h
            if (i + 1) % save_every == 0 or i == n_steps - 1:
                times.append(self.t_current)
                solutions.append(np.fft.irfft(v, n=self.grid.N))

        return np.array(times), solutions
