# Code Standards

## Naming Conventions

| Concept | Pattern | Example |
|---------|---------|---------|
| Classes | PascalCase | `KdVProblem`, `PseudoSpectralSolver` |
| Functions/Methods | snake_case | `compute_nonlinear_term`, `step_forward` |
| Constants | UPPER_SNAKE_CASE | `PI`, `DEFAULT_GRID_SIZE` |
| Variables | snake_case | `wave_numbers`, `time_step` |
| Private members | _prefix | `_fft_grid`, `_linear_operator` |

## Code Style

- **Line length**: 88 characters (black default)
- **Imports**: Standard library first, then third-party, then local
- **Type hints**: Required for all public functions
- **Docstrings**: NumPy style for public APIs

## Error Handling

- Validate input parameters in constructor/setup
- Use ValueError for invalid parameters
- Use RuntimeError for numerical failures (e.g., NaN detection)

## Numerical Code Patterns

```python
# Fourier transform convention (match NumPy)
u_hat = np.fft.rfft(u)  # returns N/2+1 coefficients
u = np.fft.irfft(u_hat, n=N)  # inverse

# Wave numbers for N points on [0, L)
k = np.fft.rfftfreq(N, d=dx) * 2 * np.pi
# k = [0, 2π/L, 4π/L, ..., π/(L dx)] (Nyquist)

# Derivatives in Fourier space
u_x = np.fft.irfft(1j * k * u_hat, n=N)
u_xxx = np.fft.irfft(-1j * k**3 * u_hat, n=N)
```

## Testing Standards

- **Unit tests**: One file per module, prefixed with `test_`
- **Test naming**: `test_<what>_<condition>_<expected>`
- **Fixtures**: Use pytest fixtures for expensive setup (grid, solver)
- **Tolerances**: Use np.testing.assert_allclose with explicit rtol/atol
