# Validation Contract

## Testing Layers

### Unit Tests (`tests/unit/`)
- Grid generation and wave number computation
- FFT-based derivative correctness (verify against known functions)
- Linear operator application (u_xxx for pure Fourier modes)
- Nonlinear term evaluation (6uu_x)

### Feature Tests (`tests/features/`)
- Single-soliton propagation: verify shape and speed
- Multi-soliton interaction: verify soliton resilience
- Energy conservation: ∫u² dx = constant for KdV

### End-to-End Tests (`tests/e2e/`)
- Full time integration from t=0 to t=t_final
- Comparison with analytical solution at multiple times
- Convergence study: refine grid and verify spectral decay

## Numerical Tolerances

| Test Type | Tolerance | Notes |
|-----------|-----------|-------|
| Single-soliton L2 error | atol=1e-6, rtol=1e-4 | Spectral accuracy expected |
| Single-soliton L∞ error | atol=1e-6, rtol=1e-4 | Max norm |
| Derivative verification | atol=1e-12, rtol=1e-10 | Pure Fourier modes |
| Energy conservation | | ΔE/E₀ < 1e-4 |

## Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Time to solution (N=256, t=1) | < 1s | Development target |
| Spectral convergence | Error ∝ e^(-αN) | For smooth solutions |

## Pass/Fail Criteria

- **Pass**: All tests within specified tolerances
- **Fail**: Any test exceeds tolerance by > 10×
- **Warning**: Test exceeds tolerance by 2-10× (may indicate numerical issue)

## Test Fixtures

- `tests/fixtures/soliton_data/` - Analytical solution reference data
- `tests/fixtures/grid_configs/` - Grid configuration files
- `tests/fixtures/reference_solutions/` - Pre-computed solutions for comparison

## Validation Test: Single Soliton

**Test**: Propagate single soliton for time t and compare with analytical solution

**Analytical solution**:
```
u(x,t) = 2κ² sech²(κ(x - 4κ²t))
```

**Validation metrics**:
- L2 norm: ``||u_numerical - u_analytical||₂ / ||u_analytical||₂ < tol``
- L∞ norm: `max|u_numerical - u_analytical| < tol`
- Phase error: `|x_center_numerical - x_center_analytical| < tol`
- Amplitude error: `|amp_numerical - amp_analytical| < tol`
