# Tech Stack

## Core Language

- **Python 3.10+** (primary), with type hints

## Numerical Computing

- **NumPy** (v1.24+) - array operations, FFT, mathematical functions
- **SciPy** (v1.11+) - optional: special functions, optimization

## Visualization

- **matplotlib** (v3.7+) - plotting solutions and error analysis

## Testing

- **pytest** (v7.4+) - unit and integration tests
- **pytest-cov** - coverage reporting

## Build & Dependency Management

- **pip** for Python dependencies
- `requirements.txt` for version pinning
- No compilation step (pure Python/NumPy)

## Reproducibility

- **Deterministic FFT**: NumPy's FFT is deterministic within a version
- **Random seeds**: All tests must set explicit numpy.random.seed()
- **Numerical tolerances**: documented in validation-contract.md
- **Version pinning**: All dependencies pinned in requirements.txt

## Development Tools

- **black** - code formatting
- **flake8** or **ruff** - linting
- **mypy** - type checking (optional but recommended)
