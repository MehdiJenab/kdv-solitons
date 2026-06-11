# Project: kdv-solitons

## Environment
- Python 3.11+ via `.venv/` — always activate before running commands
- Install: `pip install -e ".[dev]"`

## Toolchain
- Lint + format: `ruff check --fix` and `ruff format`
- Type check: `mypy src/`
- Tests: `pytest` (slow tests excluded by default)
- Full suite: `make check`

## Rules
- Tests are the specification. Never modify a test to make it pass.
- Coverage is enforced at 100%. Every new code path needs a test.
- All functions must be type-annotated. No `# type: ignore` without explanation.
- Pre-commit hooks must pass. Never use `--no-verify`.

## Before making changes
1. Run `make check` — it must pass before you start
2. Read `specs/constitution/` to understand the project goals
3. Read `src/kdv_solver/` to understand the current implementation
