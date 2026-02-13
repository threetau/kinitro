# Contributing to Kinitro

Welcome! Kinitro is a Bittensor subnet for evaluating generalist robotics
policies. See the [README](README.md) for an overview of the project
architecture and quick-start instructions.

This guide covers everything you need to develop, test, and contribute to
Kinitro.

## Prerequisites and Setup

- Python 3.12 or higher is required.
- Install with [uv](https://github.com/astral-sh/uv) (recommended): `uv sync`
- Editable install: `pip install -e .`
- Dev extras (if using pip): `pip install -e ".[dev]"`
- Basilica CLI (for miner deployments): `curl -sSL https://basilica.ai/install.sh | bash`

## Common Commands

### Lint

- `ruff check .` (checks entire project)

### Format

- `ruff format .` (format all files)
- `ruff format --check .` (check formatting without modifying — used in CI)

### Type Check

- `ty check .`

### Tests

- `pytest tests/`
- With MuJoCo: `MUJOCO_GL=egl pytest tests/`

### Single Test (examples)

- `pytest tests/unit/test_pareto.py::TestEpsilonDominates::test_clear_dominance`
- `pytest tests/unit/test_pareto.py -k test_clear_dominance`
- `pytest -k pareto tests/unit`

### CLI Examples

- List environments: `uv run kinitro env list`
- Test an env: `uv run kinitro env test metaworld/pick-place-v3`
- Build env image: `uv run kinitro env build metaworld --tag my-env:v1`

## Git Hooks

- Hook script: `.githooks/pre-commit` invokes the `pre-commit` tool for ruff formatting/linting, then runs `ty` type checking.
- Setup: `git config core.hooksPath .githooks && uv tool install pre-commit`
- The hook uses `.pre-commit-config.yaml` for ruff rules via the pre-commit framework.

## Services (Local Development)

- API: `uv run kinitro api --database-url postgresql://user:pass@host/db`
- Scheduler: `uv run kinitro scheduler --netuid <id> --network finney --database-url postgresql://user:pass@host/db`
  - Filter to specific environment families: `--env-families metaworld` or `--env-families metaworld,genesis`
- Executor: `uv run kinitro executor --api-url http://localhost:8000`
  - Docker mode: `--eval-mode docker --eval-images '{"metaworld":"image:tag"}'`
  - Basilica mode: `--eval-mode basilica --eval-images '{"metaworld":"image:tag"}'`
- Validator: `uv run kinitro validate --backend-url https://api.kinitro.ai --netuid <id> --network finney`

## Backend Setup

- Init DB: `uv run kinitro db init --database-url postgresql://user:pass@host/db`
- DB status: `uv run kinitro db status --database-url postgresql://user:pass@host/db`

## Environment Config

- See `.env.example` for common env vars.
- Runtime settings are read via Pydantic settings classes in `kinitro/config.py`.
- Keep secrets out of the repo; do not commit `.env` files or any files listed in `.gitignore`.

## Code Style Guidelines

### Imports

- Group imports: standard library, third-party, then local.
- Keep imports at the top of the file (after module docstring if present).
- Sort with Ruff `I` rules (no separate isort config).
- Prefer explicit imports over wildcard imports.

### Formatting

- 4-space indentation.
- Line length target: 100 (`ruff` ignores `E501` but still keep lines reasonable).
- Use trailing commas in multi-line literals and call arguments.

### Typing

- Python 3.12 syntax (`list[int]`, `dict[str, float]`, `str | None`).
- Add type hints to all function signatures (parameters and return types), not just public ones.
- Prefer `BaseSettings` and `BaseModel` type annotations for config/DTOs.
- `ty` runs in CI; avoid `Any` unless required and explain why in code.
- Use `NewType` to distinguish domain identifiers that share an underlying primitive (e.g., `MinerUID`, `BlockNumber`, `EnvironmentId`, `Hotkey`). This prevents accidental misuse such as passing a `MinerUID` where a `BlockNumber` is expected.
- Centralize shared newtypes, type aliases, enums, and `TypedDict` definitions in `kinitro/types.py`. Import from there rather than re-defining types locally.
- When introducing a new domain concept that is fundamentally a `str`, `int`, or other primitive, create a `NewType` for it in `kinitro/types.py` and use it consistently across signatures, models, and data structures.
- Prefer `TypedDict` or `dataclasses.dataclass` over plain `dict` for structured data with known keys.

### Naming

- `snake_case` for functions, variables, modules.
- `PascalCase` for classes, Pydantic models, and custom types.
- `UPPER_CASE` for constants.
- Test files: `test_*.py`; test classes: `Test*`.

### Error Handling

- CLI commands use `typer.Exit(1)` for user-facing failures.
- Provide actionable error messages; log details with `structlog`.
- Avoid broad `except Exception` unless you re-raise or return a clear error.

### Logging

- Use `structlog.get_logger()` and structured log fields.
- Log at info/debug/warn/error levels consistently with existing patterns.

### Async and IO

- Use `async def` for I/O and DB operations.
- Use `asyncio.run(...)` at CLI boundaries.
- Use `asynccontextmanager` and session managers for DB work.

### Pydantic / API Models

- Pydantic models live in `kinitro/backend/models.py`.
- Prefer `Field(...)` for validation constraints and docs.
- Use `from_attributes = True` for ORM-to-model conversion.

### SQLAlchemy

- ORM models are in `kinitro/backend/models.py`.
- Keep DB writes inside the storage layer (`kinitro/backend/storage.py`).
- Use `select(...).where(...).limit(...)` patterns for queries.

### Tests

- Keep tests deterministic; use fixed seeds (`np.random.default_rng(42)`).
- Use clear asserts; prefer `np.testing` for array comparisons.
- Keep unit tests fast; integration tests go in `tests/integration`.

## Testing

For detailed testing docs and troubleshooting, see [`docs/e2e-testing.md`](docs/e2e-testing.md).

### Quick Reference

```bash
# Start all services with mock miner (uses default ports)
./scripts/services.sh start --mock-miner

# Run test scripts
./scripts/test-e2e.sh         # Full E2E pipeline test
./scripts/test-api.sh         # Test API endpoints
./scripts/test-mock-miner.sh  # Test mock miner endpoints

# Service management
./scripts/services.sh status  # Check running services
./scripts/services.sh logs    # Tail service logs
./scripts/services.sh stop    # Stop all services
```

### Worktree Development

For multi-worktree development (avoids port collisions between parallel checkouts):

```bash
./scripts/worktree-env.sh           # Generate isolated ports/database (once per worktree)
./scripts/services.sh start --all   # Uses worktree-specific ports
```

## How to Contribute

1. Fork the repo and clone your fork.
2. Create a branch from `main` with a descriptive name:
   - `feature/...` for new features
   - `fix/...` for bug fixes
   - `docs/...` for documentation changes
3. Make your changes, following the code style guidelines above.
4. Run checks locally before pushing:

   ```bash
   ruff format --check .
   ruff check .
   ty check .
   pytest tests/
   ```

5. Open a pull request against `main`.

## CI Checks

Every pull request runs the following checks automatically:

| Check         | Command                 | Purpose                       |
| ------------- | ----------------------- | ----------------------------- |
| Ruff format   | `ruff format --check .` | Enforce consistent formatting |
| Ruff lint     | `ruff check .`          | Catch lint issues             |
| Ty type check | `ty check .`            | Static type analysis          |

## Commit Conventions

This project follows [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<scope>): <description>
```

**Types:** `feat`, `fix`, `refactor`, `perf`, `chore`, `test`, `docs`

**Scopes** (optional): `genesis`, `executor`, `scheduler`, `scoring`, `cli`, `crypto`, `rl_interface`

Examples:

- `feat(executor): add GPU passthrough for Docker eval containers`
- `fix(cli): use lazy bittensor imports to prevent argparse hijacking`
- `refactor: deduplicate code`

## Types of Contributions

- **Code** — core package in `kinitro/`
- **Environments** — evaluation environments in `environments/` (see [environments/README.md](environments/README.md))
- **Documentation** — guides and references in `docs/`
- **Tests** — unit and integration tests in `tests/`

## Docs and References

- Developer overview: [`README.md`](README.md)
- Backend operator guide: [`docs/backend-guide.md`](docs/backend-guide.md)
- Miner guide: [`docs/miner-guide.md`](docs/miner-guide.md)
- Validator guide: [`docs/validator-guide.md`](docs/validator-guide.md)
- Scoring and incentives: [`docs/scoring-and-incentives.md`](docs/scoring-and-incentives.md)
- Adding new environments: [`environments/README.md`](environments/README.md)
