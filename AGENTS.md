# AGENTS.md

This file guides agentic coding tools working in this repo.
Keep it current when commands or conventions change.

## Scope
- Project: Kinitro robotics evaluation subnet (Python 3.12).
- Primary entry point: `kinitro` CLI (see `kinitro/cli/`).
- Services: API, scheduler, executor, validator, miner tooling.

## Repo Map
- `demos/` demonstration scripts and examples.
- `docs/` operator, validator, and miner guides.
- `environments/` evaluation environments (MetaWorld, ProcTHOR).
- `scripts/` utility scripts.
- `tests/` unit and integration tests.
- `kinitro/` core package.
- `kinitro/api/` FastAPI app and routes.
- `kinitro/backend/` storage, models, database logic.
- `kinitro/chain/` chain integration and networking.
- `kinitro/cli/` CLI command group modules.
- `kinitro/environments/` environment discovery and helpers.
- `kinitro/executor/` evaluation executor.
- `kinitro/miner/` miner tooling and workflows.
- `kinitro/scheduler/` task generation + scoring.
- `kinitro/scoring/` pareto + winners-take-all scoring.
- `kinitro/validator/` validator workflows.

### Key Modules
- `kinitro/cli/` CLI entry point and command groups.
- `kinitro/config.py` runtime settings and configuration.
- `kinitro/crypto.py` cryptography helpers.
- `kinitro/rl_interface.py` RL interface utilities (CanonicalObservation, CanonicalAction).

## Setup / Install
- Recommended: `uv sync`
- Editable install: `pip install -e .`
- Dev extras (if using pip): `pip install -e ".[dev]"`

## Common Commands
### Lint
- `ruff check .` (checks entire project: kinitro/, tests/, environments/, scripts/, demos/)

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
- Build env image: `uv run kinitro env build --env-id metaworld/pick-place-v3 --tag my-env:v1`

## Git Hooks
- Hook script: `.githooks/pre-commit` invokes the `pre-commit` tool for ruff formatting/linting, then runs `ty` type checking.
- Setup: `git config core.hooksPath .githooks && uv tool install pre-commit`
- The hook uses `.pre-commit-config.yaml` for ruff rules via the pre-commit framework.

## Services (local dev)
- API: `uv run kinitro api --database-url postgresql://user:pass@host/db`
- Scheduler: `uv run kinitro scheduler --netuid <id> --network finney --database-url postgresql://user:pass@host/db`
  - Filter to specific environment families: `--env-families metaworld` or `--env-families metaworld,procthor`
- Executor: `uv run kinitro executor --api-url http://localhost:8000`
  - Docker mode: `--eval-mode docker --eval-images '{"metaworld":"image:tag"}'`
  - Basilica mode: `--eval-mode basilica --eval-images '{"metaworld":"image:tag"}'`
- Validator: `uv run kinitro validate --backend-url https://api.kinitro.ai --netuid <id> --network finney`

## Backend Setup (operator quick start)
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
- Add return types to public functions and methods.
- Prefer `BaseSettings` and `BaseModel` type annotations for config/DTOs.
- `ty` runs in CI; avoid `Any` unless required and explain why in code.

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

## End-to-End Testing
- Full guide: `docs/e2e-testing-guide.md`
- Tests API, scheduler, executor (Docker or Basilica mode), and miner deployment
- Quick start (Docker mode, metaworld only):
  ```bash
  uv run kinitro db reset --database-url $DATABASE_URL --force
  uv run kinitro api --database-url $DATABASE_URL &
  uv run kinitro scheduler --database-url $DATABASE_URL \
    --network $NETWORK --netuid $NETUID --episodes-per-env 3 --env-families metaworld &
  uv run kinitro executor --api-url http://localhost:8000 --eval-mode docker \
    --eval-images '{"metaworld":"shr1ftyy/kinitro-metaworld:v1"}' &
  ```
- Monitor: `tail -f /tmp/scheduler.log` and `tail -f /tmp/executor.log`

## Docs and References
- Developer overview: `README.md`.
- Backend operator guide: `docs/backend-guide.md`.
- Miner guide: `docs/miner-guide.md`.
- Validator guide: `docs/validator-guide.md`.
- Scoring and incentives: `docs/scoring-and-incentives.md`.
- E2E testing guide: `docs/e2e-testing-guide.md`.

## Change Hygiene for Agents
- Do not modify files outside the task scope.
- Avoid editing generated files and caches (e.g., `.ruff_cache/`).
- Keep commits small and focused; update docs if behavior changes.
