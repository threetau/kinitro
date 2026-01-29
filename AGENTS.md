# AGENTS.md

This file guides agentic coding tools working in this repo.
Keep it current when commands or conventions change.

## Scope
- Project: Kinitro robotics evaluation subnet (Python 3.12).
- Primary entry point: `kinitro` CLI (see `kinitro/cli.py`).
- Services: API, scheduler, executor, validator, miner tooling.

## Repo Map (high signal)
- `kinitro/` core package.
- `kinitro/api/` FastAPI app and routes.
- `kinitro/backend/` storage, models, database logic.
- `kinitro/scheduler/` task generation + scoring.
- `kinitro/executor/` evaluation executor.
- `kinitro/scoring/` pareto + winners-take-all scoring.
- `tests/` pytest suites.
- `docs/` operator, validator, and miner guides.
- `eval-env/` docker build context for evaluation env.

## Rules Files
- Cursor rules: none found in `.cursor/rules/` or `.cursorrules`.
- Copilot rules: none found in `.github/copilot-instructions.md`.

## Setup / Install
- Recommended: `uv sync`
- Editable install: `pip install -e .`
- Dev extras (if using pip): `pip install -e ".[dev]"`

## Common Commands
### Lint
- `ruff check kinitro/`

### Type Check
- `mypy kinitro/`

### Tests
- `pytest tests/`
- With MuJoCo: `MUJOCO_GL=egl pytest tests/`

### Single Test (examples)
- `pytest tests/unit/test_pareto.py::TestEpsilonDominates::test_clear_dominance`
- `pytest tests/unit/test_pareto.py -k test_clear_dominance`
- `pytest -k pareto tests/unit`

### CLI Examples
- List environments: `uv run kinitro list-envs`
- Test an env: `uv run kinitro test-env metaworld/pick-place-v3`

## Git Hooks
- Hook script: `.githooks/pre-commit` (formats then lints staged Python files).
- Enable locally with: `git config core.hooksPath .githooks`

## Services (local dev)
- API: `uv run kinitro api --database-url postgresql://user:pass@host/db`
- Scheduler: `uv run kinitro scheduler --netuid <id> --network finney --database-url postgresql://user:pass@host/db`
- Executor: `uv run kinitro executor --api-url http://localhost:8000`
- Validator: `uv run kinitro validate --backend-url https://api.kinitro.ai --netuid <id> --network finney`

## Backend Setup (operator quick start)
- Build eval env image: `uv run kinitro build-eval-env --tag kinitro/eval-env:v1`
- Init DB: `uv run kinitro db init --database-url postgresql://user:pass@host/db`
- DB status: `uv run kinitro db status --database-url postgresql://user:pass@host/db`

## Environment Config
- See `.env.example` for common env vars.
- Runtime settings are read via Pydantic settings classes in `kinitro/config.py`.
- Keep secrets out of the repo; do not commit `.env` files.

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
- `mypy` is strict; avoid `Any` unless required and explain why in code.

### Naming
- `snake_case` for functions, variables, modules.
- `PascalCase` for classes and Pydantic models.
- `UPPER_CASE` for constants.
- Test files: `test_*.py`; test classes: `Test*`.

### Error Handling
- CLI commands use `typer.Exit(1)` for user-facing failures.
- Provide actionable error messages; log details with `structlog`.
- Avoid broad `except Exception` unless you re-raise or return a clear error.

### Logging
- Use `structlog.get_logger()` and structured log fields.
- Log at info/warn/error levels consistently with existing patterns.

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

## Docs and References
- Developer overview and commands: `README.md`.
- Operator guides: `docs/backend-guide.md`.
- Miner guide: `docs/miner-guide.md`.
- Validator guide: `docs/validator-guide.md`.

## Change Hygiene for Agents
- Do not modify files outside the task scope.
- Avoid editing generated files and caches (e.g., `.ruff_cache/`).
- Keep commits small and focused; update docs if behavior changes.
