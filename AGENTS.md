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
- `environments/` evaluation environments (MetaWorld, Genesis).
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
- `kinitro/rl_interface.py` RL interface utilities (Observation, Action, ProprioKeys, ActionKeys).

## Development Standards

All setup instructions, commands, code style guidelines, and testing
procedures are documented in [CONTRIBUTING.md](CONTRIBUTING.md).
Key topics:

- **Setup**: Prerequisites, `uv sync`, dev install
- **Commands**: Lint (`ruff check .`), format (`ruff format .`),
  type check (`ty check .`), tests (`pytest tests/`)
- **Git hooks**: `.githooks/pre-commit` setup
- **Local services**: API, scheduler, executor, validator
- **Code style**: Imports, formatting, typing, naming, error handling,
  logging, async, Pydantic, SQLAlchemy, tests
- **E2E testing**: Service scripts, worktree isolation
- **CI checks**: What runs on PRs (ruff format, ruff lint, ty)

## Change Hygiene for Agents

- Do not modify files outside the task scope.
- Avoid editing generated files and caches (e.g., `.ruff_cache/`).
- Keep commits small and focused; update docs if behavior changes.
- Run `ruff check .`, `ruff format --check .`, and `ty check .` before finishing.
