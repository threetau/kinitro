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
- `kinitro/rl_interface.py` RL interface utilities (Observation, Action, ProprioKeys, ActionKeys).

## Setup / Install
- Recommended: `uv sync`
- Editable install: `pip install -e .`
- Dev extras (if using pip): `pip install -e ".[dev]"`
- Basilica CLI (for miner deployments): `curl -sSL https://basilica.ai/install.sh | bash`

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

### Multi-Worktree Development
When working with multiple git worktrees, use the helper scripts to avoid port/database collisions:
```bash
./scripts/worktree-env.sh     # Generate isolated .env and docker-compose.override.yml
./scripts/services.sh         # Manage services (start, stop, status, logs)
```
The scripts calculate deterministic port offsets from the worktree name (e.g., `fix/my-feature` → PostgreSQL 5789, API 8357, database `kinitro_fix_my_feature`).

### Quick Start
```bash
# 1. Generate environment (once per worktree)
./scripts/worktree-env.sh

# 2. Start database
docker compose up -d postgres

# 3. Initialize database
uv run kinitro db init --database-url $DATABASE_URL

# 4. Start services (in separate terminals or background)
uv run kinitro api --database-url $DATABASE_URL --port 8000 --no-auth &
uv run kinitro scheduler --database-url $DATABASE_URL \
  --network $NETWORK --netuid $NETUID --env-families metaworld &
uv run kinitro executor --api-url http://localhost:8000 --eval-mode docker \
  --eval-images '{"metaworld":"kinitro/metaworld:v1"}' &
```

Or use the helper script:
```bash
./scripts/services.sh start --all
```

### Building Environment Images
```bash
uv run kinitro env build procthor --tag kinitro/procthor:v1
uv run kinitro env build metaworld --tag kinitro/metaworld:v1
```

### Mock Miner Testing
Test the evaluation pipeline without a real trained policy:
```bash
# Start mock miner (returns random actions)
uv run kinitro mock-miner --port 8001

# In another terminal, test the mock miner endpoints:
curl http://localhost:8001/health
curl -X POST http://localhost:8001/reset \
  -H "Content-Type: application/json" \
  -d '{"task_config": {"env_id": "metaworld/pick-place-v3", "seed": 42}}'
curl -X POST http://localhost:8001/act \
  -H "Content-Type: application/json" \
  -d '{"obs": {"proprio": {"ee_pos": [0.0, 0.5, 0.2], "ee_quat": [0.0, 0.0, 0.0, 1.0], "ee_vel_lin": [0.0, 0.0, 0.0], "ee_vel_ang": [0.0, 0.0, 0.0], "gripper": [1.0]}, "rgb": {}}}'
```

### Scoring Mechanism Testing
Test the Pareto scoring without running full evaluations:
```bash
# Test scoring with simulated data
uv run kinitro test-scoring --n-miners 5 --n-envs 3 --episodes-per-env 50
```
This demonstrates first-commit advantage (earlier miners win ties under Pareto dominance).

### Miner Deployment Lifecycle Testing
Test the full miner workflow locally:

```bash
# 1. Initialize a new policy from template
uv run kinitro miner init ./test-policy
cd test-policy

# 2. Start the local policy server
uvicorn server:app --host 0.0.0.0 --port 8001

# 3. Test endpoints locally (in another terminal)
curl http://localhost:8001/health
curl -X POST http://localhost:8001/reset \
  -H "Content-Type: application/json" \
  -d '{"task_config": {"env_id": "metaworld/pick-place-v3", "seed": 42}}'

# 4. For local testing only - commit to local chain
uv run kinitro miner commit \
  --repo test-user/test-policy \
  --revision $(git rev-parse HEAD) \
  --endpoint http://localhost:8001 \
  --netuid 2 \
  --network local \
  --wallet-name test-wallet \
  --hotkey-name hotkey0

# 5. Verify commitment
uv run kinitro miner show-commitment \
  --netuid 2 \
  --network local \
  --wallet-name test-wallet \
  --hotkey-name hotkey0
```

### Basilica Deployment Testing
Deploy a miner to Basilica for realistic E2E testing:

```bash
# 1. Initialize policy template
uv run kinitro miner init ./test-policy

# 2. Deploy to Basilica (uploads to HuggingFace, deploys, commits on-chain)
# Note: Use your HuggingFace username as the repo namespace
uv run kinitro miner deploy \
  --repo <hf-username>/test-policy \
  --path ./test-policy \
  --network $NETWORK \
  --netuid $NETUID \
  --wallet-name alice \
  --hotkey-name hotkey0

# 3. Verify deployment is healthy
curl https://<deployment-id>.deployments.basilica.ai/health

# 4. List your deployments
basilica deploy ls

# 5. Delete deployment when done (--yes for non-interactive)
basilica deploy delete <deployment-id> --yes
```

### Full E2E Flow: Services → Miner → Evaluation → Scoring
Complete end-to-end test with all components:

```bash
# Option A: Start everything including mock miner (simplest)
./scripts/services.sh start --mock-miner

# Option B: Start services and miner separately
# Terminal 1: Start backend services
./scripts/services.sh start --all

# Terminal 2: Start mock miner
uv run kinitro mock-miner --port 8001

# Monitor evaluation logs
tail -f /tmp/kinitro_*/executor.log

# After evaluations complete, check scores via API:
curl http://localhost:$API_PORT/v1/scores/latest
curl http://localhost:$API_PORT/v1/weights/latest
```

### API Endpoints for Testing
Note: The API prefix is `/v1/` (not `/api/v1/`).
```bash
# Health check
curl http://localhost:$API_PORT/health

# List miners and their scores
curl http://localhost:$API_PORT/v1/miners

# List available environments
curl http://localhost:$API_PORT/v1/environments

# Get scores for a cycle
curl http://localhost:$API_PORT/v1/scores/latest
curl http://localhost:$API_PORT/v1/scores/<CYCLE_ID>

# Get computed weights
curl http://localhost:$API_PORT/v1/weights/latest
curl http://localhost:$API_PORT/v1/weights/<BLOCK_NUMBER>

# Task pool stats
curl http://localhost:$API_PORT/v1/tasks/stats
```

### Logs
- Service logs: `/tmp/kinitro_<worktree>/api.log`, `scheduler.log`, `executor.log`, `mock-miner.log`
- Container logs: `docker logs <container_name>` or `docker logs -f <name>`
- List eval containers: `docker ps --filter "name=kinitro-eval"`

### Troubleshooting
- Port conflicts: Run `./scripts/worktree-env.sh` to regenerate ports, then `./scripts/services.sh stop`
- Reset database: `PGPASSWORD=postgres psql -h localhost -p $POSTGRES_PORT -U postgres -c "DROP DATABASE $POSTGRES_DB; CREATE DATABASE $POSTGRES_DB;"`
- Stop eval containers: `docker stop $(docker ps -q --filter "name=kinitro-eval")`
- Check miner commitment: `uv run kinitro miner show-commitment --netuid ... --wallet-name ...`
- Verify miner endpoint: `curl <MINER_ENDPOINT>/health`

### Cleanup Commands
```bash
# Stop all kinitro processes
pkill -f "kinitro" || true

# Stop and remove evaluation containers
docker stop $(docker ps -q --filter "name=kinitro-eval") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=kinitro-eval") 2>/dev/null || true

# Stop postgres and remove volumes (full reset)
docker compose down -v

# Delete Basilica deployment
basilica deploy delete <deployment-id> --yes
```

### Database Schema Reference
The backend uses these PostgreSQL tables:
- `evaluation_cycles` - Cycle metadata (id, block_number, status, n_miners, n_environments, duration_seconds)
- `miner_scores` - Per-miner per-environment scores (uid, hotkey, env_id, success_rate, mean_reward, episodes_completed)
- `computed_weights` - Final weights per cycle (cycle_id, block_number, weights_json, weights_u16_json)
- `task_pool` - Individual evaluation tasks (task_uuid, cycle_id, miner_uid, env_id, seed, status, result)

Query examples:
```bash
source .env
PGPASSWORD=postgres psql -h localhost -p $POSTGRES_PORT -U postgres -d $POSTGRES_DB

# In psql:
SELECT id, block_number, status, n_miners FROM evaluation_cycles;
SELECT uid, env_id, success_rate FROM miner_scores WHERE cycle_id = 1;
SELECT weights_json FROM computed_weights ORDER BY id DESC LIMIT 1;
SELECT status, COUNT(*) FROM task_pool GROUP BY status;
```

## Docs and References
- Developer overview: `README.md`.
- Backend operator guide: `docs/backend-guide.md`.
- Miner guide: `docs/miner-guide.md`.
- Validator guide: `docs/validator-guide.md`.
- Scoring and incentives: `docs/scoring-and-incentives.md`.
- **Adding new environments**: `environments/README.md` - Complete guide for integrating new robotics environments.

## Change Hygiene for Agents
- Do not modify files outside the task scope.
- Avoid editing generated files and caches (e.g., `.ruff_cache/`).
- Keep commits small and focused; update docs if behavior changes.
