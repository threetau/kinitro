# End-to-End Testing Guide

This guide covers E2E testing procedures for the kinitro evaluation pipeline.

## Quick Start

```bash
# Run full E2E test (uses default ports if no .env exists)
./scripts/test-e2e.sh

# Or start services and run component tests individually:
./scripts/services.sh start --mock-miner
./scripts/test-api.sh
./scripts/test-mock-miner.sh
./scripts/services.sh stop
```

## Multi-Worktree Development

When working with multiple git worktrees, generate isolated ports and databases to avoid collisions:

```bash
./scripts/worktree-env.sh     # Generate isolated .env and docker-compose.override.yml (once per worktree)
./scripts/services.sh start   # Uses worktree-specific ports from .env
```

The scripts calculate deterministic port offsets from the worktree name (e.g., `fix/my-feature` → PostgreSQL 5789, API 8357, database `kinitro_fix_my_feature`).

Without `worktree-env.sh`, all scripts use default ports (API=8000, PostgreSQL=5432, database=`kinitro`).

## Building Environment Images

Before running evaluations, build the environment Docker images:

```bash
uv run kinitro env build metaworld --tag kinitro/metaworld:v1
uv run kinitro env build genesis --tag kinitro/genesis:v1
```

## Scoring Mechanism Testing

Test the Pareto scoring without running full evaluations:

```bash
uv run kinitro test-scoring --n-miners 5 --n-envs 3 --episodes-per-env 50
```

This demonstrates first-commit advantage (earlier miners win ties under Pareto dominance).

## Miner Deployment Lifecycle Testing

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

## Basilica Deployment Testing

Deploy a miner to Basilica for realistic E2E testing:

```bash
# 1. Initialize policy template
uv run kinitro miner init ./test-policy

# 2. Deploy to Basilica (uploads to HuggingFace, deploys, commits on-chain)
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

# 5. Delete deployment when done
basilica deploy delete <deployment-id> --yes
```

## Logs

- Service logs: `/tmp/kinitro_<worktree>/api.log`, `scheduler.log`, `executor.log`, `mock-miner.log`
- Container logs: `docker logs <container_name>` or `docker logs -f <name>`
- List eval containers: `docker ps --filter "name=kinitro-eval"`

## Troubleshooting

| Problem                | Solution                                                                                                                             |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| Port conflicts         | Run `./scripts/worktree-env.sh` to regenerate ports, then `./scripts/services.sh stop`                                               |
| Database issues        | `PGPASSWORD=postgres psql -h localhost -p $POSTGRES_PORT -U postgres -c "DROP DATABASE $POSTGRES_DB; CREATE DATABASE $POSTGRES_DB;"` |
| Stuck eval containers  | `docker stop $(docker ps -q --filter "name=kinitro-eval")`                                                                           |
| Check miner commitment | `uv run kinitro miner show-commitment --netuid ... --wallet-name ...`                                                                |
| Verify miner endpoint  | `curl <MINER_ENDPOINT>/health`                                                                                                       |

## Cleanup Commands

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

## Database Schema Reference

The backend uses these PostgreSQL tables:

| Table               | Description                                                                                           |
| ------------------- | ----------------------------------------------------------------------------------------------------- |
| `evaluation_cycles` | Cycle metadata (id, block_number, status, n_miners, n_environments, duration_seconds)                 |
| `miner_scores`      | Per-miner per-environment scores (uid, hotkey, env_id, success_rate, mean_reward, episodes_completed) |
| `computed_weights`  | Final weights per cycle (cycle_id, block_number, weights_json, weights_u16_json)                      |
| `task_pool`         | Individual evaluation tasks (task_uuid, cycle_id, miner_uid, env_id, seed, status, result)            |

### Query Examples

```bash
source .env
PGPASSWORD=postgres psql -h localhost -p $POSTGRES_PORT -U postgres -d $POSTGRES_DB
```

```sql
-- In psql:
SELECT id, block_number, status, n_miners FROM evaluation_cycles;
SELECT uid, env_id, success_rate FROM miner_scores WHERE cycle_id = 1;
SELECT weights_json FROM computed_weights ORDER BY id DESC LIMIT 1;
SELECT status, COUNT(*) FROM task_pool GROUP BY status;
```
