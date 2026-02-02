# End-to-End Testing Guide

This guide documents how to run end-to-end tests for the Kinitro evaluation system.

## Prerequisites

### Required Services

1. **PostgreSQL Database** - For storing tasks and results
2. **Bittensor Local Node** (or testnet) - For on-chain commitments
3. **Docker** - For running evaluation environments

### Required Credentials

Set these in your environment or `.bashrc`:

```bash
# HuggingFace token for uploading/downloading models
export HF_TOKEN="hf_xxx..."

# Basilica API token (for basilica mode miner deployments)
export BASILICA_API_TOKEN="basilica_xxx..."
```

### Build Docker Images

```bash
# Build evaluation environment images
uv run kinitro env build metaworld --tag kinitro/metaworld:v1
uv run kinitro env build procthor --tag kinitro/procthor:v1  # Optional, x86_64 Linux only

# Build miner runner image (for Docker mode miner deployments)
uv run kinitro env build miner --tag kinitro/miner-runner:v1
```

## Test Configuration

### Database

```bash
DATABASE_URL="postgresql://myuser:mypassword@localhost/backenddb"
```

### Bittensor Network

For local testing:
```bash
NETWORK="ws://10.0.0.3:9944"  # Local subtensor node
NETUID=2
WALLET_NAME="bob"
HOTKEY_NAME="hotkey0"
```

For testnet:
```bash
NETWORK="test"
NETUID=<your_netuid>
```

## Running End-to-End Tests

### Step 1: Reset Database

```bash
uv run kinitro db reset --database-url $DATABASE_URL --force
uv run kinitro db init --database-url $DATABASE_URL
```

### Step 2: Create and Deploy Test Policy

```bash
# Create test policy from template
rm -rf /tmp/kinitro-test-policy
uv run kinitro miner init /tmp/kinitro-test-policy

# Deploy to HuggingFace and commit on-chain
uv run kinitro miner deploy \
  --repo <your-username>/kinitro-test-policy \
  --path /tmp/kinitro-test-policy \
  --netuid $NETUID \
  --network $NETWORK \
  --wallet-name $WALLET_NAME \
  --hotkey-name $HOTKEY_NAME
```

### Step 3: Start API Server

```bash
uv run kinitro api \
  --database-url $DATABASE_URL \
  --host 0.0.0.0 \
  --port 8000 \
  --log-level DEBUG
```

### Step 4: Start Scheduler

In a separate terminal:

```bash
uv run kinitro scheduler \
  --database-url $DATABASE_URL \
  --network $NETWORK \
  --netuid $NETUID \
  --eval-interval 60 \
  --episodes-per-env 2 \
  --log-level DEBUG
```

### Step 5: Start Executor

In a separate terminal:

```bash
# For Docker mode (local containers)
# Uses affinetes for both eval envs and miner deployments
export KINITRO_EXECUTOR_MINER_HF_TOKEN=$HF_TOKEN

uv run kinitro executor \
  --api-url http://localhost:8000 \
  --batch-size 4 \
  --log-level DEBUG

# For Basilica mode (cloud deployments)
# Uses affinetes for eval envs, Basilica SDK directly for miner deployments
export KINITRO_EXECUTOR_MINER_BASILICA_API_TOKEN=$BASILICA_API_TOKEN
export KINITRO_EXECUTOR_MINER_HF_TOKEN=$HF_TOKEN
export KINITRO_EXECUTOR_EVAL_MEM_LIMIT="8Gi"  # Kubernetes format for Basilica

uv run kinitro executor \
  --api-url http://localhost:8000 \
  --eval-mode basilica \
  --batch-size 4 \
  --log-level DEBUG
```

### Step 6: Monitor Progress

```bash
# Check task stats
curl -s http://localhost:8000/v1/tasks/stats | python3 -m json.tool

# Check health
curl -s http://localhost:8000/health

# Check scores (after cycle completes)
curl -s http://localhost:8000/v1/scores/latest | python3 -m json.tool
```

## Expected Results

### Successful Miner Deployment Flow

1. Scheduler finds miners with commitments on-chain
2. Scheduler creates tasks (miners × environments × episodes)
3. Executor fetches tasks from API
4. For each task:
   - Executor creates miner deployment (Docker or Basilica)
   - Deployment is cached by `(repo, revision)`
   - Evaluation runs in MetaWorld/ProcTHOR container
   - Results submitted to API
5. Scheduler computes Pareto scores when cycle complete

### Log Messages to Watch For

```
# Scheduler
commitments_loaded       count=N total_miners=M
tasks_created            cycle_id=1 total_tasks=X

# Executor
miner_deployment_created  miner_uid=0 repo=user/policy url=http://...
deployment_cache_hit      repo=user/policy revision=abc123
task_executed            score=0.0 success=False

# Cycle completion
cycle_completed          cycle_id=1
```

## Troubleshooting

### "No miners with commitments"

- Check that the miner has committed on the correct netuid
- Verify with: `uv run kinitro miner show-commitment --netuid $NETUID --network $NETWORK`

### "Failed to pull image"

- Build the required Docker images (see Prerequisites)
- For ProcTHOR: requires x86_64 Linux (not ARM64 or emulated)

### "Miner deployment not ready"

- Check HuggingFace token is valid
- Check the miner's `server.py` starts correctly
- Increase `--miner-deployment-warmup-timeout`

### "No miner endpoint available"

- Ensure `miner_deployment_enabled=true`
- For Basilica mode: ensure `KINITRO_EXECUTOR_MINER_BASILICA_API_TOKEN` is set
- Check miner has valid `miner_repo` and `miner_revision` in task

## Quick Test Script

```bash
#!/bin/bash
# quick-e2e-test.sh

set -e

DATABASE_URL="postgresql://myuser:mypassword@localhost/backenddb"
NETWORK="ws://10.0.0.3:9944"
NETUID=2
HF_REPO="syeam-alt/kinitro-test-policy"

# Reset database
uv run kinitro db reset --database-url $DATABASE_URL --force
uv run kinitro db init --database-url $DATABASE_URL

# Start services in background
uv run kinitro api --database-url $DATABASE_URL &
API_PID=$!
sleep 3

uv run kinitro scheduler \
  --database-url $DATABASE_URL \
  --network $NETWORK \
  --netuid $NETUID \
  --eval-interval 60 \
  --episodes-per-env 2 &
SCHEDULER_PID=$!
sleep 5

# Run executor (foreground, Ctrl+C to stop)
uv run kinitro executor --api-url http://localhost:8000 --batch-size 4

# Cleanup
kill $API_PID $SCHEDULER_PID 2>/dev/null || true
```

## Configuration Reference

### Executor Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `KINITRO_EXECUTOR_EVAL_MODE` | `docker` | `docker` or `basilica` |
| `KINITRO_EXECUTOR_EVAL_MEM_LIMIT` | `8g` | Memory limit (`8g` for Docker, `8Gi` for Basilica) |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_ENABLED` | `true` | Enable miner deployments |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_IMAGE` | `kinitro/miner-runner:v1` | Miner runner image |
| `KINITRO_EXECUTOR_MINER_BASILICA_API_TOKEN` | - | Basilica API token (required for basilica mode) |
| `KINITRO_EXECUTOR_MINER_HF_TOKEN` | - | HuggingFace token (optional, for private repos) |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_TTL_SECONDS` | `600` | Deployment cache TTL |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_WARMUP_TIMEOUT` | `300` | Deployment ready timeout |

### Mode Differences

| Aspect | Docker Mode | Basilica Mode |
|--------|-------------|---------------|
| Eval envs | affinetes (local containers) | affinetes (Basilica backend) |
| Miner deployments | affinetes (local containers) | Basilica SDK directly |
| Memory format | Docker format (`8g`) | Kubernetes format (`8Gi`) |
| Requires | Docker daemon | `BASILICA_API_TOKEN` |
