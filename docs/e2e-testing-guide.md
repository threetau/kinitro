# End-to-End Testing Guide

This guide covers running end-to-end tests for the kinitro validator backend.

## Prerequisites

- Docker and Docker Compose
- Python 3.11+ with `uv` package manager
- Access to a Bittensor network (testnet or local)
- Deployed miner instances (for full E2E tests)

## Quick Start

```bash
# 1. Generate worktree-specific environment
./scripts/worktree-env.sh

# 2. Start services
./scripts/start-services.sh

# 3. Check status
./scripts/start-services.sh status

# 4. View logs
./scripts/start-services.sh logs
```

## Components

### API Server

The API server handles task management and result submission.

```bash
uv run kinitro api \
  --database-url "$DATABASE_URL" \
  --port "$API_PORT" \
  --no-auth \
  --log-level INFO
```

### Scheduler

The scheduler creates evaluation cycles and generates tasks for miners.

```bash
uv run kinitro scheduler \
  --database-url "$DATABASE_URL" \
  --network "ws://10.0.0.3:9944" \
  --netuid 2 \
  --eval-interval 300 \
  --log-level INFO
```

### Executor

The executor runs evaluation tasks against miner agents.

```bash
uv run kinitro executor \
  --api-url "$API_URL" \
  --eval-mode docker \
  --batch-size 3 \
  --poll-interval 5 \
  --eval-images '{"procthor":"kinitro/procthor:v1"}' \
  --env-families procthor \
  --log-level INFO
```

## Multi-Worktree Support

When working with multiple git worktrees in parallel, each worktree needs isolated services to avoid port collisions and database conflicts.

### Quick Start

```bash
# In your worktree directory
./scripts/worktree-env.sh    # Generate isolated config
./scripts/start-services.sh  # Start services
```

### How It Works

The `worktree-env.sh` script:

1. Calculates a deterministic port offset based on the worktree directory name
2. Generates a `.env` file with worktree-specific ports and database name
3. Generates a `docker-compose.override.yml` for container isolation

Example for worktree `fix/my-feature`:
- PostgreSQL: 5432 + offset (e.g., 5789)
- API: 8000 + offset (e.g., 8357)
- Database: `kinitro_fix_my_feature`

### Helper Scripts

| Script | Description |
|--------|-------------|
| `scripts/worktree-env.sh` | Generate worktree-isolated environment configuration |
| `scripts/start-services.sh` | Start/stop/manage services for current worktree |

### Commands

```bash
# Start all services
./scripts/start-services.sh start

# Stop all services
./scripts/start-services.sh stop

# Check service status
./scripts/start-services.sh status

# Tail service logs
./scripts/start-services.sh logs
```

## Building Environment Images

Before running evaluations, build the required Docker images:

```bash
# Build ProcTHOR environment
uv run kinitro env build procthor --tag kinitro/procthor:v1

# Build MetaWorld environment
uv run kinitro env build metaworld --tag kinitro/metaworld:v1
```

## Checking Logs

### Service Logs

Service logs are stored in `/tmp/kinitro_<worktree_name>/`:

```bash
# API logs
tail -f /tmp/kinitro_<worktree>/api.log

# Scheduler logs
tail -f /tmp/kinitro_<worktree>/scheduler.log

# Executor logs
tail -f /tmp/kinitro_<worktree>/executor.log
```

### Docker Container Logs

For environment containers (ProcTHOR, MetaWorld, etc.):

```bash
# List running containers
docker ps --filter "name=kinitro-eval"

# View container logs
docker logs <container_name>

# Follow logs in real-time
docker logs -f <container_name>
```

## Troubleshooting

### Port Conflicts

If you see "address already in use" errors:

1. Check if another worktree's services are running on the same ports
2. Regenerate environment: `./scripts/worktree-env.sh`
3. Stop conflicting services: `./scripts/start-services.sh stop`

### Database Issues

```bash
# Reset database
PGPASSWORD=postgres psql -h localhost -p $POSTGRES_PORT -U postgres \
  -c "DROP DATABASE IF EXISTS $POSTGRES_DB;"
PGPASSWORD=postgres psql -h localhost -p $POSTGRES_PORT -U postgres \
  -c "CREATE DATABASE $POSTGRES_DB;"
```

### Container Issues

```bash
# Stop all evaluation containers
docker stop $(docker ps -q --filter "name=kinitro-eval")

# Remove stopped containers
docker container prune -f
```
