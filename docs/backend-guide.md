# Backend Guide (Subnet Operator)

This guide explains how to run the evaluation backend for Kinitro. The backend is operated by the **subnet owner** and handles all compute-heavy evaluation of miner policies.

> **Note**: This guide is for subnet operators only. If you're a validator, see the [Validator Guide](./validator-guide.md). If you're a miner, see the [Miner Guide](./miner-guide.md).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATION BACKEND (this component)                            │
│    - PostgreSQL database for storing results                    │
│    - FastAPI server exposing weights API                        │
│    - Scheduler running periodic evaluation cycles               │
│    - Executors download miner models and create deployments     │
│    - affinetes containers for isolated simulation               │
│    - Computes epsilon-Pareto scores and weights                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP API
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATORS (run by community)                                  │
│    - Poll GET /v1/weights/latest                                │
│    - Submit weights to chain                                    │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16 GiB+ (for MuJoCo simulation)
- **GPU**: Optional, but recommended for faster evaluation
- **Storage**: 50 GiB+ for Docker images and database
- **Docker**: Required for Affinetes container-based evaluation
- **PostgreSQL**: 13+ for storing evaluation results

## Quick Start

### 1. Install the Package

```bash
git clone https://github.com/threetau/kinitro.git
cd kinitro
uv sync
```

### 2. Set Up PostgreSQL

Option A - Docker (recommended):

```bash
docker run -d \
  --name kinitro-postgres \
  -e POSTGRES_USER=kinitro \
  -e POSTGRES_PASSWORD=your-secure-password \
  -e POSTGRES_DB=kinitro \
  -p 5432:5432 \
  -v kinitro_postgres_data:/var/lib/postgresql/data \
  postgres:15
```

Option B - System PostgreSQL:

```bash
sudo -u postgres createuser kinitro
sudo -u postgres createdb kinitro -O kinitro
sudo -u postgres psql -c "ALTER USER kinitro PASSWORD 'your-secure-password';"
```

### 3. Build the Docker Images

The backend uses Docker containers managed by Affinetes. Build the required images:

```bash
# Build MetaWorld evaluation environment (~1GB image, works on any platform)
uv run kinitro env build metaworld --tag kinitro/metaworld:v1

# Build ProcTHOR evaluation environment (~3GB image, requires x86_64 Linux)
uv run kinitro env build procthor --tag kinitro/procthor:v1

# Build miner policy runner (~200MB image, for Docker mode deployments)
uv run kinitro env build miner --tag kinitro/miner-runner:v1
```

**Evaluation environment images** are self-contained with:
- Simulation engine (MuJoCo for MetaWorld, ProcTHOR renderer on AI2-THOR framework)
- HTTP client for calling miner endpoints
- Environment wrappers and dependencies

**Miner runner image** downloads and runs miner policies from HuggingFace. Only needed for Docker mode (`eval_mode=docker`).

**Note**: ProcTHOR requires native x86_64 Linux and will not work on ARM64 or under emulation.

### 4. Initialize the Database

```bash
# Create database (skip if using Docker - already created)
uv run kinitro db create --database-url postgresql://kinitro:your-secure-password@localhost/kinitro

# Initialize schema
uv run kinitro db init --database-url postgresql://kinitro:your-secure-password@localhost/kinitro
```

### 5. Start the Backend Services

The backend uses a split architecture with three services:

```bash
# Terminal 1: API Service (stateless REST API)
uv run kinitro api \
  --database-url postgresql://kinitro:your-secure-password@localhost/kinitro

# Terminal 2: Scheduler Service (task generation and scoring)
uv run kinitro scheduler \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --database-url postgresql://kinitro:your-secure-password@localhost/kinitro \
  --eval-interval 3600 \
  --episodes-per-env 50

# Terminal 3+: Executor Service(s) (run evaluations - can run multiple)
uv run kinitro executor \
  --api-url http://localhost:8000
```

The services work together:

- **API**: Provides REST endpoints for validators and executors
- **Scheduler**: Reads miner commitments, creates tasks, computes weights
- **Executor**: Fetches tasks from API, runs evaluations in Docker containers

## Configuration

Each service (API, Scheduler, Executor) has its own configuration with service-specific environment variable prefixes.

### API Service Configuration

| CLI Flag         | Environment Variable       | Default                    | Description               |
| ---------------- | -------------------------- | -------------------------- | ------------------------- |
| `--host`         | `KINITRO_API_HOST`         | `0.0.0.0`                  | API server bind address   |
| `--port`         | `KINITRO_API_PORT`         | `8000`                     | API server port           |
| `--database-url` | `KINITRO_API_DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection URL |
| `--log-level`    | `KINITRO_API_LOG_LEVEL`    | `INFO`                     | Logging level             |

Additional API settings:

| Environment Variable                       | Default | Description                                  |
| ------------------------------------------ | ------- | -------------------------------------------- |
| `KINITRO_API_TASK_STALE_THRESHOLD_SECONDS` | `300`   | Time after which assigned tasks become stale |

### Scheduler Service Configuration

| CLI Flag             | Environment Variable                      | Default                    | Description                 |
| -------------------- | ----------------------------------------- | -------------------------- | --------------------------- |
| `--database-url`     | `KINITRO_SCHEDULER_DATABASE_URL`          | `postgresql+asyncpg://...` | PostgreSQL connection URL   |
| `--network`          | `KINITRO_SCHEDULER_NETWORK`               | `finney`                   | Bittensor network           |
| `--netuid`           | `KINITRO_SCHEDULER_NETUID`                | `1`                        | Subnet UID                  |
| `--eval-interval`    | `KINITRO_SCHEDULER_EVAL_INTERVAL_SECONDS` | `3600`                     | Seconds between eval cycles |
| `--episodes-per-env` | `KINITRO_SCHEDULER_EPISODES_PER_ENV`      | `50`                       | Episodes per environment    |
| `--log-level`        | `KINITRO_SCHEDULER_LOG_LEVEL`             | `INFO`                     | Logging level               |

Additional Scheduler settings:

| Environment Variable                             | Default | Description                                  |
| ------------------------------------------------ | ------- | -------------------------------------------- |
| `KINITRO_SCHEDULER_PARETO_TEMPERATURE`           | `1.0`   | Softmax temperature for weight conversion    |
| `KINITRO_SCHEDULER_TASK_STALE_THRESHOLD_SECONDS` | `300`   | Time after which assigned tasks become stale |
| `KINITRO_SCHEDULER_CYCLE_TIMEOUT_SECONDS`        | `7200`  | Maximum time to wait for cycle completion    |

### Executor Service Configuration

| CLI Flag          | Environment Variable                     | Default                 | Description                             |
| ----------------- | ---------------------------------------- | ----------------------- | --------------------------------------- |
| `--api-url`       | `KINITRO_EXECUTOR_API_URL`               | `http://localhost:8000` | URL of the Kinitro API service          |
| `--batch-size`    | `KINITRO_EXECUTOR_BATCH_SIZE`            | `10`                    | Number of tasks to fetch at a time      |
| `--poll-interval` | `KINITRO_EXECUTOR_POLL_INTERVAL_SECONDS` | `5`                     | Seconds between polling for tasks       |
| `--eval-images`   | `KINITRO_EXECUTOR_EVAL_IMAGES`           | See below               | Env family to Docker image mapping      |
| `--eval-mode`     | `KINITRO_EXECUTOR_EVAL_MODE`             | `docker`                | Evaluation mode: 'docker' or 'basilica' |
| `--log-level`     | `KINITRO_EXECUTOR_LOG_LEVEL`             | `INFO`                  | Logging level                           |

**Eval Images Configuration:**

The executor uses different Docker images for each environment family. The default configuration is:

```json
{
  "metaworld": "kinitro/metaworld:v1",
  "procthor": "kinitro/procthor:v1"
}
```

You can override this via CLI or environment variable:

```bash
# Via CLI
uv run kinitro executor --eval-images '{"metaworld": "myregistry/metaworld:latest"}'

# Via environment variable
export KINITRO_EXECUTOR_EVAL_IMAGES='{"metaworld": "kinitro/metaworld:v1", "procthor": "kinitro/procthor:v1"}'
```

The executor automatically selects the correct image based on each task's environment family. This allows a single executor to handle all environment types.

Additional Executor settings:

| Environment Variable              | Default             | Description                                 |
| --------------------------------- | ------------------- | ------------------------------------------- |
| `KINITRO_EXECUTOR_EXECUTOR_ID`    | `executor-<random>` | Unique identifier for this executor         |
| `KINITRO_EXECUTOR_ENV_IDS`        | `null`              | Filter tasks by environment IDs (JSON list) |
| `KINITRO_EXECUTOR_EVAL_MEM_LIMIT` | `8g`                | Memory limit for evaluation container       |
| `KINITRO_EXECUTOR_EVAL_HOSTS`     | `["localhost"]`     | Docker hosts for evaluation (JSON list)     |
| `KINITRO_EXECUTOR_MAX_TIMESTEPS`  | `500`               | Maximum timesteps per episode               |
| `KINITRO_EXECUTOR_ACTION_TIMEOUT` | `0.5`               | Timeout for miner responses (seconds)       |
| `KINITRO_EXECUTOR_EVAL_TIMEOUT`   | `300`               | Timeout for individual evaluation (seconds) |
| `KINITRO_EXECUTOR_USE_IMAGES`     | `true`              | Include camera images in observations       |

### Executor Miner Deployment Settings

The executor manages miner policy deployments on-demand via affinetes. When a task requires a miner's policy, the executor downloads the model from HuggingFace and creates a temporary deployment with TTL-based caching.

Miner deployments use the same `eval_mode` as evaluation environments:
- **Docker mode** (`eval_mode=docker`): Runs miner policies in local Docker containers
- **Basilica mode** (`eval_mode=basilica`): Deploys to Basilica cloud platform

| Variable                                           | Default | Description                                       |
| -------------------------------------------------- | ------- | ------------------------------------------------- |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_ENABLED`        | `true`  | Enable executor-managed miner deployments         |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_IMAGE`          | `kinitro/miner-runner:v1` | Docker image for miner policies |
| `KINITRO_EXECUTOR_MINER_BASILICA_API_TOKEN`        | `null`  | Basilica API token (required for basilica mode)   |
| `KINITRO_EXECUTOR_MINER_HF_TOKEN`                  | `null`  | HuggingFace token for private repos (optional)    |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_TTL_SECONDS`    | `600`   | TTL for idle deployments before cleanup           |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_WARMUP_TIMEOUT` | `300`   | Timeout for deployment to become ready (seconds)  |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_GPU_COUNT`      | `0`     | Number of GPUs for miner deployments              |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_MIN_GPU_MEMORY_GB` | `null` | Minimum GPU memory (GB) for miner deployments   |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_CPU`            | `1`     | CPU allocation for miner deployments              |
| `KINITRO_EXECUTOR_MINER_DEPLOYMENT_MEMORY`         | `4Gi`   | Memory allocation for miner deployments           |

**How it works:**

1. Miner commits `repo:revision` on-chain (no need to deploy themselves)
2. Scheduler creates tasks with `miner_repo` and `miner_revision` (no endpoint)
3. Executor fetches task and creates deployment on-demand (Docker or Basilica)
4. Deployments are cached by `(repo, revision)` tuple with TTL
5. After batch completion, expired deployments are cleaned up
6. On shutdown, all managed deployments are terminated

This approach simplifies the miner workflow (just upload to HuggingFace) while ensuring evaluation integrity (executors control the deployment).

## API Reference

The backend exposes a REST API that validators use to fetch weights.

### Health & Status

#### `GET /health`

Health check endpoint.

```json
{"status": "healthy", "database": "connected"}
```

#### `GET /v1/status`

Current backend status.

```json
{
  "status": "running",
  "current_block": 12345,
  "last_eval_block": 12300,
  "next_eval_in_seconds": 1800,
  "miners_registered": 15,
  "environments_enabled": ["metaworld/pick-place-v3", "metaworld/push-v3"]
}
```

### Weights (Used by Validators)

#### `GET /v1/weights/latest`

Get the latest computed weights.

```json
{
  "block": 12300,
  "weights": {
    "0": 0.15,
    "1": 0.25,
    "5": 0.60
  },
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### `GET /v1/weights/{block}`

Get weights for a specific block.

### Scores

#### `GET /v1/scores/latest`

Get latest evaluation scores.

```json
{
  "cycle_id": 42,
  "block": 12300,
  "scores": {
    "0": {
      "metaworld/pick-place-v3": 0.85,
      "metaworld/push-v3": 0.72
    },
    "1": {
      "metaworld/pick-place-v3": 0.90,
      "metaworld/push-v3": 0.88
    }
  },
  "pareto_frontier": [1],
  "created_at": "2024-01-15T10:30:00Z"
}
```

#### `GET /v1/scores/{cycle_id}`

Get scores for a specific evaluation cycle.

### Miners

#### `GET /v1/miners`

List all discovered miners.

```json
{
  "miners": [
    {
      "uid": 0,
      "hotkey": "5F...",
      "endpoint": "https://xxx.deployments.basilica.ai",
      "last_evaluated": "2024-01-15T10:30:00Z",
      "status": "active"
    }
  ]
}
```

### Environments

#### `GET /v1/environments`

List enabled evaluation environments.

## Database Management

### View Status

```bash
uv run kinitro db status --database-url postgresql://kinitro:pass@localhost/kinitro
```

Shows:

- Total evaluation cycles
- Unique miners evaluated
- Latest completed cycle info
- Currently running cycle (if any)
- Latest weights

### Reset Database

```bash
# WARNING: Destroys all data
uv run kinitro db reset --database-url postgresql://kinitro:pass@localhost/kinitro --force
```

### Backup

```bash
pg_dump -h localhost -U kinitro kinitro > backup_$(date +%Y%m%d).sql
```

### Restore

```bash
psql -h localhost -U kinitro kinitro < backup_20240115.sql
```

## Production Deployment

### Docker Compose

Create `docker-compose.backend.yml`:

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: kinitro
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: kinitro
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U kinitro"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: always

  api:
    build: .
    command: >
      kinitro api
      --database-url postgresql://kinitro:${POSTGRES_PASSWORD}@postgres/kinitro
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "8000:8000"
    restart: always

  scheduler:
    build: .
    command: >
      kinitro scheduler
      --netuid ${NETUID}
      --network ${NETWORK}
      --database-url postgresql://kinitro:${POSTGRES_PASSWORD}@postgres/kinitro
      --eval-interval ${EVAL_INTERVAL:-3600}
      --episodes-per-env ${EPISODES_PER_ENV:-50}
    depends_on:
      postgres:
        condition: service_healthy
    restart: always

  executor:
    build: .
    command: >
      kinitro executor
      --api-url http://api:8000
    environment:
      - KINITRO_EXECUTOR_MINER_BASILICA_API_TOKEN=${BASILICA_API_TOKEN}
      - KINITRO_EXECUTOR_MINER_HF_TOKEN=${HF_TOKEN}
    depends_on:
      api:
        condition: service_started
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # Required for affinetes
    restart: always

volumes:
  postgres_data:
```

Create `.env`:

```bash
POSTGRES_PASSWORD=your-secure-password
NETUID=123
NETWORK=finney
EVAL_INTERVAL=3600
EPISODES_PER_ENV=50
BASILICA_API_TOKEN=your-basilica-token
HF_TOKEN=your-huggingface-token  # Optional, for private repos
```

Run:

```bash
docker-compose -f docker-compose.backend.yml up -d
```

### Systemd Services

Create separate services for each component:

```bash
# /etc/systemd/system/kinitro-api.service
[Unit]
Description=Kinitro API Service
After=network.target postgresql.service

[Service]
Type=simple
User=kinitro
WorkingDirectory=/opt/kinitro
ExecStart=/opt/kinitro/.venv/bin/uv run kinitro api \
  --database-url postgresql://kinitro:password@localhost/kinitro
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# /etc/systemd/system/kinitro-scheduler.service
[Unit]
Description=Kinitro Scheduler Service
After=network.target postgresql.service

[Service]
Type=simple
User=kinitro
WorkingDirectory=/opt/kinitro
ExecStart=/opt/kinitro/.venv/bin/uv run kinitro scheduler \
  --netuid 123 \
  --network finney \
  --database-url postgresql://kinitro:password@localhost/kinitro \
  --eval-interval 3600
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# /etc/systemd/system/kinitro-executor.service
[Unit]
Description=Kinitro Executor Service
After=network.target docker.service kinitro-api.service

[Service]
Type=simple
User=kinitro
WorkingDirectory=/opt/kinitro
ExecStart=/opt/kinitro/.venv/bin/uv run kinitro executor \
  --api-url http://localhost:8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Reverse Proxy (nginx)

```nginx
server {
    listen 443 ssl;
    server_name backend.kinitro.ai;

    ssl_certificate /etc/letsencrypt/live/backend.kinitro.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/backend.kinitro.ai/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring

### Logs

```bash
# Increase verbosity (for any service)
uv run kinitro scheduler --log-level DEBUG ...
uv run kinitro api --log-level DEBUG ...

# View structured logs
uv run kinitro scheduler ... 2>&1 | jq .
```

### Prometheus Metrics

The backend exposes metrics at `/metrics`:

- `kinitro_eval_cycles_total` - Total evaluation cycles
- `kinitro_eval_duration_seconds` - Evaluation duration histogram
- `kinitro_miners_active` - Currently active miners
- `kinitro_miner_scores` - Per-miner scores by environment

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Database status
uv run kinitro db status --database-url postgresql://...

# Check evaluation status
curl http://localhost:8000/v1/status
```

### Alerting

Monitor these conditions:

- Backend health endpoint returns non-200
- No new evaluation cycles in 2x eval_interval
- Database connection failures
- Docker socket access issues

## Troubleshooting

### "Database connection failed"

- Check PostgreSQL is running: `docker ps` or `systemctl status postgresql`
- Verify connection URL format: `postgresql://user:pass@host:port/dbname`
- Check network connectivity

### "Failed to build eval container"

- Ensure Docker is running: `docker ps`
- Check Docker socket permissions: `ls -la /var/run/docker.sock`
- Verify environment images exist: `docker images | grep kinitro`
- For ProcTHOR: ensure you're on native x86_64 Linux (not ARM64 or emulated)

### "No miners found"

- Check subnet has registered miners: `btcli subnet list --netuid YOUR_NETUID`
- Verify miners have valid commitments
- Check chain connectivity

### Evaluation taking too long

- Reduce `--episodes-per-env`
- Check miner endpoint latency
- Consider using GPU for simulation
- Review container resource limits

### "Container failed to start"

- Check Docker logs: `docker logs <container_id>`
- Verify image exists: `docker images | grep kinitro`
- Check memory limits aren't too restrictive
- Ensure Docker socket is accessible

### Memory issues

- Increase `KINITRO_BACKEND_EVAL_MEM_LIMIT`
- Monitor container memory: `docker stats`
- Consider running fewer concurrent evaluations

## Security Considerations

1. **Database**: Use strong passwords, restrict network access, enable SSL
2. **Backend API**:
   - Consider adding authentication for sensitive endpoints
   - Use TLS (HTTPS) in production
   - Rate limit API endpoints
3. **Docker**:
   - Limit container privileges
   - Use non-root user in containers
   - Restrict Docker socket access
4. **Network**:
   - Use firewall rules
   - Only expose port 8000 (or via reverse proxy)

### GPU Acceleration

For faster simulation:

- Use NVIDIA Docker runtime
- Set `MUJOCO_GL=egl` for headless rendering
- Allocate GPU to eval containers

## Additional Resources

- [Validator Guide](./validator-guide.md) - For validators polling this backend
- [Miner Guide](./miner-guide.md) - For miners participating in the subnet
- [affinetes](https://github.com/affine-lab/affinetes) - Container-based evaluation library
- [Bittensor Docs](https://docs.bittensor.com/) - Bittensor documentation
