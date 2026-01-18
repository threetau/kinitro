# Backend Guide (Subnet Operator)

This guide explains how to run the evaluation backend for the Robotics Generalization Subnet. The backend is operated by the **subnet owner** and handles all compute-heavy evaluation of miner policies.

> **Note**: This guide is for subnet operators only. If you're a validator, see the [Validator Guide](./validator-guide.md). If you're a miner, see the [Miner Guide](./miner-guide.md).

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATION BACKEND (this component)                             │
│    - PostgreSQL database for storing results                     │
│    - FastAPI server exposing weights API                         │
│    - Scheduler running periodic evaluation cycles                │
│    - affinetes containers for isolated simulation                │
│    - Calls miner Chutes endpoints for policy actions             │
│    - Computes epsilon-Pareto scores and weights                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP API
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  VALIDATORS (run by community)                                   │
│    - Poll GET /v1/weights/latest                                 │
│    - Submit weights to chain                                     │
└─────────────────────────────────────────────────────────────────┘
```

## Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 16GB+ (for MuJoCo simulation)
- **GPU**: Optional but recommended for faster evaluation
- **Storage**: 50GB+ for Docker images and database
- **Docker**: Required for affinetes container-based evaluation
- **PostgreSQL**: 13+ for storing evaluation results

## Quick Start

### 1. Install the Package

```bash
git clone https://github.com/your-org/robo-subnet.git
cd robo-subnet
uv sync
```

### 2. Set Up PostgreSQL

Option A - Docker (recommended):
```bash
docker run -d \
  --name robo-postgres \
  -e POSTGRES_USER=robo \
  -e POSTGRES_PASSWORD=your-secure-password \
  -e POSTGRES_DB=robo \
  -p 5432:5432 \
  -v robo_postgres_data:/var/lib/postgresql/data \
  postgres:15
```

Option B - System PostgreSQL:
```bash
sudo -u postgres createuser robo
sudo -u postgres createdb robo -O robo
sudo -u postgres psql -c "ALTER USER robo PASSWORD 'your-secure-password';"
```

### 3. Build the Evaluation Environment

The evaluation runs in Docker containers managed by affinetes:

```bash
uv run robo build-eval-env --tag robo-subnet/eval-env:v1
```

This builds a self-contained image with:
- MuJoCo and MetaWorld simulation
- HTTP client for calling miner endpoints
- All environment wrappers

### 4. Initialize the Database

```bash
# Create database (skip if using Docker - already created)
uv run robo db create --database-url postgresql://robo:your-secure-password@localhost/robo

# Initialize schema
uv run robo db init --database-url postgresql://robo:your-secure-password@localhost/robo
```

### 5. Start the Backend

```bash
uv run robo backend \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --database-url postgresql://robo:your-secure-password@localhost/robo \
  --eval-interval 3600 \
  --episodes-per-env 50
```

The backend will:
- Start the FastAPI server on port 8000
- Begin periodic evaluation cycles
- Discover miners from chain commitments
- Run evaluations in Docker containers
- Store results in PostgreSQL
- Expose weights via REST API

## Configuration

### Command Line Options

| Flag | Environment Variable | Default | Description |
|------|---------------------|---------|-------------|
| `--host` | `ROBO_BACKEND_HOST` | `0.0.0.0` | API server bind address |
| `--port` | `ROBO_BACKEND_PORT` | `8000` | API server port |
| `--database-url` | `ROBO_BACKEND_DATABASE_URL` | - | PostgreSQL connection URL |
| `--network` | `ROBO_BACKEND_NETWORK` | `finney` | Bittensor network |
| `--netuid` | `ROBO_BACKEND_NETUID` | - | Subnet UID |
| `--eval-interval` | `ROBO_BACKEND_EVAL_INTERVAL_SECONDS` | `3600` | Seconds between eval cycles |
| `--episodes-per-env` | `ROBO_BACKEND_EPISODES_PER_ENV` | `50` | Episodes per environment |
| `--log-level` | `ROBO_BACKEND_LOG_LEVEL` | `INFO` | Logging level |

### Advanced Settings (Environment Variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `ROBO_BACKEND_EVAL_IMAGE` | `robo-subnet/eval-env:v1` | Docker image for evaluation |
| `ROBO_BACKEND_EVAL_MEM_LIMIT` | `8g` | Memory limit for eval container |
| `ROBO_BACKEND_ACTION_TIMEOUT_MS` | `100` | Timeout for miner responses |
| `ROBO_BACKEND_MAX_TIMESTEPS_PER_EPISODE` | `500` | Max steps per episode |
| `ROBO_BACKEND_PARETO_TEMPERATURE` | `1.0` | Softmax temperature for weights |

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
      "endpoint": "https://miner0.chutes.ai",
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
uv run robo db status --database-url postgresql://robo:pass@localhost/robo
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
uv run robo db reset --database-url postgresql://robo:pass@localhost/robo --force
```

### Backup

```bash
pg_dump -h localhost -U robo robo > backup_$(date +%Y%m%d).sql
```

### Restore

```bash
psql -h localhost -U robo robo < backup_20240115.sql
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
      POSTGRES_USER: robo
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_DB: robo
    volumes:
      - postgres_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U robo"]
      interval: 5s
      timeout: 5s
      retries: 5
    restart: always

  backend:
    build: .
    command: >
      robo backend
      --netuid ${NETUID}
      --network ${NETWORK}
      --database-url postgresql://robo:${POSTGRES_PASSWORD}@postgres/robo
      --eval-interval ${EVAL_INTERVAL:-3600}
      --episodes-per-env ${EPISODES_PER_ENV:-50}
    depends_on:
      postgres:
        condition: service_healthy
    ports:
      - "8000:8000"
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock  # Required for affinetes
    environment:
      - ROBO_BACKEND_EVAL_IMAGE=robo-subnet/eval-env:v1
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
```

Run:
```bash
docker-compose -f docker-compose.backend.yml up -d
```

### Systemd Service

```bash
# /etc/systemd/system/robo-backend.service
[Unit]
Description=Robotics Subnet Backend
After=network.target postgresql.service docker.service

[Service]
Type=simple
User=robo
WorkingDirectory=/opt/robo-subnet
Environment=ROBO_BACKEND_DATABASE_URL=postgresql://robo:password@localhost/robo
ExecStart=/opt/robo-subnet/.venv/bin/python -m robo.cli backend \
  --netuid 123 \
  --network finney \
  --eval-interval 3600
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

### Reverse Proxy (nginx)

```nginx
server {
    listen 443 ssl;
    server_name backend.robo-subnet.ai;

    ssl_certificate /etc/letsencrypt/live/backend.robo-subnet.ai/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/backend.robo-subnet.ai/privkey.pem;

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
# Increase verbosity
uv run robo backend --log-level DEBUG ...

# View structured logs
uv run robo backend ... 2>&1 | jq .
```

### Prometheus Metrics

The backend exposes metrics at `/metrics`:
- `robo_eval_cycles_total` - Total evaluation cycles
- `robo_eval_duration_seconds` - Evaluation duration histogram
- `robo_miners_active` - Currently active miners
- `robo_miner_scores` - Per-miner scores by environment

### Health Checks

```bash
# Backend health
curl http://localhost:8000/health

# Database status
uv run robo db status --database-url postgresql://...

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
- Verify eval-env image exists: `docker images | grep eval-env`

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
- Verify image exists: `docker images | grep robo-subnet`
- Check memory limits aren't too restrictive
- Ensure Docker socket is accessible

### Memory issues
- Increase `ROBO_BACKEND_EVAL_MEM_LIMIT`
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

## Scaling

### Horizontal Scaling

For high miner counts, run multiple evaluation workers:
- Use a job queue (Redis, RabbitMQ) for evaluation tasks
- Run multiple backend instances with shared database
- Load balance API requests

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
