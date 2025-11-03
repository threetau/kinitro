---
section: 'Start Validating'
---

# Validator

Validators are responsible for evaluating the performance of miner-submitted agents on a variety of tasks.

**Choose your validator type first:**

1. [Full Validator (full pipeline)](#full-validator-full-pipeline) – runs evaluations, streams logs, and requires the evaluator + Postgres stack.
2. [Lite Validator (HTTP weight setter)](#lite-validator-http-weight-setter) – polls the public weight endpoint and only sets weights on-chain.

After picking the implementation, choose how you want to deploy it:

1. [Bare Metal](#setup---bare-metal)
2. [Containerized deployment](#setup---containerized-deployment)

## Setup - Bare Metal

### Setting up environment variables

Copy the `.env.validator.example` file to `.env` and fill in the required environment variables:

```bash
cp .env.validator.example .env
```

You will need to create an R2 bucket and set the relevant environment variables. This is required for storing some evaluation data. For more information please refer to Cloudflare's [R2 documentation](https://developers.cloudflare.com/r2/buckets/).

If you are running a Full Validator (*not* a lite validator), You will need to set `KINITRO_API_KEY` to obtain access to the Kinitro backend. Please contact us on our [discord channel](https://discord.gg/96SdmpeMqG) for access.

### Configuration

To configure a validator, start by copying the example configuration file:

```bash
cp config/validator.toml.example validator.toml
```

The example config now includes both the Full validator and the lite HTTP-based weight setter. Core knobs look like:

```toml
validator_mode = "full"         # switch to "lite" to run the HTTP weight setter
weights_url = "https://api.kinitro.ai/weights"
weights_poll_interval = 30.0
weights_request_timeout = 10.0
weights_stale_threshold = 180.0
```

Use the default `weights_url` unless you operate your own backend; the lite mode polls this endpoint and pushes updates on-chain.

#### Full validator (full pipeline)

Set `validator_mode = "full"` to run the full evaluator pipeline. This mode:

- maintains a WebSocket connection to the backend for job distribution and telemetry,
- requires the PostgreSQL queue (`pg_database`) and evaluator service,
- forwards evaluation results back to the backend.

You will also need `evaluator.toml` for the orchestrator that executes jobs:

```bash
cp config/evaluator.toml.example evaluator.toml
```

Edit `evaluator.toml` to set your desired parameters, such as the PostgreSQL database connection string, R2 credentials, and logging intervals.

Key resource knobs in `evaluator.toml`:

- `ray_num_cpus`, `ray_num_gpus`, `ray_memory_gb`, `ray_object_store_memory_gb` – tune the Ray head resources the orchestrator reserves when it boots.
- `worker_num_cpus`, `worker_num_gpus`, `worker_memory_gb`, `worker_max_restarts`, `worker_max_task_retries` – control how much CPU/GPU/memory each rollout worker actor requests from Ray.

#### Lite validator (HTTP weight setter)

Set `validator_mode = "lite"` when you only need to mirror backend weight decisions on-chain. This mode:

- polls `weights_url` over HTTPS for the latest snapshot,
- uses your Bittensor wallet/hotkey to submit weights,
- does **not** require the evaluator service or Postgres queue.

You still need valid wallet credentials and chain connectivity, but no backend API key is required because the `/weights` endpoint is public.

### Setting up database

The Full validator requires a PostgreSQL database for queuing evaluation jobs and results. The lite validator can skip this section.

To set up the database, you can either:

1. **Reset the database** (drops and recreates the database with all migrations):

   ```bash
   chmod +x ./scripts/reset_validator_db.sh
   ./scripts/reset_validator_db.sh
   ```

2. **Run migrations only** (on an existing database):

   ```bash
   chmod +x ./scripts/migrate_validator_db.sh
   ./scripts/migrate_validator_db.sh
   ```

The migration script will check if the database exists and run Alembic migrations to bring it up to date. It will also ensure the pgq extension is installed if needed.

### Running the validator

Regardless of mode, launch the process with:

```bash
python -m validator --config validator.toml
```

- With `validator_mode = "full"` the service opens the backend WebSocket and requires the evaluator plus database to be running.
- With `validator_mode = "lite"` the service polls `/weights` and immediately applies updates on-chain. No evaluator or database is needed.

### Running the Evaluator

Only required for the Full validator. Start it once your validator is up:

```bash
python -m evaluator.orchestrator --config evaluator.toml
```

## Setup - Containerized deployment

We ship Docker recipes for the validator stack in `deploy/docker/`. The workflow below covers both CPU-only and GPU-enabled setups. The provided Compose profiles target the full validator; the lite validator can be run as a lightweight bare-metal process alongside the stack if desired.

### 1. Prerequisites

- **Docker Compose v2** (bundled with modern Docker releases).
- **Environment variables** – export `KINITRO_API_KEY` in your shell or place it in `deploy/docker/validator-config/.env` before you launch the stack.
- **Bittensor wallets** – point `BITTENSOR_HOME` at your wallet directory (defaults to `$HOME/.bittensor`). For example:

  ```bash
  export KINITRO_API_KEY=xxxxxxxx
  export BITTENSOR_HOME="$HOME/.bittensor"
  ```

- **GPU hosts only** – install the NVIDIA Container Toolkit (see the [official guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)).
- **Minikube (optional)** – required when you plan to run GPU evaluations; install it and start with GPU support as described in the [Minikube start documentation](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fbinary+download).

### 2. Prepare configuration files

Copy your bare-metal configs into the Compose folder so the containers mount them read-only:

```bash
mkdir -p deploy/docker/validator-config deploy/docker/evaluator-config
cp validator.toml deploy/docker/validator-config/
cp evaluator.toml deploy/docker/evaluator-config/
cp .env deploy/docker/.env                      # optional, keeps secrets out of the compose file
```

Keep `evaluator.toml` in sync with the resource hints you need (CPU/GPU counts, worker memory, etc.). The container reads these files from `/etc/kinitro` at runtime.

### 3. Run the CPU evaluator stack

The CPU evaluator lives in the `cpu` profile, so it will only start if you ask for it. Bring up the base services (Postgres, validator, watchtower) and the CPU evaluator:

```bash
docker compose -f deploy/docker/compose.yaml up -d postgres validator watchtower
docker compose -f deploy/docker/compose.yaml --profile cpu up -d evaluator
```

Use `scripts/update_validator.sh` to pull new images, apply migrations with the `migrator` profile, and restart the services automatically. Set `USE_GPU_EVALUATOR=1` when you want the helper script to restart the GPU profile instead of the CPU evaluator:

```bash
./scripts/update_validator.sh              # CPU stack
USE_GPU_EVALUATOR=1 ./scripts/update_validator.sh  # GPU stack
```

### 4. Run the GPU evaluator (Minikube + CUDA)

1. Start Minikube with GPU support so the evaluator can create submission pods that request GPUs:

   ```bash
   minikube start --driver=docker --gpu
   ```

   This also creates the external Docker network named `minikube`, which the evaluator containers join for API access.
2. Launch the GPU evaluator profile (CPU evaluator stays off unless you start the `cpu` profile):

   ```bash
   docker compose -f deploy/docker/compose.yaml --profile gpu --compatibility up -d evaluator-gpu
   ```

If you prefer to keep both profiles running, bring up each profile explicitly (`--profile cpu up -d evaluator` and `--profile gpu up -d evaluator-gpu`).

### Image matrix

| Image | Purpose | Variant |
| --- | --- | --- |
| `ghcr.io/threetau/kinitro-validator` | Full validator service | CPU |
| `ghcr.io/threetau/kinitro-evaluator` | Orchestrator & Ray rollout workers | CPU / `-gpu` |
| `ghcr.io/threetau/kinitro-miner-agent` | Submission runtime for evaluator-launched pods (Minikube) | CPU / `-gpu` |
| `kinitro-migrator` (local build) | Alembic + pgq migrations | CPU |

For local development or private registries, use the Docker Compose `build` targets to push images to your infrastructure. Update `deploy/docker/compose.yaml` to point at your registry/tag naming scheme.
