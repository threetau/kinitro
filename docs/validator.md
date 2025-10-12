---
section: 'Get Started'
---

# Validator

Validators are responsible for evaluating the performance of miner-submitted agents on a variety of tasks.

## Setup

### Setting up environment variables
Copy the `.env.validator.example` file to `.env` and fill in the required environment variables:
```bash
cp .env.validator.example .env
```

You will need to create an R2 bucket and set the relevant environment variables. This is required for storing some evaluation data. For more information please refer to Cloudflare's [R2 documentation](https://developers.cloudflare.com/r2/buckets/). 

You will need to set `KINITRO_API_KEY` to obtain access to the Kinitro backend. Please contact us on our [discord channel](https://discord.gg/96SdmpeMqG) for access.

### Configuration
To configure the validator websocket app, you will need to create a configuration file. You can start by copying the example configuration file:

```bash
cp config/validator.toml.example validator.toml
```
Edit `validator.toml` to set your desired parameters, such as the Bittensor wallet to use, the backend websocket URL, and other settings.

You will also need to set up the evaluator configuration file. You can start by copying the example configuration file:

```bash
cp config/evaluator.toml.example evaluator.toml
```
Edit `evaluator.toml` to set your desired parameters, such as the PostgreSQL database connection string, R2 credentials, and logging intervals.

Key resource knobs in `evaluator.toml`:
- `ray_num_cpus`, `ray_num_gpus`, `ray_memory_gb`, `ray_object_store_memory_gb` – tune the Ray head resources the orchestrator reserves when it boots.
- `worker_num_cpus`, `worker_num_gpus`, `worker_memory_gb`, `worker_max_restarts`, `worker_max_task_retries` – control how much CPU/GPU/memory each rollout worker actor requests from Ray.

### Setting up database
The validator requires a PostgreSQL database for queuing evaluation jobs and results.

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

### Running the Websocket app
The websocket app will connect to the Kinitro backend, listen for evaluation jobs, and forward them to the evaluator to execute.
Once your configuration file is set up, you can run the validator using the following command:
```bash
python -m validator --config validator.toml
```

### Running the Evaluator
The evaluator is responsible for executing the evaluation jobs received from the websocket app. It will run the agents in the specified environments and log the results to the database.
To start the evaluator, use the following command:
```bash
python -m evaluator.orchestrator --config evaluator.toml
```

## Containerized deployment

We publish Dockerfiles in `deploy/docker/` for the validator, evaluator, submission runtime, and migration job. Build the images locally:
```bash
docker compose -f deploy/docker/compose.yaml build
```

Create configuration directories with your `validator.toml`, `evaluator.toml`, and `.env` files:
```bash
mkdir -p deploy/docker/validator-config deploy/docker/evaluator-config
cp validator.toml deploy/docker/validator-config/
cp evaluator.toml deploy/docker/evaluator-config/
cp .env deploy/docker/
```

Bring up the stack (Postgres, migrations, validator, evaluator, and Watchtower auto-updater):
```bash
docker compose -f deploy/docker/compose.yaml up -d postgres validator evaluator watchtower
```

> **Note:** The compose file references `KINITRO_API_KEY`. Export it in your shell (`export KINITRO_API_KEY=...`) or populate `deploy/docker/validator-config/.env` before starting the stack to avoid a blank default.

> **Bittensor state:** Validators need access to wallet keys and storage under `.bittensor`. Mount it via `BITTENSOR_HOME` before starting Docker Compose (defaults to `$HOME/.bittensor` if unset):
> ```bash
> export BITTENSOR_HOME="$HOME/.bittensor"
> ```

The helper script `scripts/update_validator.sh` wraps `docker compose pull`, runs migrations using the `migrator` profile, and restarts the services. Add a systemd timer or cron entry that runs this script nightly to keep validators current. Set `USE_GPU_EVALUATOR=1` to switch from the CPU evaluator to the GPU deployment defined in the compose file.

### GPU-enabled evaluators

- GPU nodes must install the NVIDIA driver and NVIDIA Container Toolkit.
- Build the CUDA images (`Dockerfile.evaluator-cuda`, `Dockerfile.miner-agent-cuda`) and enable the `gpu` profile in Docker Compose:
  ```bash
  docker compose -f deploy/docker/compose.yaml --profile gpu up -d evaluator-gpu
  ```
- Submission pods request GPUs via `src/evaluator/containers/podspec.yaml` (set `nvidia.com/gpu` limits when the competition requires it).

### Kubernetes manifests

Reference manifests live in `deploy/k8s/`:

- `00-namespace.yaml` / `01-rbac.yaml` bootstrap namespace isolation and evaluator permissions to create submission pods.
- `03-postgres.yaml` provisions a PostgreSQL `StatefulSet` in-cluster (swap for managed Postgres if available).
- `10-migrator-job.yaml` applies Alembic migrations before each rollout.
- `20-validator-deployment.yaml` and `30-evaluator-deployment.yaml` run the CPU stack.
- `31-evaluator-gpu-deployment.yaml` targets GPU nodes with the CUDA evaluator image.
- `40-rollout-cronjob.yaml` restarts deployments nightly so new images roll out automatically.

Apply the manifests with `kubectl apply -f deploy/k8s/` after filling in backend URLs, database credentials, and R2 configuration. Ensure the NVIDIA device plugin is installed on GPU node pools before applying the GPU deployment manifest.

### Image matrix

| Image | Purpose | Variant |
| --- | --- | --- |
| `ghcr.io/threetau/kinitro-validator` | WebSocket validator service | CPU |
| `ghcr.io/threetau/kinitro-evaluator` | Orchestrator & Ray rollout workers | CPU / `-gpu` |
| `ghcr.io/threetau/kinitro-miner-agent` | Submission runtime for Kubernetes pods | CPU / `-gpu` |
| `kinitro-migrator` (local build) | Alembic + pgq migrations | CPU |

For local development or private registries, use the Docker Compose `build` targets to push images to your infrastructure. Update `deploy/docker/compose.yaml` or the Kubernetes manifests to point at your registry/tag naming scheme.
