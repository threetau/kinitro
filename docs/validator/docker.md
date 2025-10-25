---
section: 'Start Validating'
---

# Validator - More about Containerization

## Images

- `validator.dockerfile` – WebSocket validator service (`ghcr.io/threetau/kinitro-validator`).
- `evaluator.dockerfile` – CPU evaluator orchestrator (`ghcr.io/threetau/kinitro-evaluator`).
- `evaluator-cuda.dockerfile` – CUDA-enabled evaluator (`ghcr.io/threetau/kinitro-evaluator:*-gpu`).
- `miner-agent.dockerfile` – Submission runtime image used by evaluator-created pods.
- `miner-agent-cuda.dockerfile` – GPU variant of the submission runtime.
- `migrator.dockerfile` – Alembic/pgq migration job image (locally tagged as `kinitro-migrator`).

Shared entrypoints live beside the Dockerfiles: `entrypoint-validator.sh`, `entrypoint-evaluator.sh`.

## Compose Stack

The stack is split across layered Compose files:

- `compose.base.yaml` – Postgres, migrator, validator, evaluator, watchtower.
- `compose.backend.yaml` – Backend API (optional when the backend runs elsewhere).
- `compose.local-dev.yaml` – Local overrides (MinIO, seed data, port mappings).

Combine the files you need. For example, the base validator deployment includes:

- `postgres` – Local Postgres instance.
- `migrator` – Runs migrations on demand (`--profile ops`).
- `validator`, `evaluator`, `evaluator-gpu` – Service containers.
- `watchtower` – Auto-updates images nightly (`WATCHTOWER_SCHEDULE` env).

Build everything locally:

```bash
docker compose -f deploy/docker/compose.base.yaml build
```

> [!NOTE] **Minikube networking:** The evaluator services join the external Docker network named `minikube` (created automatically when Minikube runs with the Docker driver). Make sure Minikube is running (`minikube start --driver=docker`; see the [Minikube start guide](https://minikube.sigs.k8s.io/docs/start/?arch=%2Fmacos%2Farm64%2Fstable%2Fbinary+download)) before launching the stack so this network exists.

> [!TIP] When running the full local stack, use `scripts/docker/local_setup.py bootstrap-demo` to generate API keys and seed the competition described in `deploy/docker/local/competition.json`. The script writes credentials into `deploy/docker/env/local.env` for the containers. The helper scripts automatically look for `$HOME/.kube/config` and `$HOME/.minikube`; export `HOST_KUBECONFIG` / `HOST_MINIKUBE_DIR` if you keep them elsewhere.

### CPU-based Evaluator

Bring the stack up after providing configuration files in `deploy/docker/config/base/` (the CPU evaluator lives in the `cpu` profile and won’t start unless you ask for that profile explicitly):

```bash
docker compose -f deploy/docker/compose.base.yaml up -d
```

### GPU-based Evaluator

Verify `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi` works before starting the GPU profile.

Update `deploy/docker/evaluator-config/evaluator.toml` (or your custom config) so Ray knows to request a GPU:

```toml
ray_num_cpus = 4
ray_num_gpus = 1  # set to 0 for CPU-only runs
```

Enable the GPU evaluator (the CPU evaluator stays off because it sits in the `cpu` profile):

```bash
docker compose -f deploy/docker/compose.base.yaml --profile gpu --compatibility up -d evaluator-gpu
```

Export `KINITRO_API_KEY` (or add it to `deploy/docker/env/base.env`) before starting the stack so the validator can authenticate with the backend.

Mount your Bittensor state directory (wallets + storb cache) by exporting an absolute path (defaults to `$HOME/.bittensor` if unset):

```bash
export BITTENSOR_HOME="$HOME/.bittensor"
```

Run migrations + restart (used by `scripts/update_validator.sh`):

```bash
docker compose -f deploy/docker/compose.base.yaml --profile ops run --rm migrator
docker compose -f deploy/docker/compose.base.yaml up -d validator evaluator
```

> [!NOTE] **GPU runtime prerequisites:** Install the NVIDIA Container Toolkit (follow the [official install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)) and restart Docker so the `nvidia` runtime is available:
>
>```bash
>sudo apt install -y nvidia-container-toolkit
>sudo nvidia-ctk runtime configure --runtime=docker
>sudo systemctl restart docker
>```
