# Kinitro Validator Kubernetes Manifests

This directory contains reference manifests for running the validator stack in Kubernetes.

## Files
- `00-namespace.yaml` – Namespace isolation for all validator resources.
- `01-rbac.yaml` – ServiceAccount and Role allowing the evaluator to create submission pods and trigger rollouts.
- `02-configs.yaml` – Sample config maps (`validator.toml`, `evaluator.toml`) and API key secret.
- `03-postgres.yaml` – StatefulSet providing a dedicated validator Postgres instance.
- `10-migrator-job.yaml` – Job that applies Alembic migrations before rolling out workloads.
- `20-validator-deployment.yaml` – WebSocket validator deployment.
- `30-evaluator-deployment.yaml` – CPU evaluator orchestrator.
- `31-evaluator-gpu-deployment.yaml` – GPU evaluator variant requesting an NVIDIA GPU.
- `40-rollout-cronjob.yaml` – Nightly CronJob that restarts deployments to pick up new images.

## Usage
1. **Bootstrap namespace & RBAC**
   ```bash
   kubectl apply -f deploy/k8s/00-namespace.yaml
   kubectl apply -f deploy/k8s/01-rbac.yaml
   ```
2. **Provide configuration** – Edit `02-configs.yaml` to match your backend URLs, database URIs, and Cloudflare R2 credentials. Apply the manifests:
   ```bash
   kubectl apply -f deploy/k8s/02-configs.yaml
   kubectl apply -f deploy/k8s/03-postgres.yaml
   ```
3. **Run migrations**
   ```bash
   kubectl apply -f deploy/k8s/10-migrator-job.yaml
   kubectl wait --for=condition=complete job/validator-migrator -n kinitro-validator
   ```
4. **Deploy services**
   ```bash
   kubectl apply -f deploy/k8s/20-validator-deployment.yaml
   kubectl apply -f deploy/k8s/30-evaluator-deployment.yaml
   # GPU variant (optional)
   kubectl apply -f deploy/k8s/31-evaluator-gpu-deployment.yaml
   ```
5. **Nightly rollouts**
   ```bash
   kubectl apply -f deploy/k8s/40-rollout-cronjob.yaml
   ```

### GPU Nodes
Ensure the cluster has the NVIDIA device plugin installed. Update `nodeSelector`/`tolerations` in `31-evaluator-gpu-deployment.yaml` to match your GPU pool labels. Submission pods created by the evaluator should also request GPUs in `src/evaluator/containers/podspec.yaml`.

### Managed Postgres
If you use a managed PostgreSQL service, omit `03-postgres.yaml` and update the secrets/config maps to reference the external endpoint.
