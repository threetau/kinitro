---
section: 'Start Mining'
---

# Miner

## Setting up environment variables

Copy the `.env.miner.example` file to `.env` and fill in the required environment variables:

```bash
cp .env.miner.example .env
```

## Configuration

Copy the example configuration file, and edit it to include parameters like your backend URL, Bittensor wallet details, etc.

```bash
cp config/miner.toml.example miner.toml
```

Edit `miner.toml` to set your desired parameters.

## Uploading your agent

You can package your submission, submit an upload request, and upload the artifact directly to the validator-controlled vault.

To upload your agent, run:

```bash
python -m miner upload --config miner.toml
```

This command will:

- bundle the directory defined in `submission_dir` into `submission.tar.gz`
- calculate the SHA-256 hash and byte size of the archive
- sign an upload request with your hotkey keypair
- call the backend (`backend_url`) for a presigned upload URL
- upload the artifact to the vault

Ensure your `miner.toml` contains:

- `backend_url`: e.g. `https://api.kinitro.ai`
- `submission_dir`: path to your agent package
- wallet/hotkey configuration (used for signing)

## Local Docker stack

For end-to-end smoke tests against local infrastructure we ship layered Compose files under `deploy/docker/`. Use the helper script to combine them:

1. Copy the example environment and tweak credentials if needed:

   ```bash
   cp deploy/docker/env/local.env.example deploy/docker/env/local.env
   ```

   The helper script automatically reuses `$HOME/.kube/config` / `$HOME/.minikube` when they exist so the evaluator can reach your Minikube cluster. When `kubectl` (or PyYAML) is available it flattens the kubeconfig and inlines the certificates into `deploy/docker/config/local/_generated/kubeconfig.yaml`; otherwise it falls back to copying the `.minikube` assets with relaxed permissions for the non-root `kinitro` user. Rerun the script if your kubeconfig changes, and export `HOST_KUBECONFIG` / `HOST_MINIKUBE_DIR` first if you keep them elsewhere.

2. Bring up the full local stack (Postgres, backend, validator, evaluator, MinIO):

   ```bash
   scripts/docker/stack.sh up local --profile cpu --build
   ```

   The backend is now reachable at `http://localhost:8080`, MinIO at `http://localhost:9000`, and the optional console at `http://localhost:9001`.

3. Generate local credentials and seed the demo competition:

   ```bash
   scripts/docker/local_setup.py bootstrap-demo
   ```

   This writes fresh admin/validator API keys to `deploy/docker/env/local.env` and creates a sample MT10 competition. Edit `deploy/docker/local/competition.json` before running the script if you want to change the benchmark or episode settings.

   Recreate the validator/evaluator containers so they load the new key:

   ```bash
   scripts/docker/stack.sh up validator --force-recreate -d validator
   scripts/docker/stack.sh up validator --profile cpu --force-recreate -d evaluator
   ```

4. Point your `miner.toml` at the local backend:

   ```toml
   backend_url = "http://localhost:8080"
   ```

5. Run your usual `upload` command. The backend will stash artifacts in MinIO and the validator/evaluator containers will process jobs just like the hosted stack.

6. When finished, tear everything down:

   ```bash
   scripts/docker/stack.sh down local -v
   ```

## Committing submission info to the blockchain

After a successful upload, commit the returned submission to the Bittensor blockchain:

```bash
python -m miner commit --config miner.toml --submission-id <SUBMISSION_ID>
```

Only the submission id is required on-chain; the validator uses its own stored metadata (hash and size) during evaluation.
