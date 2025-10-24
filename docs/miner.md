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

For end-to-end smoke tests against local infrastructure we ship a Compose setup under `deploy/docker/compose.yaml`. It runs Postgres, MinIO (S3-compatible storage), the backend API, validator, and evaluator using the same entrypoints as production.

1. Review `deploy/docker/.env.local` and adjust credentials if needed (set `ENV_FILE` to point at a different file if you prefer to keep overrides outside the repo).

2. Start the CPU evaluator stack (omit `--detach` to follow logs):

   ```bash
   docker compose -f deploy/docker/compose.yaml --profile cpu up --build
   ```

   The backend is now reachable at `http://localhost:8080`, MinIO at `http://localhost:9000`, and the optional console at `http://localhost:9001`.

3. Point your `miner.toml` at the local backend:

   ```toml
   backend_url = "http://localhost:8080"
   ```

4. Run your usual `upload` command. The backend will stash artifacts in MinIO and the validator/evaluator containers will process jobs just like the hosted stack.

5. When finished, tear everything down:

   ```bash
   docker compose -f deploy/docker/compose.yaml down -v
   ```

## Committing submission info to the blockchain

After a successful upload, commit the returned submission to the Bittensor blockchain:

```bash
python -m miner commit --config miner.toml --submission-id <SUBMISSION_ID>
```

Only the submission id is required on-chain; the validator uses its own stored metadata (hash and size) during evaluation.
