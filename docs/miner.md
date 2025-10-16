---
section: 'Start Mining'
---

# Miner
### Setting up environment variables
Copy the `.env.miner.example` file to `.env` and fill in the required environment variables:
```bash
cp .env.miner.example .env
```

### Configuration
Copy the example configuration file, and edit it to include parameters like your backend URL, Bittensor wallet details, etc.
```bash
cp config/miner.toml.example miner.toml
```
Edit `miner.toml` to set your desired parameters.

### Uploading your agent
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
- optional `holdout_seconds` if you want to override the default private window

### Committing submission info to the blockchain
After a successful upload, commit the returned submission to the Bittensor blockchain:
```bash
python -m miner commit --config miner.toml --submission-id <SUBMISSION_ID>
```
Only the submission id is required on-chain; the validator uses its own stored metadata (hash and size) during evaluation.
