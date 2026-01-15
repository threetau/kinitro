---
section: 'Start Validating'
---

# Validator

Validators are responsible for setting weights on the Bittensor chain based on miner performance. The validator periodically polls the backend's `/weights` endpoint and commits any changes on-chain.

## Architecture Overview

The validator is intentionally lightweight:

```
Backend ─────GET /weights────► Validator ─────set_weights────► Bittensor Chain
```

- **No WebSocket connection** - The validator uses simple HTTP polling
- **No database required** - Weights are fetched fresh each cycle
- **No evaluator needed** - Evaluators connect directly to the backend (see [Evaluator docs](../architecture/evaluator.md))

## Setup - Bare Metal

### Setting up environment variables

Copy the `.env.validator.example` file to `.env` and fill in the required environment variables:

```bash
cp .env.validator.example .env
```

The only required environment variables are for your Bittensor wallet configuration.

### Configuration

To configure a validator, start by copying the example configuration file:

```bash
cp config/validator.toml.example validator.toml
```

Key configuration options:

```toml
# Backend endpoint to poll for weights
weights_url = "https://api.kinitro.ai/weights"

# How often to poll for weight updates (seconds)
weights_poll_interval = 300.0  # 5 minutes

# HTTP request timeout (seconds)
weights_request_timeout = 10.0

# How old weights can be before considered stale (seconds)
weights_stale_threshold = 900.0  # 15 minutes

# Bittensor configuration
netuid = 123
network = "finney"  # or "test" for testnet
wallet_name = "default"
hotkey_name = "default"
```

Use the default `weights_url` unless you operate your own backend.

### Running the validator

Launch the validator with:

```bash
python -m validator --config validator.toml
```

The validator will:
1. Poll `GET /weights` at the configured interval
2. Compare fetched weights against the last committed values
3. Call `set_weights` on the Bittensor chain when changes occur
4. Log all weight-setting activity

### Wallet Setup

Ensure your Bittensor wallet is properly configured:

```bash
# Check wallet exists
btcli wallet list

# Register on subnet if needed
btcli subnet register --netuid <netuid> --wallet.name <wallet> --wallet.hotkey <hotkey>
```

The validator needs sufficient stake to set weights on the subnet.

## Setup - Containerized Deployment

We ship Docker recipes for the validator in `deploy/docker/`. The validator container is lightweight and only requires network access to the backend and Bittensor chain.

### 1. Prerequisites

- **Docker Compose v2** (bundled with modern Docker releases)
- **Bittensor wallets** - Point `BITTENSOR_HOME` at your wallet directory (defaults to `$HOME/.bittensor`)

```bash
export BITTENSOR_HOME="$HOME/.bittensor"
```

### 2. Prepare configuration files

Copy your configuration into the Compose folder:

```bash
mkdir -p deploy/docker/validator-config
cp validator.toml deploy/docker/validator-config/
```

### 3. Run the validator

```bash
docker compose -f deploy/docker/compose.yaml up -d validator
```

The validator container mounts your wallet directory read-only and uses the configuration from `validator-config/`.

## Running an Evaluator (Optional)

If you want to contribute to the evaluation network, you can run an evaluator separately. Evaluators connect directly to the backend via WebSocket and do not require the validator.

See the [Evaluator documentation](../architecture/evaluator.md) for setup instructions.

### Evaluator Prerequisites

Running an evaluator requires:
- **Kubernetes cluster** (Minikube for local development, or a managed K8s service)
- **Ray cluster** for distributed rollouts
- **API key** from the Kinitro team (contact us on [Discord](https://discord.gg/96SdmpeMqG))

### Evaluator Configuration

```bash
cp config/evaluator.toml.example evaluator.toml
```

Key settings in `evaluator.toml`:

- `backend_url` - WebSocket URL for the backend
- `api_key` - Your evaluator API key
- `max_concurrent_jobs` - Number of parallel evaluations
- Ray and worker resource settings

### Running the Evaluator

```bash
python -m evaluator --config evaluator.toml
```

## Image Matrix

| Image | Purpose |
| --- | --- |
| `ghcr.io/threetau/kinitro-validator` | Weight-setting validator service |
| `ghcr.io/threetau/kinitro-evaluator` | Evaluation orchestrator (CPU / `-gpu`) |
| `ghcr.io/threetau/kinitro-miner-agent` | Submission runtime for evaluation pods (CPU / `-gpu`) |

## Troubleshooting

### Validator not setting weights

1. Check wallet registration: `btcli subnet list --netuid <netuid>`
2. Verify sufficient stake for weight setting
3. Check network connectivity to `weights_url`
4. Review logs for HTTP errors

### Stale weights warning

If weights are older than `weights_stale_threshold`, the validator logs a warning. This typically means:
- The backend is not receiving evaluation results
- No approved leaders exist for competitions
- Network issues between validator and backend

### Connection errors

```bash
# Test connectivity to backend
curl -s https://api.kinitro.ai/weights | jq .
```

If the endpoint returns a valid JSON response with weights, the issue is likely with your local configuration.
