# Validator Guide

This guide explains how to run a validator for Kinitro. Validators are **lightweight** - they simply poll the evaluation backend for weights and submit them to the Bittensor chain.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  EVALUATION BACKEND (operated by subnet owner)                  │
│    - Runs evaluations on miner policies                         │
│    - Computes epsilon-Pareto scores                             │
│    - Exposes REST API: GET /v1/weights/latest                   │
└──────────────────────────┬──────────────────────────────────────┘
                           │ HTTP
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  YOUR VALIDATOR                                                 │
│    - Polls backend for weights                                  │
│    - Submits weights to chain                                   │
│    - No GPU required                                            │
└─────────────────────────────────────────────────────────────────┘
```

The backend handles all the heavy computation (running simulations, evaluating miners). Your validator just needs to:

1. Fetch the latest weights from the backend API
2. Submit those weights to the Bittensor chain

## Requirements

- **CPU**: Minimal (1 core sufficient)
- **RAM**: 2GB minimum
- **No GPU required**
- **Bittensor wallet** registered as a validator on the subnet

## Quick Start

### 1. Install the Package

```bash
git clone https://github.com/threetau/kinitro.git
cd kinitro
uv sync
```

### 2. Start the Validator

```bash
uv run kinitro validate \
  --backend-url BACKEND_URL \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --wallet-name your-validator-wallet \
  --hotkey-name your-hotkey
```

Replace `BACKEND_URL` with the official backend endpoint (provided by subnet owner).

That's it! The validator will:

- Poll the backend for latest weights
- Submit weights to the Bittensor chain
- Handle errors and retries automatically

## Configuration

| Flag | Default | Description |
|------|---------|-------------|
| `--backend-url` | (required) | URL of the evaluation backend |
| `--network` | `finney` | Bittensor network (`finney`, `test`, or WebSocket URL) |
| `--netuid` | (required) | Subnet UID |
| `--wallet-name` | `default` | Wallet name |
| `--hotkey-name` | `default` | Hotkey name |
| `--log-level` | `INFO` | Logging level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |

## Running as a Service

For production, run the validator as a systemd service:

```bash
# /etc/systemd/system/kinitro-validator.service
[Unit]
Description=Kinitro Subnet Validator
After=network.target

[Service]
Type=simple
User=validator
WorkingDirectory=/home/validator/kinitro
ExecStart=/home/validator/.local/bin/uv run kinitro validate \
  --backend-url BACKEND_URL \
  --netuid YOUR_NETUID \
  --network finney \
  --wallet-name validator \
  --hotkey-name default
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable kinitro-validator
sudo systemctl start kinitro-validator
sudo journalctl -u kinitro-validator -f  # View logs
```

## Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
RUN pip install uv

COPY . .
RUN uv sync

CMD ["uv", "run", "kinitro", "validate", \
     "--backend-url", "${BACKEND_URL}", \
     "--netuid", "${NETUID}", \
     "--network", "${NETWORK}", \
     "--wallet-name", "${WALLET_NAME}", \
     "--hotkey-name", "${HOTKEY_NAME}"]
```

Run with:

```bash
docker run -d \
  -e BACKEND_URL=https://backend.kinitro.ai \
  -e NETUID=123 \
  -e NETWORK=finney \
  -e WALLET_NAME=validator \
  -e HOTKEY_NAME=default \
  -v ~/.bittensor:/root/.bittensor:ro \
  kinitro-validator
```

## Troubleshooting

### "Cannot connect to backend"

- Verify the backend URL is correct
- Check your network/firewall allows outbound HTTPS
- Try: `curl BACKEND_URL/health`

### "Weight submission failed"

- Check your wallet has sufficient stake
- Verify your validator is registered on the subnet
- Check chain connectivity: `btcli subnet list`

### "Validator not receiving emissions"

- Ensure you're submitting weights regularly
- Check your validator registration: `btcli wallet overview --netuid YOUR_NETUID`
- Verify the backend is operational

## Monitoring

Check validator status:

```bash
# View logs
uv run kinitro validate --log-level DEBUG ...

# Check if backend is healthy
curl BACKEND_URL/health

# View your validator on chain
btcli wallet overview --netuid YOUR_NETUID --wallet-name your-wallet
```

## Additional Resources

- [Miner Guide](./miner-guide.md) - For miners participating in the subnet
- [Backend Guide](./backend-guide.md) - For subnet operators running the evaluation backend
- [Bittensor Docs](https://docs.bittensor.com/) - Bittensor documentation
