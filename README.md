# Kinitro: Incentivized Embodied Intelligence

Kinitro incentivizes the emergence of agents that can conquer various tasks across different environments. Miners publish agents to compete, evaluators perform rollouts and evaluate the agents, validators set weights on the Bittensor chain, and miners are rewarded based on the results. All this happens in real-time and can easily be viewed by anyone through our [dashboard](https://kinitro.ai/dashboard).

For a visual overview of how these pieces interact, see the [architecture introduction](docs/architecture/introduction.md).

## Architecture Overview

```
                                    ┌─────────────────────┐
                                    │   Bittensor Chain   │
                                    │   (Commitments +    │
                                    │    Weights)         │
                                    └──────────┬──────────┘
                                               │
                    ┌──────────────────────────┼──────────────────────────┐
                    │                          │                          │
                    ▼                          ▼                          ▼
           ┌────────────────┐         ┌────────────────┐         ┌────────────────┐
           │     Miner      │         │    Backend     │         │   Validator    │
           │  (Submission)  │         │   (FastAPI)    │         │(Polls /weights)│
           └───────┬────────┘         └───────┬────────┘         └───────┬────────┘
                   │                          │                          │
                   │   upload artifact        │   GET /weights           │
                   └─────────────────────────►│◄─────────────────────────┘
                                              │
                                              │   WebSocket (direct)
                                              ▼
                                    ┌─────────────────┐
                                    │   Evaluator     │
                                    │   (Ray + K8s)   │
                                    └─────────────────┘
```

- **Backend**: Central orchestration service that manages competitions, submissions, and job scheduling
- **Evaluators**: Connect directly to the backend via WebSocket, receive jobs, and stream results back
- **Validators**: Poll the backend's `/weights` endpoint and set weights on the Bittensor chain

## Repository Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/threetau/kinitro
   cd kinitro
   ```

2. **Create a virtual environment and install dependencies**

   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv sync --dev
   uv pip install -e .
   ```

## Documentation & Support

- [Overview](https://kinitro.ai/docs/overview)
- [Architecture](https://kinitro.ai/docs/architecture/introduction)
- [Miner Guide](https://kinitro.ai/docs/miner)
- [Validator Guide](https://kinitro.ai/docs/validator)

## Project Layout

- **Backend service** (`src/backend/`): FastAPI backend with a realtime broadcaster, chain monitor, job scheduler, evaluator hub, and scoring engine backed by PostgreSQL.
- **Evaluator cluster** (`src/evaluator/`): Ray-powered orchestrator that connects directly to the backend via WebSocket, spins up submission pods in Kubernetes, runs rollout workers, and streams metrics/results back.
- **Validator node** (`src/validator/`): Lightweight service that polls the backend's `/weights` endpoint and sets weights on the Bittensor chain.
- **Miner tooling** (`src/miner/`): CLI helpers that package models, upload artifacts to Hugging Face, and notarize submissions on the Bittensor chain.
- **Shared core** (`src/core/`): Message formats, chain helpers, database models, and logging utilities that keep every component speaking the same language.
- **Docs** (`docs/`): User guides and deep dives (architecture, miner, validator, overview).
- **Scripts** (`scripts/`): Database migration/reset helpers and operational scripts.
- **Deploy artifacts** (`deploy/`): Dockerfiles and Docker Compose stack (with Minikube integration) for running validators/evaluators with CPU or GPU workloads.

## Running Components

### Backend
```bash
python -m backend
```

### Evaluator
```bash
python -m evaluator
```

### Validator
```bash
python -m validator
```

## Database Migrations

```bash
# Backend database
./scripts/migrate_backend_db.sh

# Evaluator database
./scripts/migrate_evaluator_db.sh
```

## Testing

```bash
python -m pytest src/backend/tests/ src/backend/scoring/tests/ \
    src/evaluator/executors/tests/ src/evaluator/providers/tests/ -v
```

Questions or ideas? Open an issue or reach out to us on [our channel in the Bittensor Discord server](https://discord.gg/96SdmpeMqG). Contributions via pull requests are welcome.

## License

Released under the MIT License. See [LICENSE](LICENSE) for details.
