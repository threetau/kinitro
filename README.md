# Kinitro: Incentivized Evaluation for Embodied AI

Kinitro coordinates miners, validators, and evaluators to produce trustworthy scores for robotic and embodied intelligence agents. Miners publish submissions, validators coordinate secure evaluations, and the backend keeps the system in sync with the Bittensor network while streaming real-time results for anyone watching competitions unfold.

## Platform Components
- **Backend service** (`src/backend/`): FastAPI API with a realtime broadcaster, chain monitor, job scheduler, and scoring engine backed by PostgreSQL.
- **Validator node** (`src/validator/`): WebSocket client that authenticates with the backend, relays evaluation jobs into a persistent `pgqueuer` queue, and streams results and episode logs back.
- **Evaluator cluster** (`src/evaluator/`): Ray-powered orchestrator that spins Kubernetes submission pods, runs rollout workers, logs per-step data, and pushes metrics/results into the validator queue.
- **Miner tooling** (`src/miner/`): CLI helpers that package models, upload artifacts to Hugging Face, and notarize submissions on the Bittensor chain.
- **Shared core** (`src/core/`): Message formats, chain helpers, database models, and logging utilities that keep every component speaking the same language.

For a visual overview of how these pieces interact, see the [architecture introduction](docs/architecture/introduction.md).

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
3. **Copy configuration templates that you need**
   ```bash
   cp config/backend.toml.example backend.toml        # backend service
   cp config/validator.toml.example validator.toml    # validator websocket app
   cp config/evaluator.toml.example evaluator.toml    # evaluator orchestrator
   cp config/miner.toml.example miner.toml            # miner CLI
   ```
   Update the copied files with your database URLs, wallet information, Hugging Face repos, R2 credentials, and API keys as required. Each process also reads environment variables via `python-dotenv`. Copy the relevant template when you run a component, for example `cp .env.validator.example .env` before starting the validator, or export the variables manually in your shell.

## Running the Services
### Backend API & Realtime Gateway
The backend hosts REST endpoints, WebSocket endpoints for validators and dashboards, and background jobs that mirror chain commitments into evaluation jobs.
```bash
python -m backend
```
This launches `uvicorn` with the FastAPI application defined in `src/backend/endpoints.py`. The backend automatically boots the realtime broadcaster (`src/backend/realtime.py`) so connected clients receive competition, job, and evaluation events instantly. Runtime settings come from `backend.toml`, environment variables, or CLI flags on `BackendConfig`.

Useful CLI administration commands live in `python -m backend.cli` (API key creation, listing, activation, etc.).

### Validator WebSocket Node
Validators maintain an authenticated WebSocket connection to the backend, cache jobs in PostgreSQL via `pgqueuer`, keep heartbeats alive, and forward evaluation results back to the backend.
```bash
export KINITRO_API_KEY=<validator-api-key>
python -m validator --config validator.toml
```
The validator process expects a database with the pgq extension for durable queues. Use `scripts/reset_validator_db.sh` or `scripts/migrate_validator_db.sh` to prepare and update that database.

### Evaluator Orchestrator
The evaluator pulls jobs from the validator queue, launches rollout workers on Ray, provisions submission containers via Kubernetes, and pushes per-episode metrics back through the queue.
```bash
python -m evaluator.orchestrator --config evaluator.toml
```
Ensure your evaluator settings point at the same database as the validator, include R2 credentials for artifact uploads, and declare the wallet that signs weight updates.

### Miner Workflow
Miners package and publish agents, then commit metadata on-chain so the backend can discover new submissions.
```bash
python -m miner upload --config miner.toml
python -m miner commit --config miner.toml
```
This uploads to Hugging Face and records the submission commitment on Bittensor. Inspect the [miner docs](docs/miner.md) for environment variables and advanced options.

## Project Layout
- `src/backend/`: FastAPI app, database models, realtime broadcaster, chain monitor, and scoring tasks.
- `src/validator/`: WebSocket validator service, pgqueuer integration, and validator database migrations.
- `src/evaluator/`: Ray orchestrator, rollout workers, RPC bridge to submission containers, and episode logging.
- `src/miner/`: CLI wrappers for packaging, uploading, and committing miner submissions.
- `src/core/`: Shared utilities, SQLModel definitions, chain helpers, message schemas, and logging.
- `docs/`: User guides and deep dives (architecture, miner, validator, overview).
- `scripts/`: Database migration/reset helpers and operational scripts.

## Documentation & Support
- [Overview](docs/overview.mdx)
- [Architecture](docs/architecture/introduction.md)
- [Miner Guide](docs/miner.md)
- [Validator Guide](docs/validator.md)

Questions or ideas? Open an issue or reach out on the Kinitro or Bittensor Discord servers. Contributions via pull requests are welcome.

## License
Released under the MIT License. See [LICENSE](LICENSE) for details.
