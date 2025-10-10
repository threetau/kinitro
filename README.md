# Kinitro: Incentivized Emobodied Intelligence

Kinitro incentivizes the emergence of agents that can conquer various tasks across different environments. Miners publish agents to compete, validators peform rollouts and evaluate the agents, and reward miners based on the results. All this happens in real-time and can easily be viewed by anyone through our [dashboard](https://kinitro.ai/dashboard).



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

## Documentation & Support
- [Overview](https://kinitro.ai/docs/overview)
- [Architecture](https://kinitro.ai/docs/architecture/introduction)
- [Miner Guide](https://kinitro.ai/docs/miner)
- [Validator Guide](https://kinitro.ai/docs/validator)

## Project Layout
- **Backend service** (`src/backend/`): FastAPI backend with a realtime broadcaster, chain monitor, job scheduler, and scoring engine backed by PostgreSQL.
- **Validator node** (`src/validator/`): WebSocket client that authenticates with the backend, relays evaluation jobs into a persistent `pgqueuer` queue, and streams results and episode logs back.
- **Evaluator cluster** (`src/evaluator/`): Ray-powered orchestrator that spins Kubernetes submission pods, runs rollout workers, logs per-step data, and pushes metrics/results into the validator queue.
- **Miner tooling** (`src/miner/`): CLI helpers that package models, upload artifacts to Hugging Face, and notarize submissions on the Bittensor chain.
- **Shared core** (`src/core/`): Message formats, chain helpers, database models, and logging utilities that keep every component speaking the same language.
- **Docs** (`docs/`): User guides and deep dives (architecture, miner, validator, overview).
- **Scripts** (`scripts/`): Database migration/reset helpers and operational scripts.



Questions or ideas? Open an issue or reach out to us on [our channel in the Bittensor Discord server](https://discord.gg/96SdmpeMqG). Contributions via pull requests are welcome.

## License
Released under the MIT License. See [LICENSE](LICENSE) for details.
