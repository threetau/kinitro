# Kinitro - Robotics Generalization Subnet

A Bittensor subnet for evaluating generalist robotics policies across diverse simulated environments.

## Overview

This subnet incentivizes the development of **generalist robotics policies** - AI systems that can control robots across multiple different tasks and environments. Unlike narrow RL policies trained for single tasks, miners must submit policies that perform well across multiple **MetaWorld manipulation tasks**: pick-and-place, pushing, drawer opening, button pressing, and more.

Only policies that **generalize across ALL environments** earn rewards, using ε-Pareto dominance scoring.

## Key Features

### Vision-Based Observations

Miners receive **limited observations** to prevent overfitting:
- **Proprioceptive**: End-effector XYZ position + gripper state (4 values)
- **Visual**: RGB camera images from corner cameras (84x84)
- Object positions are **NOT exposed** - miners must learn from visual input

### Anti-Overfitting by Design

- **Procedural task generation**: Every evaluation uses fresh, procedurally-generated task instances
- **Seed rotation**: Seeds change each block - miners can't pre-compute solutions
- **Domain randomization**: Physics parameters and visual properties are randomized

### Anti-Gaming Mechanisms

- **Sybil-proof**: Copies tie under Pareto dominance, no benefit from multiple identities
- **Copy-proof**: Must improve on the leader to earn, not just match them
- **Specialization-proof**: Must dominate on ALL environments, not just one

### Scoring: ε-Pareto Dominance

Miners are scored using winners-take-all over environment subsets:

1. For each subset of environments, find the miner that dominates
2. Award points scaled by subset size (larger subsets = more points)
3. Convert to weights via softmax

This rewards true generalists over specialists.

## Architecture

The subnet uses a **split service architecture** with miners deployed on **Chutes**:

```mermaid
flowchart TB
    subgraph Chain["Bittensor Chain"]
        Commitments[("Miner Commitments")]
        Weights[("Validator Weights")]
    end

    subgraph API["API Service (kinitro api)"]
        RestAPI["REST API"]
        TaskPool["Task Pool Manager"]
        DB[("PostgreSQL")]
    end

    subgraph Scheduler["Scheduler Service (kinitro scheduler)"]
        TaskGen["Task Generator"]
        Scoring["Pareto Scoring"]
    end

    subgraph Executor["Executor Service(s) (kinitro executor)"]
        E1["Executor 1 (GPU)"]
        E2["Executor 2 (GPU)"]
        En["Executor N (GPU)"]
    end

    subgraph Validators["Validator(s) (kinitro validate)"]
        V1["Validator 1"]
        Vn["Validator N"]
    end

    subgraph Chutes["Chutes (Miner Policy Servers)"]
        M1["Miner 1"]
        Mn["Miner N"]
    end

    %% Chain interactions
    Commitments -->|"Read miners"| Scheduler
    V1 & Vn -->|"Submit weights"| Weights

    %% Scheduler flow
    TaskGen -->|"Create tasks"| DB
    DB -->|"Read scores"| Scoring
    Scoring -->|"Save weights"| DB

    %% Executor flow
    E1 & E2 & En -->|"Fetch tasks"| TaskPool
    E1 & E2 & En -->|"Submit results"| TaskPool
    TaskPool <-->|"Read/Write"| DB

    %% Executor to Miners
    E1 & E2 & En -->|"Get actions"| M1 & Mn

    %% Validator flow
    RestAPI -->|"GET /weights"| V1 & Vn
```

### Service Components

| Service | Command | Purpose | Scaling |
|---------|---------|---------|---------|
| **API** | `kinitro api` | REST API, task pool management | Horizontal (stateless) |
| **Scheduler** | `kinitro scheduler` | Task generation, scoring, weight computation | Single instance |
| **Executor** | `kinitro executor` | Run MuJoCo evaluations via [affinetes](https://github.com/AffineFoundation/affinetes/) | Horizontal (GPU machines) |
| **Validator** | `kinitro validate` | Submit weights to chain | Per validator |

### Evaluation Flow

1. **Miners** deploy policy servers to **Chutes** and commit their endpoint on-chain
2. **Scheduler** reads miner commitments from chain to discover Chutes endpoints
3. **Scheduler** creates evaluation tasks in PostgreSQL (task pool)
4. **Executor(s)** fetch tasks from API (`POST /v1/tasks/fetch`)
5. **Executor** runs MuJoCo simulation, calls miner endpoints for actions
6. **Executor** submits results to API (`POST /v1/tasks/submit`)
7. **Scheduler** computes Pareto scores when cycle complete and saves weights
8. **Validators** poll `GET /v1/weights/latest` and submit to chain

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/threetau/kinitro.git
cd kinitro

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### For Validators

See the full [Validator Guide](docs/validator-guide.md) for detailed instructions.

```bash
# Start the validator (submits weights to chain)
uv run kinitro validate \
  --backend-url https://api.kinitro.ai \
  --netuid 26 \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### For Miners

See the full [Miner Guide](docs/miner-guide.md) for detailed instructions.

```bash
# 1. Initialize a policy template
uv run kinitro init-miner ./my-policy
cd my-policy

# 2. Implement your policy in policy.py

# 3. Test locally
uvicorn server:app --port 8001

# 4. Deploy to Chutes (or self-host)
chutes deploy chute:chute

# 5. Register on chain
uv run kinitro commit \
  --repo your-user/kinitro-policy \
  --revision $(git rev-parse HEAD) \
  --chute-id YOUR_CHUTE_ENDPOINT \
  --netuid YOUR_NETUID \
  --network finney
```


### For Backend Operators (subnet owners)
```bash
# 1. Start PostgreSQL
docker run -d --name kinitro-postgres \
  -e POSTGRES_USER=kinitro -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=kinitro \
  -p 5432:5432 postgres:15

# 2. Build the evaluation environment
uv run kinitro build-eval-env --tag kinitro/eval-env:v1

# 3. Initialize database
uv run kinitro db init --database-url postgresql://kinitro:secret@localhost/kinitro

# 4. Start the services (split architecture)
# Terminal 1: API Service
uv run kinitro api --database-url postgresql://kinitro:secret@localhost/kinitro

# Terminal 2: Scheduler Service
uv run kinitro scheduler \
  --netuid YOUR_NETUID \
  --network finney \
  --database-url postgresql://kinitro:secret@localhost/kinitro

# Terminal 3+: Executor(s) - can run multiple on different GPU machines
uv run kinitro executor --api-url http://localhost:8000
```

## CLI Reference

```bash
## Backend operator commands
# Database commands
kinitro db create         # Create database
kinitro db init           # Initialize schema
kinitro db status         # Show database statistics
kinitro db reset          # Drop and recreate database
kinitro db drop           # Drop database

# Service commands 
kinitro api               # Run API service (REST endpoints, task pool)
kinitro scheduler         # Run scheduler (task generation, scoring)
kinitro executor          # Run executor (MuJoCo evaluations)

## Validator commands
kinitro validate          # Run validator (polls API, sets weights)

## Environment commands
kinitro list-envs         # List available environments
kinitro test-env ENV_ID   # Test an environment locally
kinitro test-scoring      # Test the scoring mechanism

## Miner commands
kinitro init-miner DIR    # Initialize miner template
kinitro build PATH --tag TAG [--push]  # Build Docker image
kinitro commit            # Commit model to chain
kinitro mock-miner        # Run mock miner for testing
```

## API Endpoints

The API service exposes these endpoints:

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /v1/status` | Current backend status |
| `GET /v1/weights/latest` | Latest computed weights |
| `GET /v1/weights/{block}` | Weights for specific block |
| `GET /v1/scores/latest` | Latest evaluation scores |
| `GET /v1/scores/{cycle_id}` | Scores for specific cycle |
| `GET /v1/miners` | List evaluated miners |
| `GET /v1/environments` | List environments |
| `POST /v1/tasks/fetch` | Fetch tasks (for executors) |
| `POST /v1/tasks/submit` | Submit results (for executors) |
| `GET /v1/tasks/stats` | Task pool statistics |

## Project Structure

```
kinitro/
├── kinitro/
│   ├── api/                   # API service (REST endpoints, task pool)
│   │   ├── app.py             # FastAPI application
│   │   ├── config.py          # API configuration
│   │   ├── deps.py            # Dependency injection
│   │   └── routes/            # Route handlers
│   │       ├── health.py      # Health check endpoints
│   │       ├── miners.py      # Miner info endpoints
│   │       ├── scores.py      # Score endpoints
│   │       ├── tasks.py       # Task pool endpoints
│   │       └── weights.py     # Weight endpoints
│   ├── scheduler/             # Scheduler service (task generation, scoring)
│   │   ├── config.py          # Scheduler configuration
│   │   ├── main.py            # Scheduler loop
│   │   ├── scoring.py         # Pareto scoring utilities
│   │   └── task_generator.py  # Task and seed generation
│   ├── executor/              # Executor service (MuJoCo evaluations)
│   │   ├── api_client.py      # HTTP client for API
│   │   ├── config.py          # Executor configuration
│   │   ├── main.py            # Executor loop
│   │   └── worker.py          # Evaluation worker (affinetes)
│   ├── backend/               # Shared storage layer and models
│   │   ├── models.py          # Database & API models
│   │   └── storage.py         # PostgreSQL storage layer
│   ├── environments/          # Robotics environment wrappers
│   │   ├── base.py            # Base environment interface
│   │   ├── metaworld_env.py   # MetaWorld wrapper
│   │   ├── procedural.py      # Procedural generation utilities
│   │   └── registry.py        # Environment registry
│   ├── scoring/               # ε-Pareto dominance and weights
│   │   ├── pareto.py          # Pareto frontier computation
│   │   └── winners_take_all.py # Subset scoring
│   ├── chain/                 # Bittensor chain integration
│   │   ├── commitments.py     # Miner commitment reading
│   │   └── weights.py         # Weight submission
│   ├── validator/             # Lightweight validator client
│   │   ├── client.py          # API client for weights
│   │   └── main.py            # Validator loop
│   ├── miner/                 # Miner templates
│   │   └── template/          # Policy server template
│   │       ├── env.py         # Environment utilities
│   │       ├── policy.py      # Policy interface
│   │       └── server.py      # FastAPI server
│   ├── cli.py                 # CLI commands
│   └── config.py              # Global configuration
├── eval-env/                  # Evaluation environment (runs in Docker)
├── tests/
│   └── unit/                  # Unit tests
└── pyproject.toml
```

## Environments

### MetaWorld (Manipulation)

- `metaworld/pick-place-v3`
- `metaworld/push-v3`
- `metaworld/drawer-open-v3`
- `metaworld/peg-insert-v3`
- `metaworld/reach-v3`
- `metaworld/door-open-v3`
- `metaworld/drawer-close-v3`
- `metaworld/button-press-v3`

Use `kinitro list-envs` to see all available environments.

## Miner Policy Interface

Miners deploy a FastAPI server with these endpoints:

```python
# POST /reset - Reset for new episode
async def reset(task_config: dict) -> str:
    """Called at start of each episode. Returns episode_id."""
    pass

# POST /act - Get action for observation
async def act(observation: np.ndarray, images: dict | None) -> np.ndarray:
    """
    Return action for observation. Must respond within 500ms.
    
    Args:
        observation: Proprioceptive state [ee_x, ee_y, ee_z, gripper_state]
        images: Optional camera images {"corner": (84,84,3), "gripper": (84,84,3)}
    
    Returns:
        Action as numpy array in [-1, 1] range
    """
    return action
```

See the [Miner Guide](docs/miner-guide.md) and `kinitro/miner/template/` for complete examples.

## Documentation

- [Miner Guide](docs/miner-guide.md) - How to train and deploy a policy
- [Validator Guide](docs/validator-guide.md) - How to run a validator (lightweight)
- [Backend Guide](docs/backend-guide.md) - How to run the evaluation backend (subnet operators only)

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with MuJoCo
MUJOCO_GL=egl pytest tests/

# Type checking
mypy kinitro/

# Linting
ruff check kinitro/
```

## Configuration

Environment variables (or `.env` file):

| Variable | Description | Default |
|----------|-------------|---------|
| `NETWORK` | Bittensor network | `finney` |
| `NETUID` | Subnet UID | (required) |
| `WALLET_NAME` | Wallet name | `default` |
| `HOTKEY_NAME` | Hotkey name | `default` |
| `POSTGRES_USER` | Database user | `postgres` |
| `POSTGRES_PASSWORD` | Database password | `postgres` |
| `POSTGRES_DB` | Database name | `kinitro` |
| `EVAL_INTERVAL` | Seconds between evals | `3600` |
| `EPISODES_PER_ENV` | Episodes per environment | `50` |

## License

MIT

## References

- [Bittensor](https://bittensor.com/) - Decentralized AI network
- [MetaWorld](https://github.com/Farama-Foundation/Metaworld) - Manipulation benchmark
