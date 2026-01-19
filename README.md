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

The subnet uses a **separated backend/validator architecture** with miners deployed on **Chutes**:

```mermaid
flowchart TB
    subgraph Chain["Bittensor Chain"]
        Commitments[("Miner Commitments<br/>(HuggingFace repo + Chutes endpoint)")]
        Weights[("Validator Weights")]
    end

    subgraph Backend["Evaluation Backend (GPU)"]
        Scheduler["Background Scheduler"]
        API["REST API"]
        DB[("PostgreSQL<br/>Scores & Weights")]
        EvalEnv["Eval Environment<br/>(MuJoCo + MetaWorld)"]
    end

    subgraph Validators["Validator(s) (Lightweight)"]
        V1["Validator 1"]
        V2["Validator 2"]
        Vn["Validator N"]
    end

    subgraph Chutes["Chutes (Decentralized GPU Cloud)"]
        M1["Miner 1 Policy Server<br/>/reset, /act"]
        M2["Miner 2 Policy Server<br/>/reset, /act"]
        Mn["Miner N Policy Server<br/>/reset, /act"]
    end

    %% Miner registration flow
    M1 & M2 & Mn -->|"1. Commit endpoint"| Commitments

    %% Backend reads commitments
    Commitments -->|"2. Read commitments"| Scheduler

    %% Evaluation flow
    Scheduler -->|"3. Start eval cycle"| EvalEnv
    EvalEnv -->|"4. Get actions<br/>(obs → action)"| M1 & M2 & Mn
    EvalEnv -->|"5. Store scores"| DB
    DB -->|"6. Compute Pareto weights"| API

    %% Validator flow
    API -->|"7. GET /v1/weights/latest"| V1 & V2 & Vn
    V1 & V2 & Vn -->|"8. Submit weights"| Weights
```

### Evaluation Flow

1. **Miners** deploy policy servers to **Chutes** and commit their endpoint on-chain
2. **Backend** reads miner commitments from chain to discover Chutes endpoints
3. **Backend scheduler** starts evaluation cycle (triggered periodically)
4. **Eval environment** runs MuJoCo simulation, calls each miner's `/act` endpoint
5. **Scores** are stored in PostgreSQL after each evaluation
6. **Pareto weights** are computed from scores and exposed via REST API
7. **Validators** poll `GET /v1/weights/latest` to fetch computed weights
8. **Validators** submit weights to Bittensor chain

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/AffineFoundation/kinitro.git
cd kinitro

# Install with uv (recommended)
uv sync

# Or with pip
pip install -e .
```

### For Validators

See the full [Validator Guide](docs/validator-guide.md) for detailed instructions.

```bash
# 1. Start PostgreSQL
docker run -d --name kinitro-postgres \
  -e POSTGRES_USER=kinitro -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=kinitro \
  -p 5432:5432 postgres:15

# 2. Build the evaluation environment
uv run kinitro build-eval-env --tag kinitro/eval-env:v1

# 3. Initialize database
uv run kinitro db init --database-url postgresql://kinitro:secret@localhost/kinitro

# 4. Start the backend (runs evaluations)
uv run kinitro backend \
  --netuid YOUR_NETUID \
  --network finney \
  --database-url postgresql://kinitro:secret@localhost/kinitro

# 5. Start the validator (submits weights to chain)
uv run kinitro validate \
  --backend-url http://localhost:8000 \
  --netuid YOUR_NETUID \
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

## CLI Reference

```bash
# Database commands
kinitro db create         # Create database
kinitro db init           # Initialize schema
kinitro db status         # Show database statistics
kinitro db reset          # Drop and recreate database
kinitro db drop           # Drop database

# Backend commands
kinitro backend           # Run evaluation backend service

# Validator commands
kinitro validate          # Run validator (polls backend, sets weights)

# Environment commands
kinitro list-envs         # List available environments
kinitro test-env ENV_ID   # Test an environment locally
kinitro test-scoring      # Test the scoring mechanism

# Miner commands
kinitro init-miner DIR    # Initialize miner template
kinitro build PATH --tag TAG [--push]  # Build Docker image
kinitro commit            # Commit model to chain
```

## Backend API

The backend exposes a REST API:

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

## Project Structure

```
kinitro/
├── kinitro/
│   ├── backend/          # Evaluation backend service
│   │   ├── app.py        # FastAPI application
│   │   ├── routes.py     # REST API endpoints
│   │   ├── scheduler.py  # Background evaluation loop
│   │   ├── storage.py    # PostgreSQL storage layer
│   │   └── models.py     # Database & API models
│   ├── environments/     # Robotics environment wrappers
│   │   └── metaworld_env.py
│   ├── evaluation/       # Episode rollout and parallel evaluation
│   ├── scoring/          # ε-Pareto dominance and weights
│   ├── chain/            # Bittensor chain integration
│   ├── validator/        # Lightweight validator client
│   │   ├── main.py       # Polls backend, sets weights
│   │   └── client.py     # HTTP client for backend API
│   └── miner/            # Miner templates
├── tests/
├── docker-compose.yml
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
