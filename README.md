# Kinitro: Robotics Generalization Subnet

A Bittensor subnet for evaluating generalist robotics policies across diverse simulated environments.

## Overview

This subnet incentivizes the development of **generalist robotics policies** - AI systems that can control robots across multiple different tasks and environments. Unlike narrow RL policies trained for single tasks, miners must submit policies that perform well across a diverse set of simulated robotics environments.

Only policies that **generalize across ALL environments** earn rewards, using ε-Pareto dominance scoring.

https://github.com/user-attachments/assets/37942435-8143-41cf-aa78-39f4e8a04509

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
- **Deployment verification**: Spot-checks verify Basilica deployments match HuggingFace uploads

### Scoring: ε-Pareto Dominance

Miners are scored using winners-take-all over environment subsets:

1. For each subset of environments, find the miner that dominates
2. Award points scaled by subset size (larger subsets = more points)
3. Convert to weights via softmax

This rewards true generalists over specialists.

## Architecture

The subnet uses a **split service architecture** with miners deployed on **Basilica**:

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

    subgraph Basilica["Basilica (Miner Policy Servers)"]
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
| **Executor** | `kinitro executor` | Run MuJoCo evaluations via [Affinetes](https://github.com/AffineFoundation/affinetes/) | Horizontal (GPU machines) |
| **Validator** | `kinitro validate` | Submit weights to chain | Per validator |

### Evaluation Flow

1. **Miners** deploy policy servers to **Basilica** and commit their endpoint on-chain
2. **Scheduler** reads miner commitments from chain to discover Basilica endpoints
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

### For Miners

See the full [Miner Guide](docs/miner-guide.md) for detailed instructions on how to train and deploy a policy.

```bash
# 1. Initialize a policy template
uv run kinitro miner init ./my-policy
cd my-policy

# 2. Implement your policy in policy.py

# 3. Test locally
uvicorn server:app --port 8001

# 4. Upload to HuggingFace
huggingface-cli upload your-username/kinitro-policy .

# 5. Deploy to Basilica
export BASILICA_API_TOKEN="your-api-token"

uv run kinitro miner push \
  --repo your-username/kinitro-policy \
  --revision YOUR_HF_COMMIT_SHA \
  --gpu-count 1 --min-vram 16

# 6. Register on chain
uv run kinitro miner commit \
  --repo your-username/kinitro-policy \
  --revision YOUR_HF_COMMIT_SHA \
  --endpoint YOUR_BASILICA_URL \
  --netuid YOUR_NETUID \
  --network finney
```

### For Validators

See the full [Validator Guide](docs/validator-guide.md) for detailed instructions on setting up a validator (lightweight).

```bash
# Start the validator (submits weights to chain)
uv run kinitro validate \
  --backend-url https://api.kinitro.ai \
  --netuid 26 \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### For Backend Operators (subnet owners)

See the full [Backend Guide](docs/backend-guide.md) for instructions on how to run the evaluation backend (subnet operators only).

```bash
# 1. Start PostgreSQL
docker run -d --name kinitro-postgres \
  -e POSTGRES_USER=kinitro -e POSTGRES_PASSWORD=secret -e POSTGRES_DB=kinitro \
  -p 5432:5432 postgres:15

# 2. Build the evaluation environment images
uv run kinitro env build metaworld --tag kinitro/metaworld:v1
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

## Environments

### MetaWorld (Manipulation)

MuJoCo-based robot arm manipulation tasks:

- `metaworld/reach-v3` - Move end-effector to target position
- `metaworld/push-v3` - Push object to goal location
- `metaworld/pick-place-v3` - Pick up object and place at target
- `metaworld/door-open-v3` - Open a door
- `metaworld/drawer-open-v3` - Open a drawer
- `metaworld/drawer-close-v3` - Close a drawer
- `metaworld/button-press-v3` - Press a button from top-down
- `metaworld/peg-insert-v3` - Insert peg into hole

Use `kinitro env list` to see all available environments.

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

Use environment variables or a `.env` file. See `.env.example` for configuration options.

## License

MIT

## References

- [Bittensor](https://bittensor.com/) - Decentralized AI network
- [MetaWorld](https://github.com/Farama-Foundation/Metaworld) - Manipulation benchmark
