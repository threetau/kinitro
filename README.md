# Robotics Generalization Subnet

A Bittensor subnet for evaluating generalist robotics policies across diverse simulated environments.

## Overview

This subnet incentivizes the development of **generalist robotics policies** - AI systems that can control robots across multiple different tasks and environments. Unlike narrow RL policies trained for single tasks, miners must submit policies that perform well across:

- **Manipulation** (MetaWorld): Pick-and-place, pushing, drawer opening, etc.
- **Locomotion** (DM Control): Walking, running, balancing
- **Dexterous manipulation** (ManiSkill): Complex object manipulation

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

The subnet uses a **separated backend/validator architecture**:

```
┌─────────────────────────────────────┐
│         EVALUATION BACKEND          │
│       (Compute-heavy, GPU)          │
├─────────────────────────────────────┤
│  - PostgreSQL database              │
│  - Background evaluation scheduler  │
│  - REST API for weights/scores      │
│  - Miner discovery from chain       │
│  - MuJoCo simulation (GPU)          │
└──────────────┬──────────────────────┘
               │ HTTP API
               ▼
┌─────────────────────────────────────┐
│         VALIDATOR(S)                │
│       (Lightweight)                 │
├─────────────────────────────────────┤
│  - Polls backend for weights        │
│  - Submits weights to chain         │
│  - No GPU required                  │
└─────────────────────────────────────┘
```

This separation allows:
- Heavy evaluation to run on dedicated GPU hardware
- Multiple validators to share evaluation infrastructure
- Easier scaling and debugging

## Quick Start

### Installation

```bash
# Clone and install
git clone https://github.com/your-org/robo-subnet.git
cd robo-subnet
pip install -e .

# Or with uv
uv sync
```

### Running the Backend

The backend runs evaluations and exposes a REST API:

```bash
# Create and initialize database
robo db create --database-url postgresql://user:pass@localhost/robo
robo db init --database-url postgresql://user:pass@localhost/robo

# Start the backend
robo backend \
  --netuid YOUR_NETUID \
  --network finney \
  --database-url postgresql://user:pass@localhost/robo
```

### Running a Validator

Validators poll the backend and submit weights to chain:

```bash
robo validate \
  --backend-url http://localhost:8000 \
  --netuid YOUR_NETUID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### Docker Deployment

```bash
cp env.example .env
# Edit .env with your settings

# Start everything (postgres + backend + validator)
docker-compose up -d
```

### For Miners

1. Initialize a policy template:
```bash
robo init-miner ./my-policy
cd my-policy
```

2. Implement your policy in `env.py` (see Miner Policy Interface below)

3. Build and push:
```bash
robo build . --tag your-user/robo-policy:v1 --push
```

4. Register on chain:
```bash
robo commit \
  --repo your-user/robo-policy \
  --revision YOUR_COMMIT_SHA \
  --chute-id YOUR_CHUTE_ID \
  --netuid YOUR_NETUID
```

## CLI Reference

```bash
# Database commands
robo db create         # Create database
robo db init           # Initialize schema
robo db status         # Show database statistics
robo db reset          # Drop and recreate database
robo db drop           # Drop database

# Backend commands
robo backend           # Run evaluation backend service

# Validator commands
robo validate          # Run validator (polls backend, sets weights)

# Environment commands
robo list-envs         # List available environments
robo test-env ENV_ID   # Test an environment locally
robo test-scoring      # Test the scoring mechanism

# Miner commands
robo init-miner DIR    # Initialize miner template
robo build PATH --tag TAG [--push]  # Build Docker image
robo commit            # Commit model to chain
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
robo-subnet/
├── robo/
│   ├── backend/          # Evaluation backend service
│   │   ├── app.py        # FastAPI application
│   │   ├── routes.py     # REST API endpoints
│   │   ├── scheduler.py  # Background evaluation loop
│   │   ├── storage.py    # PostgreSQL storage layer
│   │   └── models.py     # Database & API models
│   ├── environments/     # Robotics environment wrappers
│   │   ├── metaworld_env.py
│   │   ├── dm_control_env.py
│   │   └── maniskill_env.py
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

### MetaWorld (Manipulation) - Core, always available
- `metaworld/pick-place-v3`
- `metaworld/push-v3`
- `metaworld/drawer-open-v3`
- `metaworld/peg-insert-side-v3`
- `metaworld/reach-v3`
- `metaworld/door-open-v3`
- `metaworld/drawer-close-v3`
- `metaworld/button-press-topdown-v3`

### DM Control (Locomotion) - Optional
```bash
pip install -e ".[dm-control]"
```

### ManiSkill (Dexterous) - Optional
```bash
pip install -e ".[maniskill]"
```

Use `robo list-envs` to see available environments.

## Miner Policy Interface

Your policy must implement the `RobotActor` class:

```python
class RobotActor:
    async def reset(self, task_config: dict) -> None:
        """Called at start of each episode with task info."""
        pass
    
    async def act(self, observation: dict) -> list[float]:
        """
        Return action for observation. Must respond within 100ms.
        
        Args:
            observation: Dict with keys:
                - end_effector_pos: [x, y, z] position
                - gripper_state: float (0=closed, 1=open)
                - camera_images: {camera_name: base64_png_string}
        
        Returns:
            Action as list of floats in [-1, 1] range
        """
        return action
```

See `robo/miner/template/env.py` for a complete example with vision processing.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Run tests with MuJoCo
MUJOCO_GL=egl pytest tests/

# Type checking
mypy robo/

# Linting
ruff check robo/
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
| `POSTGRES_DB` | Database name | `robo` |
| `EVAL_INTERVAL` | Seconds between evals | `3600` |
| `EPISODES_PER_ENV` | Episodes per environment | `50` |

## License

MIT

## References

- [Bittensor](https://bittensor.com/) - Decentralized AI network
- [MetaWorld](https://github.com/Farama-Foundation/Metaworld) - Manipulation benchmark
- [DM Control](https://github.com/google-deepmind/dm_control) - Locomotion benchmark
- [ManiSkill2](https://github.com/haosulab/ManiSkill2) - Dexterous manipulation
