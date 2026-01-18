# Robotics Generalization Subnet

A Bittensor subnet for evaluating generalist robotics policies across diverse simulated environments.

## Overview

This subnet incentivizes the development of **generalist robotics policies** - AI systems that can control robots across multiple different tasks and environments. Unlike narrow RL policies trained for single tasks, miners must submit policies that perform well across:

- **Manipulation** (MetaWorld): Pick-and-place, pushing, drawer opening, etc.
- **Locomotion** (DM Control): Walking, running, balancing
- **Dexterous manipulation** (ManiSkill): Complex object manipulation

Only policies that **generalize across ALL environments** earn rewards, using ε-Pareto dominance scoring.

## Key Features

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

## Quick Start

### Installation

```bash
# Clone and install (core - MetaWorld only)
git clone https://github.com/your-org/robo-subnet.git
cd robo-subnet
pip install -e .
```

### For Validators

```bash
# Run validator
robo validate --netuid YOUR_NETUID --network finney
```

Or with Docker:

```bash
cp env.example .env
# Edit .env with your settings
docker-compose up -d validator
```

### For Miners

1. Initialize a policy template:
```bash
robo init-miner ./my-policy
cd my-policy
```

2. Implement your policy in `env.py`

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
# Validator commands
robo validate          # Run validator
robo list-envs         # List available environments
robo test-env ENV_ID   # Test an environment locally
robo test-scoring      # Test the scoring mechanism

# Miner commands
robo init-miner DIR    # Initialize miner template
robo build PATH --tag TAG [--push]  # Build Docker image
robo commit            # Commit model to chain
```

## Architecture

```
robo-subnet/
├── robo/
│   ├── environments/     # Robotics environment wrappers
│   │   ├── metaworld_env.py
│   │   ├── dm_control_env.py
│   │   └── maniskill_env.py
│   ├── evaluation/       # Episode rollout and parallel evaluation
│   ├── scoring/          # ε-Pareto dominance and weights
│   ├── chain/            # Bittensor chain integration
│   ├── validator/        # Main validator loop
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
- `metaworld/peg-insert-v3`
- `metaworld/reach-v3`
- `metaworld/door-open-v3`
- `metaworld/drawer-close-v3`
- `metaworld/button-press-v3`

### DM Control (Locomotion) - Optional: `pip install -e ".[dm-control]"`
- `dm_control/walker-walk`
- `dm_control/walker-run`
- `dm_control/cheetah-run`
- `dm_control/humanoid-walk`
- ... and more

### ManiSkill (Dexterous) - Optional: `pip install -e ".[maniskill]"`
- `maniskill/PickCube-v1`
- `maniskill/StackCube-v1`
- `maniskill/PegInsertionSide-v1`
- ... and more

Use `robo list-envs` to see which environments are available in your installation.

## Miner Policy Interface

Your policy must implement the `RobotActor` class:

```python
class RobotActor:
    async def reset(self, task_config: dict) -> None:
        """Called at start of each episode with task info."""
        pass
    
    async def act(self, observation: list[float]) -> list[float]:
        """Return action for observation. Must respond within 50ms."""
        return action
```

See `robo/miner/template/env.py` for a complete example.

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest tests/

# Type checking
mypy robo/

# Linting
ruff check robo/
```

## License

MIT

## References

- [Affine (SN120)](https://github.com/AffineFoundation/affine-cortex) - Original architecture
- [MetaWorld](https://github.com/Farama-Foundation/Metaworld) - Manipulation benchmark
- [DM Control](https://github.com/google-deepmind/dm_control) - Locomotion benchmark
- [ManiSkill2](https://github.com/haosulab/ManiSkill2) - Dexterous manipulation
