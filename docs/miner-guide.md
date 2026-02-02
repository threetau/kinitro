# Miner Guide

This guide explains how to participate as a miner in Kinitro. As a miner, you'll train a robotics policy and upload it to HuggingFace for evaluation.

## Overview

The evaluation flow:

1. You train a robotics policy (locally or on your own infrastructure)
2. You upload your model weights to HuggingFace
3. You commit your repo and revision on-chain so validators can find you
4. The executor automatically downloads your model and creates deployments for evaluation
5. Validators periodically evaluate your policy across multiple environments
6. You earn rewards based on how well your policy generalizes

## Requirements

- **Training**: GPU compute for training your policy
- **HuggingFace Account**: For storing your model weights
- **Bittensor Wallet**: For registering as a miner and committing your model info

## Quick Start Summary

```bash
# 1. Initialize policy from template
uv run kinitro miner init ./my-policy
cd my-policy

# 2. Implement your policy in policy.py

# 3. Test locally
uvicorn server:app --port 8001

# 4. One-command deployment (upload + commit)
export HF_TOKEN="your-huggingface-token"

uv run kinitro miner deploy \
  --repo your-username/kinitro-policy \
  --path ./my-policy \
  --netuid YOUR_NETUID \
  --network finney
```

Or do each step separately:

```bash
# Upload to HuggingFace
huggingface-cli upload your-username/kinitro-policy .

# Commit on-chain
uv run kinitro miner commit \
  --repo your-username/kinitro-policy \
  --revision YOUR_HF_COMMIT_SHA \
  --netuid YOUR_NETUID
```

## Step 1: Initialize Your Policy

Use the CLI to create a new policy from the template:

```bash
# Create a new directory with the miner template
uv run kinitro miner init ./my-policy
cd my-policy
```

This creates:

- `server.py` - FastAPI server with `/reset` and `/act` endpoints (for local testing)
- `policy.py` - Policy implementation template (edit this!)
- `requirements.txt` - Python dependencies

## Step 2: Understand the Observation Space

Your policy receives the **canonical observation** to encourage true generalization:

### Proprioceptive Observations

A dictionary with:

- `ee_pos_m`: End-effector XYZ position (meters)
- `ee_quat_xyzw`: End-effector quaternion (XYZW)
- `ee_lin_vel_mps`: End-effector linear velocity (m/s)
- `ee_ang_vel_rps`: End-effector angular velocity (rad/s)
- `gripper_01`: Gripper state in [0, 1]

### Camera Images (Optional)

A dictionary of RGB images (nested lists):

- `corner`: 84x84x3 RGB image from corner camera
- `corner2`: 84x84x3 RGB image from corner camera 2

**Important**: Object positions are NOT provided! You must learn to infer object locations from camera images.

### Action Space

Return a dictionary with:

- `twist_ee_norm`: 6D twist (vx, vy, vz, wx, wy, wz) in [-1, 1]
- `gripper_01`: Gripper action in [0, 1]

`twist_ee_norm` values should be in range `[-1, 1]`; `gripper_01` should be in `[0, 1]`.

## Step 3: Implement Your Policy

Edit `policy.py` to implement your policy. Here's a minimal example:

```python
from kinitro.rl_interface import CanonicalAction, CanonicalObservation

class RobotPolicy:
    def __init__(self):
        # Load your trained model
        import torch
        self.model = torch.load("model.pt", map_location="cpu")
        self.model.eval()
        self._model_loaded = True

    def is_loaded(self) -> bool:
        return self._model_loaded

    async def reset(self, task_config: dict) -> str:
        """Called at start of each episode."""
        # Reset any episode-specific state
        # task_config contains: env_id, task_name, seed
        return uuid.uuid4().hex

    async def act(self, observation: CanonicalObservation) -> CanonicalAction:
        """Return action for current observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation.proprio_array()).unsqueeze(0)
            action = self.model(obs_tensor)
            twist = action.squeeze(0)[:6].cpu().numpy().tolist()
        return CanonicalAction(twist_ee_norm=twist, gripper_01=0.0)

    async def cleanup(self):
        """Called on shutdown."""
        self.model = None
```

### Policy Architecture Tips

1. **Vision Encoder**: Use a pre-trained vision encoder (ViT, ResNet) to process camera images
2. **Multi-Task Learning**: Your policy must generalize across environments - train on diverse tasks
3. **Action Chunking**: Consider predicting action sequences for smoother control
4. **Diffusion Policies**: These handle multi-modal action distributions well

See `policy.py` for example implementations of VLA and Diffusion policies.

## Step 4: Test Locally

Test your policy using the FastAPI server before uploading:

```bash
cd my-policy

# Start the server
uvicorn server:app --host 0.0.0.0 --port 8001

# In another terminal, test the endpoints:
# Health check
curl http://localhost:8001/health

# Reset for new episode
curl -X POST http://localhost:8001/reset \
  -H "Content-Type: application/json" \
  -d '{"task_config": {"env_id": "metaworld/pick-place-v3", "seed": 42}}'

# Get action
curl -X POST http://localhost:8001/act \
  -H "Content-Type: application/json" \
  -d '{"obs": {"ee_pos_m": [0.0, 0.5, 0.2], "ee_quat_xyzw": [0.0, 0.0, 0.0, 1.0], "ee_lin_vel_mps": [0.0, 0.0, 0.0], "ee_ang_vel_rps": [0.0, 0.0, 0.0], "gripper_01": 1.0, "rgb": {}}}'
```

The `server.py` file provides the endpoints (`/health`, `/reset`, `/act`) that the executor will call during evaluation.

## Step 5: Upload to HuggingFace

Upload your model weights to HuggingFace:

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Login to HuggingFace
huggingface-cli login

# Create a new model repository
huggingface-cli repo create your-username/kinitro-policy --type model

# Upload your model files
huggingface-cli upload your-username/kinitro-policy ./my-policy

# Note the commit SHA for the on-chain commitment
git ls-remote https://huggingface.co/your-username/kinitro-policy HEAD
```

**Size limits:** HuggingFace repositories larger than 5GB will be rejected.

## Step 6: Commit On-Chain

Register your policy on-chain so the executor can find and evaluate you.

The commitment includes two pieces of information:

- **repo**: Your HuggingFace repository (e.g., `your-username/kinitro-policy`)
- **revision**: The HuggingFace commit SHA of your model

```bash
uv run kinitro miner commit \
  --repo your-username/kinitro-policy \
  --revision YOUR_HUGGINGFACE_COMMIT_SHA \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### Commitment Format

The commitment is stored on-chain as a compact colon-separated string:

```
your-username/kinitro-policy:abc123de
```

Where:
- First part = HuggingFace model repository
- Second part = HuggingFace revision (truncated to 8 chars)

### Verify Your Commitment

```bash
uv run kinitro miner show-commitment \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### Updating Your Model

When you update your model:

1. Upload the new weights to HuggingFace
2. Commit the new revision on-chain

The executor will automatically pick up your new model at the next evaluation cycle.

## One-Command Deployment

For convenience, use `kinitro miner deploy` to upload and commit in one step:

```bash
export HF_TOKEN="your-huggingface-token"

uv run kinitro miner deploy \
  --repo your-username/kinitro-policy \
  --path ./my-policy \
  --netuid YOUR_NETUID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

This command:
1. Uploads your policy folder to HuggingFace
2. Commits the repo and revision on-chain

Use `--dry-run` to see what would happen without making changes.

## API Specification

Your policy server must implement these endpoints:

### `GET /health`

Health check endpoint.

**Response:**

```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 123.45
}
```

### `POST /reset`

Reset policy for a new episode.

**Request:**

```json
{
  "task_config": {
    "env_id": "metaworld/pick-place-v3",
    "env_name": "pick-place",
    "task_name": "pick-place-v3",
    "seed": 42,
    "task_id": 0
  }
}
```

**Response:**

```json
{
  "status": "ok",
  "episode_id": "abc123"
}
```

### `POST /act`

Get action for current observation.

**Request:**

```json
{
  "obs": {
    "ee_pos_m": [0.0, 0.5, 0.2],
    "ee_quat_xyzw": [0.0, 0.0, 0.0, 1.0],
    "ee_lin_vel_mps": [0.0, 0.0, 0.0],
    "ee_ang_vel_rps": [0.0, 0.0, 0.0],
    "gripper_01": 1.0,
    "rgb": {
      "corner": [[[0, 0, 0], ...], ...],
      "corner2": [[[0, 0, 0], ...], ...]
    }
  }
}
```

The `obs` field contains canonical observations. The `rgb` field is optional and contains camera images as nested lists (84x84x3).

**Response:**

```json
{
  "action": {
    "twist_ee_norm": [0.1, -0.2, 0.05, 0.0, 0.0, 0.0],
    "gripper_01": 1.0
  }
}
```

**Timing**: You have approximately 500ms to respond. Slower responses may be penalized.

## Scoring and Rewards

Your policy is evaluated using **epsilon-Pareto dominance**:

1. **Multi-Environment Evaluation**: Your policy runs on all MetaWorld environments
2. **Success Rate**: For each environment, we measure task success rate
3. **Pareto Frontier**: Miners that dominate on subsets of environments earn points
4. **Generalists Win**: Larger subsets give more points - you must perform well everywhere

Key implications:

- Specializing on one environment won't earn rewards
- Copying another miner's policy gives no benefit (ties under Pareto dominance)
- True generalization across all tasks is rewarded

## Troubleshooting

### Policy not being evaluated

1. Check your on-chain commitment: `uv run kinitro miner show-commitment --netuid ... --wallet-name ...`
2. Verify your HuggingFace repo is accessible: `huggingface-cli repo info your-username/kinitro-policy`
3. Ensure the revision in your commitment exists in your HuggingFace repo
4. Check validator logs for errors

### Low success rates

- Verify you're correctly interpreting the observation format
- Check action normalization (should be in [-1, 1])
- Train on more diverse tasks
- Use vision-based policies to infer object positions

### Commitment not recognized

- Verify the HuggingFace repo exists and is accessible
- Check that the revision SHA matches your HuggingFace commit
- Commitment must be under ~128 bytes

### Testing Endpoints Locally

Test your policy locally using the FastAPI server:

```bash
cd my-policy

# Start the local server
uvicorn server:app --host 0.0.0.0 --port 8001

# Test endpoints locally
curl http://localhost:8001/health
curl -X POST http://localhost:8001/reset \
  -H "Content-Type: application/json" \
  -d '{"task_config": {"task_name": "pick-place-v3", "seed": 42}}'
curl -X POST http://localhost:8001/act \
  -H "Content-Type: application/json" \
  -d '{"obs": {"ee_pos_m": [0.0, 0.0, 0.0], "ee_quat_xyzw": [0.0, 0.0, 0.0, 1.0], "ee_lin_vel_mps": [0.0, 0.0, 0.0], "ee_ang_vel_rps": [0.0, 0.0, 0.0], "gripper_01": 1.0, "rgb": {}}}'
```

## Example Training Setup

Here's a suggested training approach:

```python
import metaworld
import torch

# Create multi-task environment
mt = metaworld.MT10()  # 10 tasks

# Sample tasks
for name, env_cls in mt.train_classes.items():
    env = env_cls()
    task = random.choice([t for t in mt.train_tasks if t.env_name == name])
    env.set_task(task)

    # Collect data with your policy
    obs = env.reset()
    for _ in range(500):
        action = policy(obs)  # Your policy
        obs, reward, done, info = env.step(action)
        # Store transition for training

# Train policy on collected data
# Use behavior cloning, RL, or imitation learning
```

## Additional Resources

- [Validator Guide](./validator-guide.md) - For validators
- [Backend Guide](./backend-guide.md) - For subnet operators
- [MetaWorld Documentation](https://github.com/Farama-Foundation/Metaworld)
- [Bittensor Docs](https://docs.bittensor.com/)
- Template code in `/kinitro/miner/template/`
