# Miner Guide

This guide explains how to participate as a miner in Kinitro. As a miner, you'll train a robotics policy and deploy it as an HTTP endpoint that validators can query during evaluation.

## Overview

The evaluation flow:

1. You train a robotics policy (locally or on your own infrastructure)
2. You build a Docker image containing your policy server
3. You push your image to a container registry (e.g., Docker Hub)
4. You deploy your image to [Basilica](https://basilica.ai) (required for mainnet)
5. You commit your deployment ID on-chain so validators can find you
6. Validators periodically evaluate your policy across multiple environments
7. You earn rewards based on how well your policy generalizes

## Requirements

- **Training**: GPU compute for training your policy
- **Container Registry**: For storing your Docker image (e.g., Docker Hub)
- **Basilica Account**: For deploying your policy server (required for mainnet)
- **Bittensor Wallet**: For registering as a miner and committing your endpoint

> **Important**: For mainnet (Finney), you **must** deploy via Basilica. Self-hosted endpoints are only supported for local testing and development.

## Quick Start Summary

```bash
# 1. Initialize policy from template
uv run kinitro miner init ./my-policy
cd my-policy

# 2. Implement your policy in policy.py

# 3. Test locally
uvicorn server:app --port 8001

# 4. Build and push Docker image
docker build -t your-username/kinitro-policy:v1 .
docker push your-username/kinitro-policy:v1

# 5. Deploy to Basilica and commit on-chain
export BASILICA_API_TOKEN="your-basilica-api-token"

uv run kinitro miner deploy \
  --image your-username/kinitro-policy:v1 \
  --netuid YOUR_NETUID \
  --network finney
```

Or do each step separately:

```bash
# Deploy to Basilica
uv run kinitro miner push \
  --image your-username/kinitro-policy:v1 \
  --name my-policy

# Commit on-chain
uv run kinitro miner commit \
  --deployment-id YOUR_BASILICA_DEPLOYMENT_ID \
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

- `server.py` - FastAPI server with `/reset` and `/act` endpoints (for local testing and Basilica deployment)
- `policy.py` - Policy implementation template (edit this!)
- `Dockerfile` - For containerizing your policy
- `requirements.txt` - Python dependencies

## Step 2: Understand the Observation Space

Your policy receives **canonical observations** that vary by environment family. A generalist policy must handle both.

**Important**: Object positions are NOT provided in either family. Miners must learn to infer object locations from camera images.

### MetaWorld (Robot Arm Manipulation)

> **Note:** MetaWorld is for local testing and development only; not used in mainnet evaluations.

**Proprioceptive** (`proprio` key):

- `ee_pos`: End-effector XYZ position (meters)
- `ee_quat`: End-effector quaternion (XYZW)
- `ee_vel_lin`: End-effector linear velocity (m/s)
- `ee_vel_ang`: End-effector angular velocity (rad/s)
- `gripper`: Gripper state as a list, e.g. [1.0], in [0, 1]

**Camera images** (`rgb` key):

- `corner`: 84x84x3 RGB image from corner camera
- `corner2`: 84x84x3 RGB image from corner camera 2

**Action space** (`continuous` key):

- `ee_twist`: 6D twist (vx, vy, vz, wx, wy, wz) in [-1, 1]
- `gripper`: Gripper action as a list, e.g. [1.0], in [0, 1]

### Genesis (Humanoid Locomotion + Manipulation)

**Proprioceptive** (`proprio` key):

- `base_pos`: Base XYZ position (3 values)
- `base_quat`: Base quaternion WXYZ (4 values)
- `base_vel`: Base linear + angular velocity (6 values)
- `joint_pos`: Joint positions in radians (43 values)
- `joint_vel`: Joint velocities in rad/s (43 values)

**Camera images** (`rgb` key):

- `ego`: 84x84x3 RGB image from torso-mounted ego camera

**Extra** (`extra` key):

- `task_prompt`: Natural language task description (e.g., "Walk to the red box.")
- `task_type`: Task type string (navigate, pickup, place, push)

**Action space** (`continuous` key):

- `joint_pos_target`: 43-dimensional joint position targets in [-1, 1], scaled per-joint

## Step 3: Implement Your Policy

Edit `policy.py` to implement your policy. Here's a minimal example:

```python
from kinitro.rl_interface import Action, ActionKeys, Observation

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

    async def act(self, observation: Observation) -> Action:
        """Return action for current observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation.proprio_array()).unsqueeze(0)
            output = self.model(obs_tensor)
            twist = output.squeeze(0)[:6].cpu().numpy().tolist()
        return Action(continuous={
            ActionKeys.EE_TWIST: twist,
            ActionKeys.GRIPPER: [0.0],
        })

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

Test your policy using the FastAPI server before deploying to Basilica:

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
  -d '{"obs": {"proprio": {"ee_pos": [0.0, 0.5, 0.2], "ee_quat": [0.0, 0.0, 0.0, 1.0], "ee_vel_lin": [0.0, 0.0, 0.0], "ee_vel_ang": [0.0, 0.0, 0.0], "gripper": [1.0]}, "rgb": {}}}'
```

The `server.py` file provides the endpoints (`/health`, `/reset`, `/act`) that validators will call. This lets you test your policy logic locally before deploying to Basilica.

## Step 5: Build and Push Docker Image

Build your policy into a Docker image and push to a container registry:

```bash
# Build the Docker image
docker build -t your-username/kinitro-policy:v1 .

# Test locally with Docker
docker run -p 8001:8000 your-username/kinitro-policy:v1

# Push to Docker Hub (or any public registry)
docker push your-username/kinitro-policy:v1
```

## Step 6: Deploy to Basilica (Required for Mainnet)

[Basilica](https://basilica.ai) provides serverless container inference. **Deployment via Basilica is required for mainnet participation.**

### Prerequisites

1. **Create a Basilica account** at [basilica.ai](https://basilica.ai)
2. **Get your Basilica API token** by [using the cli](https://docs.basilica.ai/cli#account-management)

### Deploy Using kinitro CLI (Recommended)

The easiest way to deploy is using the `kinitro miner push` command:

```bash
# Set credentials
export BASILICA_API_TOKEN="your-api-token"

# Deploy to Basilica
uv run kinitro miner push \
  --image your-username/kinitro-policy:v1 \
  --name my-policy \
  --gpu-count 1 \
  --min-vram 16
```

This command deploys your pre-built Docker image to Basilica and returns the deployment ID for on-chain commitment.

### Verify Deployment

After deployment, note your **deployment ID**. You can verify your deployment:

```bash
# Test the endpoint (replace with your URL)
curl https://YOUR-DEPLOYMENT-ID.deployments.basilica.ai/health

# Verify metadata
uv run kinitro miner verify --deployment-id YOUR-DEPLOYMENT-ID
```

### GPU vs CPU Deployments

For testing, you can deploy without GPU:

```bash
uv run kinitro miner push \
  --image your-username/kinitro-policy:v1 \
  --name my-policy \
  --gpu-count 0  # CPU-only for testing
```

For production with GPU:

```bash
uv run kinitro miner push \
  --image your-username/kinitro-policy:v1 \
  --name my-policy \
  --gpu-count 1 \
  --min-vram 16
```

### Deployment Verification

> **Important**: Your Basilica deployment is verified via the Basilica metadata API.

The evaluation system checks deployments to ensure they are running and using a publicly pullable Docker image. During verification:

1. Deployment state is checked (must be "Running")
2. Docker image is verified to be publicly pullable
3. Public metadata enrollment is checked

**To pass verification:**

- Your Docker image must be publicly pullable from a container registry
- Your Basilica deployment must be running and healthy
- Enroll for public metadata (the CLI does this automatically)

## Local Testing (Development Only)

For local development and testing, you can run your policy server directly:

```bash
# Build Docker image
docker build -t my-policy:v1 .

# Run locally
docker run -p 8001:8000 my-policy:v1
```

You can then test with a local validator backend by committing your local endpoint:

```bash
# For LOCAL TESTING ONLY - not valid for mainnet
uv run kinitro miner commit \
  --deployment-id my-local-deploy \
  --netuid 2 \
  --network local \
  --wallet-name test-wallet \
  --hotkey-name hotkey0
```

> **Warning**: Self-hosted endpoints are NOT supported on mainnet. You must deploy via Basilica for your commitment to be valid.

## Step 7: Commit On-Chain

Register your deployment on-chain so validators can find and evaluate you.

The commitment stores your **Basilica deployment ID** on-chain.

### Basic Commitment (Endpoint Visible On-Chain)

```bash
uv run kinitro miner commit \
  --deployment-id YOUR_BASILICA_DEPLOYMENT_ID \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### Encrypted Commitment (Recommended)

To protect your Basilica endpoint from public disclosure, use encrypted commitments. This ensures only the backend operator can discover your endpoint:

```bash
uv run kinitro miner commit \
  --deployment-id YOUR_BASILICA_DEPLOYMENT_ID \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey \
  --encrypt \
  --backend-hotkey BACKEND_OPERATOR_HOTKEY
```

The `--backend-hotkey` is the SS58 address of the backend operator (subnet owner). This automatically fetches their public key from the chain and encrypts your deployment endpoint.

#### Why use encrypted commitments?

- Your Basilica endpoint URL is not publicly visible on-chain
- Only the backend operator (who runs evaluations) can decrypt your endpoint
- Prevents competitors from directly accessing or probing your deployment
- The backend operator publishes their public key on-chain for miners to use

#### Alternative: Provide the public key directly

If you have the backend operator's public key (64-character hex string), you can provide it directly:

```bash
uv run kinitro miner commit \
  ... \
  --encrypt \
  --backend-public-key <64-char-hex-public-key>
```

### Commitment Format

The commitment is stored on-chain in a compact format:

**Plain commitment:**
```
deployment-uuid
```

**Encrypted commitment:**
```
e:<base85-encrypted-blob>
```

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

1. Build and push a new Docker image
2. Deploy the updated image to Basilica
3. Commit the new deployment ID on-chain

Validators will automatically pick up your new endpoint at the next evaluation cycle.

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
    "proprio": {
      "ee_pos": [0.0, 0.5, 0.2],
      "ee_quat": [0.0, 0.0, 0.0, 1.0],
      "ee_vel_lin": [0.0, 0.0, 0.0],
      "ee_vel_ang": [0.0, 0.0, 0.0],
      "gripper": [1.0]
    },
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
    "continuous": {
      "ee_twist": [0.1, -0.2, 0.05, 0.0, 0.0, 0.0],
      "gripper": [1.0]
    }
  }
}
```

**Timing**: You have approximately 500ms to respond. Slower responses may be penalized.

## Scoring and Rewards

Your policy is evaluated using **epsilon-Pareto dominance**:

1. **Multi-Environment Evaluation**: Your policy runs on all environments
2. **Success Rate**: For each environment, we measure task success rate (binary for Genesis)
3. **Pareto Frontier**: Miners that dominate on subsets of environments earn points
4. **Generalists Win**: Larger subsets give more points - you must perform well everywhere

Key implications:

- Specializing on one environment won't earn rewards
- Copying another miner's policy gives no benefit (ties under Pareto dominance)
- True generalization across all tasks is rewarded

## Troubleshooting

### Policy not being evaluated

1. Check your on-chain commitment: `uv run kinitro miner show-commitment --netuid ... --wallet-name ...`
2. Verify your Basilica deployment is running - check the Basilica dashboard
3. Verify your endpoint is accessible: `curl YOUR_BASILICA_ENDPOINT/health`
4. Check validator logs for errors

### Basilica deployment issues

- **Deployment not running**: Check your Basilica dashboard for status.
- **Deployment failures**: Check Basilica logs in the dashboard.

### Slow responses / timeouts

- Optimize your model for inference speed
- Use GPU acceleration (configure appropriate GPU in Basilica config)
- Consider action chunking to reduce per-step latency
- Ensure your Basilica deployment has adequate GPU resources

### Low success rates

- Verify you're correctly interpreting the observation format
- Check action normalization (should be in [-1, 1])
- Train on more diverse tasks
- Use vision-based policies to infer object positions

### Commitment not recognized

- Ensure your deployment ID is correct
- Verify the Basilica deployment is running
- Commitment must be under ~128 bytes

### Testing Endpoints

Use the same curl commands from [Step 4](#step-4-test-locally) for local testing. For remote testing after deployment, replace `http://localhost:8001` with your Basilica endpoint URL (`https://YOUR-DEPLOYMENT-ID.deployments.basilica.ai`).

## Additional Resources

- [Validator Guide](./validator-guide.md) - For validators
- [Backend Guide](./backend-guide.md) - For subnet operators
- [MetaWorld Documentation](https://github.com/Farama-Foundation/Metaworld)
- [Genesis Documentation](https://genesis-world.readthedocs.io/)
- [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) - Robot models (Unitree G1)
- [Bittensor Docs](https://docs.bittensor.com/)
- [Basilica Documentation](https://docs.basilica.ai/)
- Template code in `/kinitro/miner/template/`
