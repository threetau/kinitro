# Miner Guide

This guide explains how to participate as a miner in Kinitro. As a miner, you'll train a robotics policy and deploy it as an HTTP endpoint that validators can query during evaluation.

## Overview

The evaluation flow:

1. You train a robotics policy (locally or on your own infrastructure)
2. You upload your model weights to HuggingFace
3. You deploy your policy server to [Basilica](https://basilica.ai) (required for mainnet)
4. You commit your endpoint info on-chain so validators can find you
5. Validators periodically evaluate your policy across multiple environments
6. You earn rewards based on how well your policy generalizes

## Requirements

- **Training**: GPU compute for training your policy
- **HuggingFace Account**: For storing your model weights
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

# 4. One-command deployment (upload + deploy + commit)
export HF_TOKEN="your-huggingface-token"
export BASILICA_API_TOKEN="your-basilica-api-token"

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

# Deploy to Basilica
uv run kinitro miner push --repo your-username/kinitro-policy --revision YOUR_HF_SHA

# Commit on-chain
uv run kinitro miner commit \
  --repo your-username/kinitro-policy \
  --revision YOUR_HF_COMMIT_SHA \
  --endpoint YOUR_BASILICA_URL \
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
- `basilica_deploy.py` - Basilica deployment script
- `Dockerfile` - For containerizing your policy (optional, for self-hosted)
- `requirements.txt` - Python dependencies

## Step 2: Understand the Observation Space

Your policy receives the **canonical observation** to encourage true generalization:

### Proprioceptive Observations

A dictionary under the `proprio` key with:

- `ee_pos`: End-effector XYZ position (meters)
- `ee_quat`: End-effector quaternion (XYZW)
- `ee_vel_lin`: End-effector linear velocity (m/s)
- `ee_vel_ang`: End-effector angular velocity (rad/s)
- `gripper`: Gripper state in [0, 1]

### Camera Images (Optional)

A dictionary of RGB images (nested lists):

- `corner`: 84x84x3 RGB image from corner camera
- `corner2`: 84x84x3 RGB image from corner camera 2

**Important**: Object positions are NOT provided! You must learn to infer object locations from camera images.

### Action Space

Return an `Action` object with a `continuous` dict containing:

- `ee_twist`: 6D twist (vx, vy, vz, wx, wy, wz) in [-1, 1]
- `gripper`: Gripper action as a list, e.g. [1.0], in [0, 1]

`ee_twist` values should be in range `[-1, 1]`; `gripper` should be in `[0, 1]`.

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

## Step 5: Upload to HuggingFace

Before deploying, upload your model weights to HuggingFace:

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
  --repo your-username/kinitro-policy \
  --revision YOUR_HUGGINGFACE_COMMIT_SHA \
  --gpu-count 1 \
  --min-vram 16
```

This command:

1. Downloads your policy from HuggingFace
2. Builds a container image with your policy
3. Deploys to Basilica
4. Returns the endpoint URL for on-chain commitment

### Verify Deployment

After deployment, note your **endpoint URL**. You can verify your deployment:

```bash
# Test the endpoint (replace with your URL)
curl https://YOUR-DEPLOYMENT-ID.deployments.basilica.ai/health
```

### GPU vs CPU Deployments

For testing, you can deploy without GPU:

```bash
uv run kinitro miner push \
  --repo your-username/kinitro-policy \
  --revision YOUR_HF_SHA \
  --gpu-count 0  # CPU-only for testing
```

For production with GPU:

```bash
uv run kinitro miner push \
  --repo your-username/kinitro-policy \
  --revision YOUR_HF_SHA \
  --gpu-count 1 \
  --min-vram 16
```

### Deployment Verification (Spot-Checks)

> **Important**: Your Basilica deployment may be spot-checked to verify it matches your HuggingFace upload.

The evaluation system performs random verification checks to ensure miners are running the same code they uploaded to HuggingFace. During verification:

1. Your policy is downloaded from HuggingFace
2. Test observations are generated with deterministic seeds
3. Local inference is compared against your Basilica endpoint
4. If outputs don't match, verification fails

**To pass verification:**

- Your Basilica deployment must serve the exact same model as your HuggingFace upload
- If your policy uses randomness, support the optional `seed` parameter in your `/act` endpoint (the template already handles this)
- Don't modify your deployment code after uploading to HuggingFace

**Size limits:** HuggingFace repositories larger than 5GB will be rejected. This limit applies to both uploads and verification downloads.

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
  --repo your-username/kinitro-policy \
  --revision $(git rev-parse HEAD) \
  --endpoint http://localhost:8001 \
  --netuid 2 \
  --network local \
  --wallet-name test-wallet \
  --hotkey-name hotkey0
```

> **Warning**: Self-hosted endpoints are NOT supported on mainnet. You must deploy via Basilica for your commitment to be valid.

## Step 7: Commit On-Chain

Register your policy endpoint on-chain so validators can find and evaluate you.

The commitment includes three pieces of information:

- **model**: Your HuggingFace repository (e.g., `your-username/kinitro-policy`)
- **revision**: The HuggingFace commit SHA of your model
- **endpoint**: Your Basilica deployment URL

### Basic Commitment (Endpoint Visible On-Chain)

```bash
uv run kinitro miner commit \
  --repo your-username/kinitro-policy \
  --revision YOUR_HUGGINGFACE_COMMIT_SHA \
  --deployment-id YOUR_BASILICA_DEPLOYMENT_UUID \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### Encrypted Commitment (Recommended)

To protect your Basilica endpoint from public disclosure, use encrypted commitments. This ensures only the backend operator can discover your endpoint:

```bash
uv run kinitro miner commit \
  --repo your-username/kinitro-policy \
  --revision YOUR_HUGGINGFACE_COMMIT_SHA \
  --deployment-id YOUR_BASILICA_DEPLOYMENT_UUID \
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

The commitment is stored on-chain as compact JSON to fit within chain limits:

**Plain commitment:**
```json
{"m":"your-username/kinitro-policy","r":"abc123def456...","d":"deployment-uuid"}
```

**Encrypted commitment:**
```json
{"m":"your-username/kinitro-policy","r":"abc123def456...","e":"<base85-encrypted-blob>"}
```

Where:

- `m` = HuggingFace model repository
- `r` = HuggingFace revision (commit SHA)
- `d` = Basilica deployment ID (UUID) - for plain commitments
- `e` = Encrypted deployment ID (base85 blob) - for encrypted commitments

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
2. Deploy the updated model to Basilica
3. Commit the new revision and endpoint on-chain

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
2. Verify your Basilica deployment is running - check the Basilica dashboard
3. Verify your endpoint is accessible: `curl YOUR_BASILICA_ENDPOINT/health`
4. Ensure the revision in your commitment matches the deployed model
5. Check validator logs for errors

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

- Ensure you're using JSON format (not legacy colon-separated)
- Verify the HuggingFace repo exists and is accessible
- Check that the revision SHA matches your HuggingFace commit
- Commitment must be under ~128 bytes (uses compact JSON with short keys)

### Testing Endpoints

#### Local Testing (Before Deployment)

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
  -d '{"obs": {"proprio": {"ee_pos": [0.0, 0.0, 0.0], "ee_quat": [0.0, 0.0, 0.0, 1.0], "ee_vel_lin": [0.0, 0.0, 0.0], "ee_vel_ang": [0.0, 0.0, 0.0], "gripper": [1.0]}, "rgb": {}}}'
```

#### Remote Testing (After Deployment)

Test your deployed Basilica endpoint:

```bash
ENDPOINT="https://YOUR-DEPLOYMENT-ID.deployments.basilica.ai"

# Health check
curl "${ENDPOINT}/health"

# Reset
curl -X POST "${ENDPOINT}/reset" \
  -H "Content-Type: application/json" \
  -d '{"task_config": {"task_name": "pick-place-v3", "seed": 42}}'

# Act
curl -X POST "${ENDPOINT}/act" \
  -H "Content-Type: application/json" \
  -d '{"obs": {"proprio": {"ee_pos": [0.0, 0.0, 0.0], "ee_quat": [0.0, 0.0, 0.0, 1.0], "ee_vel_lin": [0.0, 0.0, 0.0], "ee_vel_ang": [0.0, 0.0, 0.0], "gripper": [1.0]}, "rgb": {}}}'
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
- [Basilica Documentation](https://docs.basilica.ai/)
- Template code in `/kinitro/miner/template/`
