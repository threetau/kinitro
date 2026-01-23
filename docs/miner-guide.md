# Miner Guide

This guide explains how to participate as a miner in Kinitro. As a miner, you'll train a robotics policy and deploy it as an HTTP endpoint that validators can query during evaluation.

## Overview

The evaluation flow:
1. You train a robotics policy (locally or on your own infrastructure)
2. You upload your model weights to HuggingFace
3. You deploy your policy server to [Chutes](https://chutes.ai) (required for mainnet)
4. You commit your endpoint info on-chain so validators can find you
5. Validators periodically evaluate your policy across multiple environments
6. You earn rewards based on how well your policy generalizes

## Requirements

- **Training**: GPU compute for training your policy
- **HuggingFace Account**: For storing your model weights
- **Chutes Account**: For deploying your policy server (required for mainnet)
- **Bittensor Wallet**: For registering as a miner and committing your endpoint

> **Important**: For mainnet (Finney), you **must** deploy via Chutes. Self-hosted endpoints are only supported for local testing and development.

## Quick Start Summary

```bash
# 1. Initialize policy from template
uv run kinitro init-miner ./my-policy
cd my-policy

# 2. Implement your policy in policy.py

# 3. Test locally
uvicorn server:app --port 8001

# 4. One-command deployment (upload + deploy + commit)
export HF_TOKEN="your-huggingface-token"
export CHUTES_API_KEY="your-chutes-api-key"
export CHUTE_USER="your-chutes-username"

uv run kinitro miner-deploy \
  --repo your-username/kinitro-policy \
  --path ./my-policy \
  --netuid YOUR_NETUID \
  --network finney
```

Or do each step separately:

```bash
# Upload to HuggingFace
huggingface-cli upload your-username/kinitro-policy .

# Deploy to Chutes
uv run kinitro chutes-push --repo your-username/kinitro-policy --revision YOUR_HF_SHA

# Commit on-chain
uv run kinitro commit \
  --repo your-username/kinitro-policy \
  --revision YOUR_HF_COMMIT_SHA \
  --chute-id YOUR_CHUTE_ID \
  --netuid YOUR_NETUID
```

## Step 1: Initialize Your Policy

Use the CLI to create a new policy from the template:

```bash
# Create a new directory with the miner template
uv run kinitro init-miner ./my-policy
cd my-policy
```

This creates:
- `server.py` - FastAPI server with `/reset` and `/act` endpoints (for local testing)
- `policy.py` - Policy implementation template (edit this!)
- `chute.py` - Chutes deployment configuration with `@chute.cord()` endpoints
- `Dockerfile` - For containerizing your policy (optional, for self-hosted)
- `requirements.txt` - Python dependencies

## Step 2: Understand the Observation Space

Your policy receives **limited observations** to encourage true generalization:

### Proprioceptive Observations
A numpy array with:
- `[0:3]` - End-effector XYZ position
- `[3]` - Gripper state (0=closed, 1=open)

### Camera Images (Optional)
A dictionary of RGB images:
- `corner`: 84x84x3 RGB image from corner camera
- `gripper`: 84x84x3 RGB image from gripper camera

**Important**: Object positions are NOT provided! You must learn to infer object locations from camera images.

### Action Space
Return a numpy array with 4 values (for MetaWorld):
- `[0:3]` - Delta XYZ movement
- `[3]` - Gripper action (-1 to close, +1 to open)

All values should be in range `[-1, 1]`.

## Step 3: Implement Your Policy

Edit `policy.py` to implement your policy. Here's a minimal example:

```python
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
    
    async def act(
        self,
        observation: np.ndarray,
        images: dict[str, np.ndarray] | None = None,
    ) -> np.ndarray:
        """Return action for current observation."""
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
            action = self.model(obs_tensor)
            return action.squeeze(0).numpy()
    
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

Test your policy using the FastAPI server before deploying to Chutes:

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
  -d '{"observation": [0.0, 0.5, 0.2, 1.0]}'
```

The `server.py` file provides the same endpoints as the Chutes deployment (`/health`, `/reset`, `/act`) but runs as a standard FastAPI server. This lets you test your policy logic locally before deploying.

> **Note**: The `server.py` uses FastAPI decorators while `chute.py` uses `@chute.cord()` decorators, but both implement the same API interface.

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

## Step 6: Deploy to Chutes (Required for Mainnet)

[Chutes](https://chutes.ai) provides serverless GPU inference. **Deployment via Chutes is required for mainnet participation.**

### Prerequisites

1. **Create a Chutes account** at [chutes.ai](https://chutes.ai)
2. **Register your Chutes account with your mining hotkey** - this links your Chutes deployment to your miner identity
3. **Fund your Chutes account** with TAO to pay for GPU compute time
4. **Get your Chutes API key** from the Chutes dashboard
5. **Install the Chutes SDK**: `pip install chutes`

### Understanding chute.py

The miner template includes a `chute.py` file that defines your Chutes deployment. This file:

1. **Configures the Docker image** - installs dependencies and downloads your model from HuggingFace
2. **Defines the Chute** - specifies GPU requirements and metadata
3. **Implements `@chute.cord()` endpoints** - wraps your policy as HTTP endpoints (`/health`, `/reset`, `/act`)

The key difference from `server.py` is that Chutes uses `@chute.cord()` decorators instead of FastAPI routes:

```python
from chutes.chute import Chute, NodeSelector
from chutes.image import Image

# Configure image build
image = Image("your-username", "your-policy-name", "v1")
image.run_command("pip install torch numpy ...")
image.run_command('python -c "from huggingface_hub import snapshot_download; ..."')

# Define the chute
chute = Chute(
    username="your-username",
    name="your-policy-name", 
    image=image,
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
)

# Define endpoints with @chute.cord()
@chute.cord(public_api_path="/health", public_api_method="GET")
async def health() -> HealthResponse:
    ...

@chute.cord(public_api_path="/reset", public_api_method="POST")
async def reset(request: ResetRequest) -> ResetResponse:
    ...

@chute.cord(public_api_path="/act", public_api_method="POST")
async def act(request: ActRequest) -> ActResponse:
    ...
```

### Configure Your Deployment

Edit `chute.py` to set your configuration:

```python
# Your Chutes username
CHUTES_USER = os.environ.get("CHUTE_USER", "your-username")

# Name for your chute
CHUTE_NAME = f"{CHUTES_USER}-kinitro-policy"

# HuggingFace repo (set via environment or hardcode)
HF_REPO = os.environ.get("HF_REPO", "your-username/kinitro-policy")
HF_REVISION = os.environ.get("HF_REVISION", "main")

# GPU requirements - adjust based on your model size
GPU_COUNT = 1
MIN_VRAM_GB = 16
```

### Deploy Using kinitro CLI (Recommended)

The easiest way to deploy is using the `kinitro chutes-push` command:

```bash
# Set credentials
export CHUTES_API_KEY="your-api-key"
export CHUTE_USER="your-username"

# Deploy to Chutes
uv run kinitro chutes-push \
  --repo your-username/kinitro-policy \
  --revision YOUR_HUGGINGFACE_COMMIT_SHA \
  --gpu-count 1 \
  --min-vram 16
```

This command:
1. Generates a `chute.py` configuration
2. Builds the Docker image on Chutes infrastructure
3. Deploys the chute
4. Returns the `chute_id` for on-chain commitment

### Deploy Manually with Chutes CLI

Alternatively, deploy directly with the chutes CLI:

```bash
# Set credentials
export CHUTES_API_KEY="your-api-key"

# Deploy from your policy directory
cd my-policy
chutes deploy chute:chute --accept-fee
```

### Verify Deployment

After deployment, note your **chute_id** and **slug**. You can verify your deployment:

```bash
# List your chutes
chutes chutes list

# Test the endpoint (replace with your slug)
curl -X GET "https://YOUR-SLUG.chutes.ai/health" \
  -H "Authorization: Bearer $CHUTES_API_KEY"
```

The chute slug follows the pattern: `{username}-{chute-name}` (e.g., `myuser-myuser-kinitro-policy`).

### Keep Your Chute Warm

Chutes auto-shutdown after inactivity. To ensure validators can reach your endpoint:
- Set `shutdown_after_seconds` in your chute config (e.g., `28800` for 8 hours)
- Monitor your Chutes dashboard for uptime
- Fund your account to maintain availability

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
uv run kinitro commit \
  --repo your-username/kinitro-policy \
  --revision $(git rev-parse HEAD) \
  --chute-id http://localhost:8001 \
  --netuid 2 \
  --network local \
  --wallet-name test-wallet \
  --hotkey-name hotkey0
```

> **Warning**: Self-hosted endpoints are NOT supported on mainnet. You must deploy via Chutes for your commitment to be valid.

## Step 7: Commit On-Chain

Register your policy endpoint on-chain so validators can find and evaluate you.

The commitment includes three pieces of information:
- **model**: Your HuggingFace repository (e.g., `your-username/kinitro-policy`)
- **revision**: The HuggingFace commit SHA of your model
- **chute_id**: Your Chutes deployment ID

```bash
uv run kinitro commit \
  --repo your-username/kinitro-policy \
  --revision YOUR_HUGGINGFACE_COMMIT_SHA \
  --chute-id YOUR_CHUTE_DEPLOYMENT_ID \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### Commitment Format

The commitment is stored on-chain as compact JSON to fit within chain limits:

```json
{"m":"your-username/kinitro-policy","r":"abc123def456...","c":"chute_xyz789..."}
```

Where:
- `m` = HuggingFace model repository
- `r` = HuggingFace revision (commit SHA)
- `c` = Chutes deployment ID

### Verify Your Commitment

```bash
uv run kinitro show-commitment \
  --netuid YOUR_SUBNET_ID \
  --network finney \
  --wallet-name your-wallet \
  --hotkey-name your-hotkey
```

### Updating Your Model

When you update your model:
1. Upload the new weights to HuggingFace
2. Deploy the updated model to Chutes (or update existing deployment)
3. Commit the new revision and chute_id on-chain

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
  "observation": [0.0, 0.5, 0.2, 1.0],
  "images": {
    "corner": [[[0, 0, 0], ...], ...],
    "gripper": [[[0, 0, 0], ...], ...]
  }
}
```

The `observation` is a list of 4 floats (proprioceptive state).
The `images` field is optional and contains camera images as nested lists (84x84x3).

**Response:**
```json
{
  "action": [0.1, -0.2, 0.05, 1.0]
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

1. Check your on-chain commitment: `uv run kinitro show-commitment --netuid ... --wallet-name ...`
2. Verify your Chutes deployment is "hot" (running) - check the Chutes dashboard
3. Verify your endpoint is accessible: `curl YOUR_CHUTE_ENDPOINT/health`
4. Ensure the revision in your commitment matches the deployed model
5. Check validator logs for errors

### Chutes deployment issues

- **"Cold" Chute**: Your deployment may have auto-shutdown. Fund your account and redeploy.
- **Rate limits**: Ensure your Chutes account has sufficient balance for GPU time.
- **Deployment failures**: Check Chutes logs in the dashboard.

### Slow responses / timeouts

- Optimize your model for inference speed
- Use GPU acceleration (configure appropriate GPU in Chute config)
- Consider action chunking to reduce per-step latency
- Ensure your Chutes deployment has adequate GPU resources

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

### Testing Chutes Endpoints

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
  -d '{"observation": [0.0, 0.0, 0.0, 1.0]}'
```

#### Remote Testing (After Deployment)

Test your deployed chute endpoints:

```bash
export CHUTES_API_KEY="your-api-key"
SLUG="your-username-your-chute-name"

# Health check
curl -X GET "https://${SLUG}.chutes.ai/health" \
  -H "Authorization: Bearer $CHUTES_API_KEY"

# Reset
curl -X POST "https://${SLUG}.chutes.ai/reset" \
  -H "Authorization: Bearer $CHUTES_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"task_config": {"task_name": "pick-place-v3", "seed": 42}}'

# Act
curl -X POST "https://${SLUG}.chutes.ai/act" \
  -H "Authorization: Bearer $CHUTES_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"observation": [0.0, 0.0, 0.0, 1.0], "images": null}'
```

#### View Chute Logs

If your chute is having issues, view the logs:

```bash
# List your chutes to get instance ID
chutes chutes list

# View logs (replace INSTANCE_ID)
curl -s "https://api.chutes.ai/instances/INSTANCE_ID/logs/?backfill=1000" \
  -H "Authorization: Bearer $CHUTES_API_KEY" | jq .
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
- [Chutes Documentation](https://docs.chutes.ai/)
- Template code in `/kinitro/miner/template/`
