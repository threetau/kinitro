"""
Chutes Deployment Configuration for Robotics Policy Server

This file defines a Chute that can be deployed to the Chutes platform.
The Chute wraps your policy as HTTP endpoints that validators can query.

Usage:
    # Deploy to Chutes
    chutes deploy chute:chute --accept-fee

    # Or use the kinitro CLI
    kinitro chutes-push --repo YOUR_HF_REPO --revision YOUR_REVISION

For more info: https://docs.chutes.ai
"""

import os

import numpy as np
from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from pydantic import BaseModel

# =============================================================================
# Configuration - Update these for your policy
# =============================================================================

# Your Chutes username
CHUTES_USER = os.environ.get("CHUTE_USER", "your-username")

# Name for your chute (will appear in Chutes dashboard)
CHUTE_NAME = f"{CHUTES_USER}-robo-policy"

# HuggingFace repo containing your policy code (optional - for HF-based deployment)
HF_REPO = os.environ.get("HF_REPO", "")
HF_REVISION = os.environ.get("HF_REVISION", "main")

# GPU requirements
GPU_COUNT = 1
MIN_VRAM_GB = 16  # Adjust based on your model size

# =============================================================================
# Image Configuration
# =============================================================================

# Create image with dynamic tag
image_tag = f"{HF_REVISION[:8]}" if HF_REPO else "latest"
image = Image(
    CHUTES_USER,
    CHUTE_NAME,
    image_tag,
    readme="Robotics policy server for kinitro subnet",
)

# Install dependencies
image.run_command(
    "pip install --no-cache-dir torch numpy gymnasium pillow pydantic huggingface_hub"
)

# If using HuggingFace repo, download the model
if HF_REPO:
    image.run_command(
        f'python -c "from huggingface_hub import snapshot_download; '
        f"snapshot_download('{HF_REPO}', revision='{HF_REVISION}', local_dir='/app')\""
    )

image.set_workdir("/app")

# =============================================================================
# Chute Definition
# =============================================================================

chute = Chute(
    username=CHUTES_USER,
    name=CHUTE_NAME,
    image=image,
    readme="Robotics policy for kinitro subnet evaluation",
    node_selector=NodeSelector(
        gpu_count=GPU_COUNT,
        min_vram_gb_per_gpu=MIN_VRAM_GB,
    ),
)

# =============================================================================
# Request/Response Models
# =============================================================================


class TaskConfig(BaseModel):
    """Task configuration for episode reset."""

    env_id: str | None = None
    env_name: str | None = None
    task_name: str | None = None
    seed: int | None = None
    task_id: int | None = None


class ResetRequest(BaseModel):
    """Request body for /reset endpoint."""

    task_config: TaskConfig


class ResetResponse(BaseModel):
    """Response body for /reset endpoint."""

    status: str = "ok"
    episode_id: str | None = None


class ActRequest(BaseModel):
    """Request body for /act endpoint."""

    observation: list[float]
    images: dict[str, list] | None = None


class ActResponse(BaseModel):
    """Response body for /act endpoint."""

    action: list[float]


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model_loaded: bool
    uptime_seconds: float


# =============================================================================
# Global State (initialized on startup)
# =============================================================================

_policy = None
_current_task_config = None
_start_time = None


@chute.on_startup()
def initialize(app):
    """Initialize the policy on startup."""
    import time

    global _policy, _start_time
    _start_time = time.time()

    # Import your policy implementation
    # If using HuggingFace download, the code is at /app
    import sys

    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    # Import your policy class - update this import for your policy
    try:
        from policy import RobotPolicy

        _policy = RobotPolicy()
        print("Policy initialized successfully")
    except Exception as e:
        print(f"Failed to initialize policy: {e}")
        _policy = None


# =============================================================================
# Endpoints
# =============================================================================


@chute.cord(
    public_api_path="/health",
    public_api_method="GET",
)
async def health() -> HealthResponse:
    """Health check endpoint."""
    import time

    global _policy, _start_time

    return HealthResponse(
        status="healthy" if _policy is not None else "unhealthy",
        model_loaded=_policy is not None and _policy.is_loaded(),
        uptime_seconds=time.time() - _start_time if _start_time else 0,
    )


@chute.cord(
    public_api_path="/reset",
    public_api_method="POST",
)
async def reset(request: ResetRequest) -> ResetResponse:
    """Reset policy for a new episode."""

    global _policy, _current_task_config

    if _policy is None:
        return ResetResponse(status="error", episode_id=None)

    _current_task_config = request.task_config.model_dump()
    episode_id = await _policy.reset(_current_task_config)

    return ResetResponse(status="ok", episode_id=episode_id)


@chute.cord(
    public_api_path="/act",
    public_api_method="POST",
)
async def act(request: ActRequest) -> ActResponse:
    """Get action for current observation."""
    global _policy, _current_task_config

    if _policy is None:
        # Return zeros if policy not loaded
        return ActResponse(action=[0.0, 0.0, 0.0, 0.0])

    # Convert observation to numpy
    obs = np.array(request.observation, dtype=np.float32)

    # Convert images if provided
    images = None
    if request.images:
        images = {}
        for cam_name, img_data in request.images.items():
            img = np.array(img_data, dtype=np.uint8)
            # Convert HWC -> CHW if needed
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.transpose(2, 0, 1)
            images[cam_name] = img

    # Get action from policy
    action = await _policy.act(obs, images)

    # Ensure action is a list
    if hasattr(action, "tolist"):
        action = action.tolist()
    elif hasattr(action, "numpy"):
        action = action.numpy().tolist()

    return ActResponse(action=action)
