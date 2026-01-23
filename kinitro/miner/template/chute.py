"""
Chutes Deployment Configuration for Robotics Policy Server

Usage:
    # Test locally using server.py
    uvicorn server:app --host 0.0.0.0 --port 8001

    # Deploy to Chutes
    CHUTE_USER=myuser HF_REPO=myuser/mypolicy HF_REVISION=abc123 chutes deploy chute:chute --accept-fee

    # Or use kinitro CLI
    kinitro chutes-push --repo myuser/mypolicy --revision abc123
"""

import os

from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from pydantic import BaseModel

# =============================================================================
# Configuration
# =============================================================================

CHUTES_USER = os.environ.get("CHUTE_USER", "your-username")
HF_REPO = os.environ.get("HF_REPO", "")
HF_REVISION = os.environ.get("HF_REVISION", "main")

# Derive chute name from HF repo
_repo_name = HF_REPO.split("/")[-1] if HF_REPO else "robo-policy"
CHUTE_NAME = os.environ.get("CHUTE_NAME", f"{CHUTES_USER}-{_repo_name}")

# Image tag (can override with IMAGE_TAG for reusing built images)
IMAGE_TAG = os.environ.get("IMAGE_TAG", f"{HF_REVISION[:8]}" if HF_REPO else "latest")

# =============================================================================
# Image Build
# =============================================================================

image = Image(
    CHUTES_USER, CHUTE_NAME, IMAGE_TAG, readme=f"Policy server for {HF_REPO or 'kinitro'}"
)
image.run_command("pip install --no-cache-dir torch numpy pydantic huggingface_hub")

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
    readme="Robotics policy for kinitro subnet",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb_per_gpu=16),
)

# =============================================================================
# Request/Response Models
# =============================================================================


class TaskConfig(BaseModel):
    env_id: str | None = None
    env_name: str | None = None
    task_name: str | None = None
    seed: int | None = None
    task_id: int | None = None


class ResetRequest(BaseModel):
    task_config: TaskConfig


class ResetResponse(BaseModel):
    status: str = "ok"
    episode_id: str | None = None


class ActRequest(BaseModel):
    observation: list[float]
    images: dict[str, list] | None = None


class ActResponse(BaseModel):
    action: list[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


# =============================================================================
# Global State
# =============================================================================

_policy = None
_task_config = None
_start_time = None


@chute.on_startup()
def initialize(app):
    """Initialize the policy on startup."""
    import sys
    import time

    global _policy, _start_time
    _start_time = time.time()

    # Add /app to path for imports
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    # Try to load policy
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


@chute.cord(public_api_path="/health", public_api_method="GET")
async def health(app=None) -> HealthResponse:
    """Health check endpoint."""
    import time

    return HealthResponse(
        status="healthy" if _policy is not None else "unhealthy",
        model_loaded=_policy is not None and _policy.is_loaded(),
        uptime_seconds=time.time() - _start_time if _start_time else 0,
    )


@chute.cord(public_api_path="/reset", public_api_method="POST")
async def reset(app, request: ResetRequest) -> ResetResponse:
    """Reset policy for a new episode."""

    global _task_config

    if _policy is None:
        return ResetResponse(status="error", episode_id=None)

    _task_config = request.task_config.model_dump()
    episode_id = await _policy.reset(_task_config)

    return ResetResponse(status="ok", episode_id=episode_id)


@chute.cord(public_api_path="/act", public_api_method="POST")
async def act(app, request: ActRequest) -> ActResponse:
    """Get action for current observation."""
    import numpy as np

    if _policy is None:
        return ActResponse(action=[0.0, 0.0, 0.0, 0.0])

    # Convert observation
    obs = np.array(request.observation, dtype=np.float32)

    # Convert images if provided
    images = None
    if request.images:
        images = {k: np.array(v, dtype=np.uint8) for k, v in request.images.items()}

    # Get action
    action = await _policy.act(obs, images)

    # Convert to list
    if hasattr(action, "tolist"):
        action = action.tolist()

    return ActResponse(action=action)
