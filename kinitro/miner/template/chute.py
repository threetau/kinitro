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

import json
import logging
import os
import time
from contextlib import contextmanager
from typing import Any

from chutes.chute import Chute, NodeSelector
from chutes.image import Image
from pydantic import BaseModel

# =============================================================================
# Structured Logging
# =============================================================================


class StructuredLogger:
    """JSON structured logger for production observability."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers
        self.logger.handlers = []

        # Add JSON handler
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(handler)

        self.name = name

    def _log(self, level: str, message: str, **kwargs: Any) -> None:
        """Emit a structured log entry."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "level": level,
            "logger": self.name,
            "message": message,
            **kwargs,
        }
        log_method = getattr(self.logger, level.lower())
        log_method(json.dumps(entry))

    def info(self, message: str, **kwargs: Any) -> None:
        self._log("INFO", message, **kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        self._log("WARNING", message, **kwargs)

    def error(self, message: str, **kwargs: Any) -> None:
        self._log("ERROR", message, **kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        self._log("DEBUG", message, **kwargs)

    @contextmanager
    def measure(self, operation: str, **extra: Any):
        """Context manager to measure and log operation duration."""
        start = time.time()
        try:
            yield
        except Exception as e:
            duration = time.time() - start
            self.error(
                f"{operation} failed",
                operation=operation,
                duration_ms=round(duration * 1000, 2),
                error=str(e),
                error_type=type(e).__name__,
                **extra,
            )
            raise
        else:
            duration = time.time() - start
            self.info(
                f"{operation} completed",
                operation=operation,
                duration_ms=round(duration * 1000, 2),
                **extra,
            )


# Initialize logger
logger = StructuredLogger("kinitro.policy")

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
    # Keep chute warm for 8 hours to ensure validators can reach endpoint
    shutdown_after_seconds=28800,
    # Allow multiple concurrent requests (e.g., during batch evaluation)
    concurrency=4,
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
_request_count = 0
_error_count = 0


@chute.on_startup()
def initialize(app):
    """Initialize the policy on startup."""
    import sys

    global _policy, _start_time
    _start_time = time.time()

    logger.info("Starting policy initialization", hf_repo=HF_REPO, hf_revision=HF_REVISION)

    # Add /app to path for imports
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    # Try to load policy
    try:
        with logger.measure("policy_load"):
            from policy import RobotPolicy

            _policy = RobotPolicy()
        logger.info(
            "Policy initialized successfully",
            model_loaded=_policy.is_loaded() if _policy else False,
        )
    except Exception as e:
        logger.error(
            "Failed to initialize policy",
            error=str(e),
            error_type=type(e).__name__,
        )
        _policy = None


@chute.on_shutdown()
def cleanup(app):
    """Cleanup resources on shutdown."""
    global _policy

    logger.info(
        "Shutting down policy server",
        total_requests=_request_count,
        total_errors=_error_count,
        uptime_seconds=round(time.time() - _start_time, 2) if _start_time else 0,
    )

    if _policy is not None:
        try:
            # Release GPU memory
            import torch

            if hasattr(_policy, "cleanup"):
                _policy.cleanup()
            _policy = None

            # Force CUDA memory cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                logger.info("GPU memory released")
        except Exception as e:
            logger.error(
                "Cleanup failed",
                error=str(e),
                error_type=type(e).__name__,
            )


# =============================================================================
# Endpoints
# =============================================================================


@chute.cord(public_api_path="/health", public_api_method="GET")
async def health(app=None) -> HealthResponse:
    """Health check endpoint."""
    uptime = time.time() - _start_time if _start_time else 0
    is_healthy = _policy is not None
    model_loaded = _policy is not None and _policy.is_loaded()

    logger.info(
        "Health check",
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=model_loaded,
        uptime_seconds=round(uptime, 2),
        request_count=_request_count,
        error_count=_error_count,
    )

    return HealthResponse(
        status="healthy" if is_healthy else "unhealthy",
        model_loaded=model_loaded,
        uptime_seconds=uptime,
    )


@chute.cord(public_api_path="/reset", public_api_method="POST")
async def reset(app, request: ResetRequest) -> ResetResponse:
    """Reset policy for a new episode."""
    global _task_config, _request_count, _error_count

    _request_count += 1
    task_config = request.task_config.model_dump()

    if _policy is None:
        _error_count += 1
        logger.error(
            "Reset failed - policy not loaded",
            env_id=task_config.get("env_id"),
            task_id=task_config.get("task_id"),
        )
        return ResetResponse(status="error", episode_id=None)

    try:
        with logger.measure(
            "reset",
            env_id=task_config.get("env_id"),
            task_id=task_config.get("task_id"),
        ):
            _task_config = task_config
            episode_id = await _policy.reset(_task_config)

        return ResetResponse(status="ok", episode_id=episode_id)

    except Exception as e:
        _error_count += 1
        logger.error(
            "Reset failed",
            error=str(e),
            error_type=type(e).__name__,
            env_id=task_config.get("env_id"),
            task_id=task_config.get("task_id"),
        )
        return ResetResponse(status="error", episode_id=None)


@chute.cord(public_api_path="/act", public_api_method="POST")
async def act(app, request: ActRequest) -> ActResponse:
    """Get action for current observation."""
    import numpy as np

    global _request_count, _error_count

    _request_count += 1

    if _policy is None:
        _error_count += 1
        logger.warning("Act called but policy not loaded")
        return ActResponse(action=[0.0, 0.0, 0.0, 0.0])

    try:
        # Convert observation
        obs = np.array(request.observation, dtype=np.float32)

        # Convert images if provided
        images = None
        if request.images:
            images = {k: np.array(v, dtype=np.uint8) for k, v in request.images.items()}

        # Get action (don't log every act call to avoid log spam, but measure timing)
        start = time.time()
        action = await _policy.act(obs, images)
        duration_ms = (time.time() - start) * 1000

        # Convert to list
        if hasattr(action, "tolist"):
            action = action.tolist()

        # Log periodically (every 100 requests) or if slow
        if _request_count % 100 == 0 or duration_ms > 100:
            logger.info(
                "Act request",
                duration_ms=round(duration_ms, 2),
                request_count=_request_count,
                obs_dim=len(request.observation),
                has_images=request.images is not None,
            )

        return ActResponse(action=action)

    except Exception as e:
        _error_count += 1
        logger.error(
            "Act failed",
            error=str(e),
            error_type=type(e).__name__,
            obs_dim=len(request.observation) if request.observation else 0,
        )
        return ActResponse(action=[0.0, 0.0, 0.0, 0.0])
