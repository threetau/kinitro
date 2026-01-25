"""
Miner Policy Server for Robotics Subnet

This FastAPI server exposes your robotics policy as HTTP endpoints
that validators can query during evaluation.

DEPLOYMENT OPTIONS:

1. Basilica Platform (Recommended):
   - Use kinitro CLI: kinitro basilica-push --repo YOUR_HF_REPO --revision YOUR_REVISION
   - Or use one-command deploy: kinitro miner-deploy -r YOUR_HF_REPO -p . --netuid YOUR_NETUID

2. Self-Hosted:
   - Run this server directly with uvicorn
   - Ensure your endpoint is publicly accessible

After deployment, commit your policy on-chain:
    kinitro commit --endpoint YOUR_ENDPOINT_URL --netuid YOUR_NETUID

Endpoints:
    POST /reset - Reset policy for new episode
    POST /act   - Get action given observation
    GET  /health - Health check

Usage:
    # Local testing
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import json
import logging
import time
from contextlib import contextmanager
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

# Import your policy implementation
from policy import RobotPolicy
from pydantic import BaseModel

# =============================================================================
# Structured Logging
# =============================================================================


class StructuredLogger:
    """JSON structured logger for production observability."""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
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

app = FastAPI(
    title="Robotics Policy Server",
    description="Miner policy endpoint for robotics subnet evaluation",
    version="1.0.0",
)

# Global state
_policy: RobotPolicy | None = None
_request_count = 0
_error_count = 0


def get_policy() -> RobotPolicy:
    """Get or initialize the policy."""
    global _policy
    if _policy is None:
        with logger.measure("policy_load"):
            _policy = RobotPolicy()
        logger.info(
            "Policy initialized",
            model_loaded=_policy.is_loaded() if _policy else False,
        )
    return _policy


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
    images: dict[str, list] | None = None  # Camera images as nested lists


class ActResponse(BaseModel):
    """Response body for /act endpoint."""

    action: list[float]


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""

    status: str
    model_loaded: bool
    uptime_seconds: float


# =============================================================================
# Endpoints
# =============================================================================

_start_time = time.time()


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Validators may call this to verify your endpoint is running.
    """
    policy = get_policy()
    uptime = time.time() - _start_time
    model_loaded = policy.is_loaded()

    logger.info(
        "Health check",
        status="healthy",
        model_loaded=model_loaded,
        uptime_seconds=round(uptime, 2),
        request_count=_request_count,
        error_count=_error_count,
    )

    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        uptime_seconds=uptime,
    )


@app.post("/reset", response_model=ResetResponse)
async def reset(request: ResetRequest):
    """
    Reset policy for a new episode.

    Called by validators at the start of each evaluation episode.
    Use this to reset any internal state and optionally condition
    your policy on the task configuration.

    Args:
        request: Contains task_config with environment/task info

    Returns:
        Status and optional episode ID
    """
    global _request_count, _error_count
    _request_count += 1
    task_config = request.task_config.model_dump()

    try:
        policy = get_policy()
        with logger.measure(
            "reset",
            env_id=task_config.get("env_id"),
            task_id=task_config.get("task_id"),
        ):
            episode_id = await policy.reset(task_config)
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
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/act", response_model=ActResponse)
async def act(request: ActRequest):
    """
    Get action for current observation.

    Called by validators every timestep during evaluation.
    You have approximately 500ms to respond (configurable by validator).

    Args:
        request: Contains observation (proprioceptive state) and
                 optional camera images

    Returns:
        Action as list of floats
    """
    global _request_count, _error_count
    _request_count += 1

    try:
        policy = get_policy()

        # Convert observation to numpy
        obs = np.array(request.observation, dtype=np.float32)

        # Convert images if provided
        images = None
        if request.images:
            images = {k: np.array(v) for k, v in request.images.items()}

        # Get action from policy (measure timing)
        start = time.time()
        action = await policy.act(obs, images)
        duration_ms = (time.time() - start) * 1000

        # Log periodically (every 100 requests) or if slow
        if _request_count % 100 == 0 or duration_ms > 100:
            logger.info(
                "Act request",
                duration_ms=round(duration_ms, 2),
                request_count=_request_count,
                obs_dim=len(request.observation),
                has_images=request.images is not None,
            )

        return ActResponse(action=action.tolist())
    except Exception as e:
        _error_count += 1
        logger.error(
            "Act failed",
            error=str(e),
            error_type=type(e).__name__,
            obs_dim=len(request.observation) if request.observation else 0,
        )
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Startup/Shutdown
# =============================================================================


@app.on_event("startup")
async def startup():
    """Initialize policy on startup."""
    logger.info("Starting policy server")
    get_policy()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global _policy
    logger.info(
        "Shutting down policy server",
        total_requests=_request_count,
        total_errors=_error_count,
    )
    if _policy is not None:
        await _policy.cleanup()
        _policy = None


# =============================================================================
# Run with uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
