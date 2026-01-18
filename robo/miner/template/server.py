"""
Miner Policy Server for Robotics Subnet

This FastAPI server exposes your robotics policy as HTTP endpoints
that validators can query during evaluation.

Deploy this to Chutes or run self-hosted, then commit the endpoint
info on-chain so validators can find and evaluate your policy.

Endpoints:
    POST /reset - Reset policy for new episode
    POST /act   - Get action given observation
    GET  /health - Health check

Usage:
    # Local testing
    uvicorn server:app --host 0.0.0.0 --port 8000
    
    # Deploy to Chutes
    chutes deploy chute:chute
"""

import os
import time
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Import your policy implementation
from policy import RobotPolicy

app = FastAPI(
    title="Robotics Policy Server",
    description="Miner policy endpoint for robotics subnet evaluation",
    version="1.0.0",
)

# Global policy instance
_policy: RobotPolicy | None = None


def get_policy() -> RobotPolicy:
    """Get or initialize the policy."""
    global _policy
    if _policy is None:
        _policy = RobotPolicy()
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
    return HealthResponse(
        status="healthy",
        model_loaded=policy.is_loaded(),
        uptime_seconds=time.time() - _start_time,
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
    try:
        policy = get_policy()
        episode_id = await policy.reset(request.task_config.model_dump())
        return ResetResponse(status="ok", episode_id=episode_id)
    except Exception as e:
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
    try:
        policy = get_policy()
        
        # Convert observation to numpy
        obs = np.array(request.observation, dtype=np.float32)
        
        # Convert images if provided
        images = None
        if request.images:
            images = {k: np.array(v) for k, v in request.images.items()}
        
        # Get action from policy
        action = await policy.act(obs, images)
        
        return ActResponse(action=action.tolist())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Startup/Shutdown
# =============================================================================


@app.on_event("startup")
async def startup():
    """Initialize policy on startup."""
    get_policy()


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global _policy
    if _policy is not None:
        await _policy.cleanup()
        _policy = None


# =============================================================================
# Run with uvicorn
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
