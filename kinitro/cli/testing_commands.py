"""Testing commands for Kinitro CLI."""

import time

import numpy as np
import typer
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from kinitro.rl_interface import Action, ActionKeys, Observation
from kinitro.scoring.pareto import compute_pareto_frontier
from kinitro.scoring.threshold import compute_miner_thresholds
from kinitro.scoring.winners_take_all import (
    compute_subset_scores_with_priority,
    scores_to_weights,
)


def test_scoring(
    n_miners: int = typer.Option(5, help="Number of simulated miners"),
    n_envs: int = typer.Option(3, help="Number of environments"),
    episodes_per_env: int = typer.Option(50, help="Simulated episodes per environment"),
):
    """
    Test the Pareto scoring mechanism with simulated data.

    Demonstrates first-commit advantage: earlier miners win ties.
    """
    typer.echo(f"Testing Pareto scoring with {n_miners} miners, {n_envs} environments\n")

    # Generate random scores and commit blocks
    env_ids = [f"env_{i}" for i in range(n_envs)]
    miner_scores = {}
    miner_blocks = {}

    for uid in range(n_miners):
        miner_scores[uid] = {env_id: float(np.random.uniform(0.3, 0.9)) for env_id in env_ids}
        miner_blocks[uid] = 1000 + uid * 100  # Earlier UIDs committed earlier

    # Display scores
    typer.echo("Miner scores (earlier block = first-commit advantage):")
    for uid, scores in miner_scores.items():
        scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())
        typer.echo(f"  UID {uid} (block {miner_blocks[uid]}): {scores_str}")

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(miner_scores, env_ids, n_samples_per_env=episodes_per_env)
    typer.echo(f"\nPareto frontier: {pareto.frontier_uids}")

    # Compute thresholds and scores with priority
    miner_thresholds = compute_miner_thresholds(miner_scores, episodes_per_env)
    subset_scores = compute_subset_scores_with_priority(
        miner_scores, miner_thresholds, miner_blocks, env_ids
    )
    weights = scores_to_weights(subset_scores)

    typer.echo("\nSubset scores:")
    for uid, score in sorted(subset_scores.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"  UID {uid}: {score:.1f} points")

    typer.echo("\nFinal weights:")
    for uid, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"  UID {uid}: {weight:.4f}")


def mock_miner(
    host: str = typer.Option("127.0.0.1", help="Host to bind to (use 0.0.0.0 to expose)"),
    port: int = typer.Option(8001, help="Port to bind to"),
    random_actions: bool = typer.Option(True, help="Return random actions (won't solve tasks)"),
):
    """
    Run a mock miner policy server for testing.

    This starts a FastAPI server that implements the miner policy API
    with random actions. Useful for testing the evaluation pipeline
    without a real trained policy.

    Examples:
        # Start mock miner on default port
        kinitro mock-miner

        # Start on specific port
        kinitro mock-miner --port 8002
    """
    mock_app = FastAPI(title="Mock Miner Policy Server")

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
        obs: Observation

    class ActResponse(BaseModel):
        action: Action

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        uptime_seconds: float = 0.0

    _start_time = time.time()

    @mock_app.get("/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint for mock miner."""
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            uptime_seconds=time.time() - _start_time,
        )

    @mock_app.post("/reset", response_model=ResetResponse)
    async def reset(request: ResetRequest):
        """Reset the policy for a new task."""
        typer.echo(f"Reset: env_id={request.task_config.env_id}, seed={request.task_config.seed}")
        return ResetResponse(status="ok", episode_id="mock-episode")

    @mock_app.post("/act", response_model=ActResponse)
    async def act(request: ActRequest):
        """Generate an action given an observation."""
        if random_actions:
            twist = np.random.uniform(-1, 1, size=6).tolist()
            gripper = [float(np.random.uniform(0, 1))]
        else:
            twist = [0.0] * 6
            gripper = [0.0]
        action = Action(
            continuous={
                ActionKeys.EE_TWIST: twist,
                ActionKeys.GRIPPER: gripper,
            }
        )
        return ActResponse(action=action)

    typer.echo(f"Starting mock miner server on {host}:{port}")
    typer.echo(f"  Random actions: {random_actions}")
    typer.echo(f"  Health: http://{host}:{port}/health")
    typer.echo(f"  Reset:  POST http://{host}:{port}/reset")
    typer.echo(f"  Act:    POST http://{host}:{port}/act")
    typer.echo("\nPress Ctrl+C to stop\n")

    uvicorn.run(mock_app, host=host, port=port, log_level="info")
