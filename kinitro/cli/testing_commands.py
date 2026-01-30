"""Testing commands for Kinitro CLI."""

import typer


def test_scoring(
    n_miners: int = typer.Option(5, help="Number of simulated miners"),
    n_envs: int = typer.Option(3, help="Number of environments"),
):
    """
    Test the Pareto scoring mechanism with simulated data.
    """
    import numpy as np

    from kinitro.scoring.pareto import compute_pareto_frontier
    from kinitro.scoring.winners_take_all import compute_full_scoring

    typer.echo(f"Testing Pareto scoring with {n_miners} miners, {n_envs} environments\n")

    # Generate random scores
    env_ids = [f"env_{i}" for i in range(n_envs)]
    miner_scores = {}

    for uid in range(n_miners):
        miner_scores[uid] = {env_id: float(np.random.uniform(0.3, 0.9)) for env_id in env_ids}

    # Display scores
    typer.echo("Miner scores:")
    for uid, scores in miner_scores.items():
        scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())
        typer.echo(f"  UID {uid}: {scores_str}")

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(miner_scores, env_ids, n_samples_per_env=50)
    typer.echo(f"\nPareto frontier: {pareto.frontier_uids}")

    # Compute weights
    weights = compute_full_scoring(miner_scores, env_ids)
    typer.echo("\nFinal weights:")
    for uid, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"  UID {uid}: {weight:.4f}")


def mock_miner(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
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
    import numpy as np
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    from kinitro.rl_interface import CanonicalAction, CanonicalObservation

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
        obs: CanonicalObservation

    class ActResponse(BaseModel):
        action: CanonicalAction

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        uptime_seconds: float = 0.0

    import time

    _start_time = time.time()

    @mock_app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            uptime_seconds=time.time() - _start_time,
        )

    @mock_app.post("/reset", response_model=ResetResponse)
    async def reset(request: ResetRequest):
        typer.echo(f"Reset: env_id={request.task_config.env_id}, seed={request.task_config.seed}")
        return ResetResponse(status="ok", episode_id="mock-episode")

    @mock_app.post("/act", response_model=ActResponse)
    async def act(request: ActRequest):
        if random_actions:
            twist = np.random.uniform(-1, 1, size=6)
            gripper = float(np.random.uniform(0, 1))
        else:
            twist = np.zeros(6)
            gripper = 0.0
        action = CanonicalAction(twist_ee_norm=twist.tolist(), gripper_01=gripper)
        return ActResponse(action=action)

    typer.echo(f"Starting mock miner server on {host}:{port}")
    typer.echo(f"  Random actions: {random_actions}")
    typer.echo(f"  Health: http://{host}:{port}/health")
    typer.echo(f"  Reset:  POST http://{host}:{port}/reset")
    typer.echo(f"  Act:    POST http://{host}:{port}/act")
    typer.echo("\nPress Ctrl+C to stop\n")

    uvicorn.run(mock_app, host=host, port=port, log_level="info")
