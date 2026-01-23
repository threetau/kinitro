"""Mock miner server command."""

import time

import numpy as np
import typer
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


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
        observation: list[float]
        images: dict[str, list] | None = None

    class ActResponse(BaseModel):
        action: list[float]

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        uptime_seconds: float = 0.0

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
            action = np.random.uniform(-1, 1, size=4).tolist()
        else:
            # Zero action (no movement)
            action = [0.0, 0.0, 0.0, 0.0]
        return ActResponse(action=action)

    typer.echo(f"Starting mock miner server on {host}:{port}")
    typer.echo(f"  Random actions: {random_actions}")
    typer.echo(f"  Health: http://{host}:{port}/health")
    typer.echo(f"  Reset:  POST http://{host}:{port}/reset")
    typer.echo(f"  Act:    POST http://{host}:{port}/act")
    typer.echo("\nPress Ctrl+C to stop\n")

    uvicorn.run(mock_app, host=host, port=port, log_level="info")
