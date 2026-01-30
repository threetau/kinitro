"""Service commands (api, scheduler, executor) for Kinitro CLI."""

import asyncio

import structlog
import typer

from kinitro.api import APIConfig, run_server
from kinitro.executor import ExecutorConfig, run_executor
from kinitro.scheduler import SchedulerConfig, run_scheduler

from .utils import normalize_database_url


def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the API service (lightweight REST API).

    The API service provides:
    - Task pool endpoints for executors (fetch/submit)
    - Weights endpoint for validators
    - Scores and status endpoints

    This is the stateless API layer of the split architecture.
    Run separately from scheduler and executor.
    """
    log_level = log_level.upper()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    normalized_db_url = normalize_database_url(database_url)

    config = APIConfig(
        host=host,
        port=port,
        database_url=normalized_db_url,
        log_level=log_level,
    )

    typer.echo(f"Starting API service on {host}:{port}")
    typer.echo(f"  Database: {database_url.split('@')[-1]}")

    run_server(config)


def scheduler(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    eval_interval: int = typer.Option(3600, help="Seconds between evaluation cycles"),
    episodes_per_env: int = typer.Option(50, help="Episodes per environment"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the scheduler service.

    The scheduler is responsible for:
    - Reading miner commitments from the chain
    - Creating evaluation tasks in the task pool
    - Waiting for executors to complete tasks
    - Computing Pareto scores and weights
    - Storing results in the database

    Requires the API service to be running for executors to fetch tasks.
    """
    log_level = log_level.upper()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    normalized_db_url = normalize_database_url(database_url)

    config = SchedulerConfig(
        database_url=normalized_db_url,
        network=network,
        netuid=netuid,
        eval_interval_seconds=eval_interval,
        episodes_per_env=episodes_per_env,
        log_level=log_level,
    )

    typer.echo("Starting scheduler service")
    typer.echo(f"  Network: {network} (netuid={netuid})")
    typer.echo(f"  Database: {database_url.split('@')[-1]}")
    typer.echo(f"  Eval interval: {eval_interval}s")

    asyncio.run(run_scheduler(config))


def executor(
    api_url: str = typer.Option(
        "http://localhost:8000",
        help="URL of the Kinitro API service",
    ),
    executor_id: str | None = typer.Option(
        None,
        help="Unique ID for this executor (auto-generated if not provided)",
    ),
    batch_size: int = typer.Option(10, help="Number of tasks to fetch at a time"),
    poll_interval: int = typer.Option(5, help="Seconds between polling for tasks"),
    eval_image: str = typer.Option(
        "kinitro/eval-env:v1",
        help="Docker image for evaluation environment",
    ),
    eval_mode: str = typer.Option("docker", help="Evaluation mode: docker or basilica"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the executor service.

    The executor fetches tasks from the API and runs MuJoCo evaluations
    using affinetes. It calls miner policy endpoints to get actions.

    Multiple executors can run in parallel on different GPU machines
    to scale evaluation throughput.

    Examples:
        # Run with default settings
        kinitro executor --api-url http://api.kinitro.io:8000

        # Run on specific GPU with custom batch size
        CUDA_VISIBLE_DEVICES=0 kinitro executor --batch-size 20
    """
    log_level = log_level.upper()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    config_kwargs = {
        "api_url": api_url,
        "batch_size": batch_size,
        "poll_interval_seconds": poll_interval,
        "eval_image": eval_image,
        "eval_mode": eval_mode,
        "log_level": log_level,
    }

    if executor_id:
        config_kwargs["executor_id"] = executor_id

    config = ExecutorConfig(**config_kwargs)

    typer.echo("Starting executor service")
    typer.echo(f"  Executor ID: {config.executor_id}")
    typer.echo(f"  API URL: {api_url}")
    typer.echo(f"  Batch size: {batch_size}")
    typer.echo(f"  Eval image: {eval_image}")

    asyncio.run(run_executor(config))
