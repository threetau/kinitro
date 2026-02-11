"""Service commands (api, scheduler, executor) for Kinitro CLI."""

import asyncio
import json

import structlog
import typer

from kinitro.api import APIConfig, run_server
from kinitro.executor import ExecutorConfig, run_concurrent_executor, run_executor
from kinitro.scheduler import SchedulerConfig, run_scheduler

from .utils import normalize_database_url


def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
    no_auth: bool = typer.Option(
        False,
        "--no-auth",
        help="Disable API key authentication for task endpoints.",
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
        auth_disabled=no_auth,
        log_level=log_level,
    )

    typer.echo(f"Starting API service on {host}:{port}")
    typer.echo(f"  Database: {database_url.split('@')[-1]}")
    if no_auth:
        typer.echo("  Auth: disabled (--no-auth)")
    elif config.api_key:
        typer.echo("  Auth: enabled (KINITRO_API_API_KEY)")
    else:
        typer.echo("  Auth: enabled (WARNING: KINITRO_API_API_KEY not set)", err=True)

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
    env_families: str | None = typer.Option(
        None,
        help="Filter environments to specific families, comma-separated (e.g., metaworld,genesis)",
    ),
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

    # Parse comma-separated env_families into list
    parsed_env_families = None
    if env_families:
        parsed_env_families = [f.strip() for f in env_families.split(",") if f.strip()]

    config = SchedulerConfig(
        database_url=normalized_db_url,
        network=network,
        netuid=netuid,
        eval_interval_seconds=eval_interval,
        episodes_per_env=episodes_per_env,
        env_families=parsed_env_families,
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
    eval_images: str | None = typer.Option(
        None,
        help='JSON mapping of env family to Docker image, e.g., \'{"metaworld": "kinitro/metaworld:v1"}\'',
    ),
    eval_mode: str = typer.Option("docker", help="Evaluation mode: docker or basilica"),
    log_level: str = typer.Option("INFO", help="Logging level"),
    concurrent: bool = typer.Option(
        False,
        "--concurrent",
        help="Enable multi-process concurrent executor (one subprocess per env family)",
    ),
    max_concurrent: int = typer.Option(
        20,
        help="Max concurrent tasks per environment family (used with --concurrent)",
    ),
    eval_gpu: bool = typer.Option(
        False,
        "--eval-gpu",
        help="Enable GPU access for evaluation containers (passes --gpus all to Docker)",
    ),
    env_families: str | None = typer.Option(
        None,
        help="Comma-separated environment families to run (e.g., 'metaworld,genesis'). "
        "Defaults to families in --eval-images.",
    ),
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

    # Parse eval_images JSON if provided
    parsed_eval_images: dict[str, str] | None = None
    if eval_images:
        try:
            parsed_eval_images = json.loads(eval_images)
            if not isinstance(parsed_eval_images, dict):
                typer.echo("Error: --eval-images must be a JSON object", err=True)
                raise typer.Exit(1)
        except json.JSONDecodeError as e:
            typer.echo(f"Error: Invalid JSON for --eval-images: {e}", err=True)
            raise typer.Exit(1)

    # Validate max_concurrent when concurrent mode is enabled
    if concurrent and max_concurrent <= 0:
        typer.echo("Error: --max-concurrent must be a positive integer", err=True)
        raise typer.Exit(1)

    # Parse env_families if provided
    parsed_env_families: list[str] | None = None
    if env_families:
        parsed_env_families = [f.strip() for f in env_families.split(",") if f.strip()]

    # Validate env_families against eval_images in concurrent mode
    if concurrent and parsed_env_families:
        available_families = set(
            parsed_eval_images.keys() if parsed_eval_images else {"metaworld", "genesis"}
        )
        invalid_families = [f for f in parsed_env_families if f not in available_families]
        if invalid_families:
            typer.echo(
                f"Error: --env-families contains families without configured images: {invalid_families}. "
                f"Available families: {sorted(available_families)}",
                err=True,
            )
            raise typer.Exit(1)

    # Build config kwargs
    config_kwargs: dict = {
        "api_url": api_url,
        "batch_size": batch_size,
        "poll_interval_seconds": poll_interval,
        "eval_mode": eval_mode,
        "eval_gpu": eval_gpu,
        "log_level": log_level,
        "use_concurrent_executor": concurrent,
        "default_max_concurrent": max_concurrent,
    }
    if executor_id:
        config_kwargs["executor_id"] = executor_id
    if parsed_eval_images:
        config_kwargs["eval_images"] = parsed_eval_images
    if parsed_env_families:
        config_kwargs["env_families"] = parsed_env_families

    config = ExecutorConfig(**config_kwargs)

    typer.echo("Starting executor service")
    typer.echo(f"  Executor ID: {config.executor_id}")
    typer.echo(f"  API URL: {api_url}")
    typer.echo(f"  Auth: {'configured' if config.api_key else 'not configured'}")
    typer.echo(f"  Batch size: {batch_size}")
    typer.echo(f"  Eval images: {config.eval_images}")
    typer.echo(f"  GPU: {'enabled' if eval_gpu else 'disabled'}")
    typer.echo(f"  Concurrent mode: {concurrent}")
    if concurrent:
        typer.echo(f"  Max concurrent per family: {max_concurrent}")
        typer.echo(f"  Environment families: {config.get_env_families()}")

    if concurrent:
        asyncio.run(run_concurrent_executor(config))
    else:
        asyncio.run(run_executor(config))
