"""Executor service command."""

import asyncio

import structlog
import typer


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

    from kinitro.executor import ExecutorConfig, run_executor

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
