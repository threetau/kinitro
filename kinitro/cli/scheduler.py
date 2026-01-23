"""Scheduler service command."""

import asyncio

import structlog
import typer

from .utils import normalize_database_url


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

    from kinitro.scheduler import SchedulerConfig, run_scheduler

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
