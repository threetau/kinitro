"""API service command."""

import structlog
import typer

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

    from kinitro.api import APIConfig, run_server

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
