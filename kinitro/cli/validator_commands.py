"""Validator commands for Kinitro CLI."""

import asyncio

import structlog
import typer

from kinitro.config import ValidatorConfig
from kinitro.validator.main import run_validator


def validate(
    backend_url: str = typer.Option(..., help="URL of the evaluation backend"),
    network: str = typer.Option("finney", help="Network: finney, test, or local"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the validator.

    Polls the backend for weights and submits them to the chain.

    The validator is lightweight - all evaluation is done by the backend.
    """
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    config = ValidatorConfig(
        network=network,
        netuid=netuid,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        log_level=log_level,
    )

    typer.echo(f"Starting validator on {network} (netuid={netuid})")
    typer.echo(f"  Backend: {backend_url}")
    asyncio.run(run_validator(config, backend_url))
