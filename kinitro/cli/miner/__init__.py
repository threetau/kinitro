"""Miner commands for Kinitro CLI."""

import subprocess

import typer

from .commitment import commit, show_commitment
from .deploy import basilica_push, miner_deploy
from .mock import mock
from .template import init_miner
from .verify import verify


def build(
    env_path: str = typer.Argument(..., help="Path to env directory"),
    tag: str = typer.Option(..., help="Docker tag (e.g., user/repo:v1)"),
    push: bool = typer.Option(False, help="Push to registry after building"),
):
    """
    Build miner Docker image.

    Builds a Docker image from your policy directory.
    """

    typer.echo(f"Building Docker image: {tag}")

    # Build
    result = subprocess.run(
        ["docker", "build", "-t", tag, env_path],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        typer.echo(f"Build failed:\n{result.stderr}", err=True)
        raise typer.Exit(1)

    typer.echo("Build successful!")

    # Push if requested
    if push:
        typer.echo(f"Pushing to registry: {tag}")
        result = subprocess.run(
            ["docker", "push", tag],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            typer.echo(f"Push failed:\n{result.stderr}", err=True)
            raise typer.Exit(1)

        typer.echo("Push successful!")


# Create miner app (subcommand group)
miner_app = typer.Typer(
    name="miner",
    help="Miner-specific commands for deployment and management.",
    add_completion=False,
    no_args_is_help=True,
)


# Register commands
miner_app.command()(build)
miner_app.command()(commit)
miner_app.command(name="show-commitment")(show_commitment)
miner_app.command(name="init")(init_miner)
miner_app.command(name="push")(basilica_push)
miner_app.command(name="deploy")(miner_deploy)
miner_app.command()(mock)
miner_app.command()(verify)

__all__ = ["miner_app"]
