"""Build miner Docker image command."""

import typer


def build(
    env_path: str = typer.Argument(..., help="Path to env directory"),
    tag: str = typer.Option(..., help="Docker tag (e.g., user/repo:v1)"),
    push: bool = typer.Option(False, help="Push to registry after building"),
):
    """
    Build miner Docker image.

    Builds a Docker image from your policy directory.
    """
    import subprocess

    typer.echo(f"Building Docker image: {tag}")

    # Build
    result = subprocess.run(
        ["docker", "build", "-t", tag, env_path],
        capture_output=True,
        text=True,
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
        )

        if result.returncode != 0:
            typer.echo(f"Push failed:\n{result.stderr}", err=True)
            raise typer.Exit(1)

        typer.echo("Push successful!")
