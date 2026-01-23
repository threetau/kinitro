"""List environments command."""

import typer


def list_envs():
    """List all available robotics environments."""
    from kinitro.environments.registry import get_all_environment_ids

    typer.echo("Available Robotics Environments:\n")

    typer.echo("  METAWORLD (Manipulation):")
    for env_id in get_all_environment_ids():
        typer.echo(f"    - {env_id}")

    typer.echo()
    total = len(get_all_environment_ids())
    typer.echo(f"Total: {total} environments available")
