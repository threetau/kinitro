"""Environment-related commands for Kinitro CLI."""

import typer

from .commands import build_env, list_envs, test_env

# Create env app (subcommand group)
env_app = typer.Typer(
    name="env",
    help="Environment-related commands for building, listing, and testing.",
    add_completion=False,
    no_args_is_help=True,
)


# Register commands
env_app.command(name="build")(build_env)
env_app.command(name="list")(list_envs)
env_app.command(name="test")(test_env)

__all__ = ["env_app"]
