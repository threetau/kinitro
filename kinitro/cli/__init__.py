"""CLI commands for Kinitro."""

import typer
from click import Context
from typer.core import TyperGroup

# Import all command modules
from kinitro.cli.db_commands import db_app
from kinitro.cli.env import env_app
from kinitro.cli.miner import miner_app
from kinitro.cli.service_commands import api, executor, scheduler
from kinitro.cli.testing_commands import test_scoring
from kinitro.cli.validator_commands import validate


class OrderedCommands(TyperGroup):
    """Custom TyperGroup that preserves command order instead of sorting alphabetically."""

    def list_commands(self, ctx: Context):
        """
        Return commands in the order they were added.

        Args:
            ctx: Click context object.

        Returns:
            List of command names in their defined order.
        """
        # Define the desired order
        order = [
            "miner",
            "env",
            "executor",
            "scheduler",
            "validate",
            "api",
            "db",
            "test-scoring",
        ]
        # Return commands in the specified order, followed by any not in the list
        ordered = [cmd for cmd in order if cmd in self.commands]
        additional = [cmd for cmd in self.commands if cmd not in ordered]
        return ordered + additional


# Create main app with custom command ordering
app = typer.Typer(
    name="kinitro",
    help="Kinitro - Robotics Generalization Subnet CLI",
    add_completion=False,
    no_args_is_help=True,
    cls=OrderedCommands,
)

# Add miner commands as a subcommand group
app.add_typer(miner_app, name="miner")

# Add env commands as a subcommand group
app.add_typer(env_app, name="env")

# Add service commands
app.command()(api)
app.command()(scheduler)
app.command()(executor)

# Add validator commands
app.command()(validate)

# Add testing commands
app.command()(test_scoring)

# Add database commands as a subcommand group
app.add_typer(db_app, name="db")

__all__ = ["app"]
