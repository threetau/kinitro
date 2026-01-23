"""Database reset command."""

import typer

from .db_create import db_create
from .db_drop import db_drop
from .db_init import db_init


def db_reset(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Reset the database (drop and recreate).

    WARNING: This will delete all data!
    """
    if not force:
        confirm = typer.confirm(
            "Are you sure you want to reset the database? All data will be lost!"
        )
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    # Drop
    db_drop(database_url=database_url, force=True)

    # Create
    db_create(database_url=database_url)

    # Init schema
    db_init(database_url=database_url)

    typer.echo("Database reset complete!")
