"""Database init command."""

import asyncio

import typer

from .utils import normalize_database_url


def db_init(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
):
    """
    Initialize the database schema.

    Creates all tables if they don't exist.
    """
    # Normalize URL to use asyncpg driver
    normalized_url = normalize_database_url(database_url)

    async def _init():
        from kinitro.backend.storage import Storage

        storage = Storage(normalized_url)
        await storage.initialize()
        await storage.close()

    typer.echo(f"Initializing database: {database_url.split('@')[-1]}")
    asyncio.run(_init())
    typer.echo("Database initialized successfully!")
