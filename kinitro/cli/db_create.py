"""Database create command."""

import asyncio

import typer

from .utils import parse_database_url


def db_create(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
):
    """
    Create the database if it doesn't exist.

    Connects to the PostgreSQL server and creates the database.
    """
    try:
        user, password, host, port, dbname = parse_database_url(database_url)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    async def _create():
        import asyncpg

        # Connect to the default 'postgres' database to create our database
        conn = await asyncpg.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database="postgres",
        )

        try:
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                dbname,
            )

            if exists:
                typer.echo(f"Database '{dbname}' already exists.")
            else:
                # Create the database
                await conn.execute(f'CREATE DATABASE "{dbname}"')
                typer.echo(f"Database '{dbname}' created successfully!")
        finally:
            await conn.close()

    typer.echo(f"Creating database '{dbname}' on {host}:{port}...")
    asyncio.run(_create())
