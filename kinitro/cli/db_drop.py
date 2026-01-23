"""Database drop command."""

import asyncio

import typer

from .utils import parse_database_url


def db_drop(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Drop the database.

    WARNING: This will delete all data!
    """
    try:
        user, password, host, port, dbname = parse_database_url(database_url)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to drop database '{dbname}'? This cannot be undone!"
        )
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    async def _drop():
        import asyncpg

        conn = await asyncpg.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database="postgres",
        )

        try:
            # Terminate existing connections
            await conn.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{dbname}'
                AND pid <> pg_backend_pid()
            """)

            # Drop the database
            await conn.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
            typer.echo(f"Database '{dbname}' dropped successfully!")
        finally:
            await conn.close()

    typer.echo(f"Dropping database '{dbname}'...")
    asyncio.run(_drop())
