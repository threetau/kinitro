"""Database management commands for Kinitro CLI."""

import asyncio

import asyncpg
import typer

from kinitro.backend.storage import Storage

from .utils import normalize_database_url, parse_database_url

# Subcommand group for database management
db_app = typer.Typer(
    help="Database management commands", add_completion=False, no_args_is_help=True
)


def _quote_ident(name: str) -> str:
    """Quote a PostgreSQL identifier safely."""
    # Double any double quotes and wrap in double quotes
    return '"' + name.replace('"', '""') + '"'


@db_app.command("init")
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
        storage = Storage(normalized_url)
        await storage.initialize()
        await storage.close()

    typer.echo(f"Initializing database: {database_url.rsplit('@', maxsplit=1)[-1]}")
    asyncio.run(_init())
    typer.echo("Database initialized successfully!")


@db_app.command("create")
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
                await conn.execute(f"CREATE DATABASE {_quote_ident(dbname)}")
                typer.echo(f"Database '{dbname}' created successfully!")
        finally:
            await conn.close()

    typer.echo(f"Creating database '{dbname}' on {host}:{port}...")
    asyncio.run(_create())


@db_app.command("drop")
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
        conn = await asyncpg.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database="postgres",
        )

        try:
            # Terminate existing connections
            await conn.execute(
                "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                "WHERE datname = $1 AND pid <> pg_backend_pid()",
                dbname,
            )

            # Drop the database
            await conn.execute(f"DROP DATABASE IF EXISTS {_quote_ident(dbname)}")
            typer.echo(f"Database '{dbname}' dropped successfully!")
        finally:
            await conn.close()

    typer.echo(f"Dropping database '{dbname}'...")
    asyncio.run(_drop())


@db_app.command("reset")
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


@db_app.command("status")
def db_status(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
):
    """
    Show database status and statistics.
    """
    # Normalize URL to use asyncpg driver
    normalized_url = normalize_database_url(database_url)

    async def _status():
        storage = Storage(normalized_url)

        try:
            async with storage.session() as session:
                total_cycles = await storage.count_cycles(session)
                total_miners = await storage.count_unique_miners(session)
                latest_cycle = await storage.get_latest_cycle(session, completed_only=True)
                running_cycle = await storage.get_running_cycle(session)
                latest_weights = await storage.get_latest_weights(session)

                typer.echo("Database Status:")
                typer.echo(f"  Total evaluation cycles: {total_cycles}")
                typer.echo(f"  Unique miners evaluated: {total_miners}")

                if latest_cycle:
                    typer.echo("\nLatest completed cycle:")
                    typer.echo(f"  ID: {latest_cycle.id}")
                    typer.echo(f"  Block: {latest_cycle.block_number}")
                    typer.echo(f"  Status: {latest_cycle.status}")
                    typer.echo(f"  Miners: {latest_cycle.n_miners}")
                    typer.echo(
                        f"  Duration: {latest_cycle.duration_seconds:.1f}s"
                        if latest_cycle.duration_seconds
                        else "  Duration: N/A"
                    )
                else:
                    typer.echo("\nNo completed evaluation cycles yet.")

                if running_cycle:
                    typer.echo("\nCurrently running cycle:")
                    typer.echo(f"  ID: {running_cycle.id}")
                    typer.echo(f"  Block: {running_cycle.block_number}")
                    typer.echo(f"  Started: {running_cycle.started_at}")

                if latest_weights:
                    typer.echo("\nLatest weights:")
                    typer.echo(f"  Block: {latest_weights.block_number}")
                    typer.echo(f"  Miners: {len(latest_weights.weights_json)}")
                    typer.echo(f"  Created: {latest_weights.created_at}")

        except Exception as e:
            typer.echo(f"Error connecting to database: {e}", err=True)
            raise typer.Exit(1)
        finally:
            await storage.close()

    asyncio.run(_status())
