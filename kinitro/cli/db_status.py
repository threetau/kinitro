"""Database status command."""

import asyncio

import typer

from .utils import normalize_database_url


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
        from kinitro.backend.storage import Storage

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
