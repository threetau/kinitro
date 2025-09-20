#!/usr/bin/env python3
"""
CLI tool for managing Kinitro Backend API keys and administration.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import click
import toml
from rich.console import Console
from rich.table import Table
from snowflake import SnowflakeGenerator
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

from backend.auth import UserRole, generate_api_key, hash_api_key
from backend.models import ApiKey

console = Console()


class ApiKeyManager:
    """Manages API key operations."""

    def __init__(self, database_url: str):
        self.database_url = database_url
        self.engine = create_async_engine(database_url)
        self.id_generator = SnowflakeGenerator(42)

    async def create_api_key(
        self,
        name: str,
        role: str,
        description: Optional[str] = None,
        associated_hotkey: Optional[str] = None,
        expires_at: Optional[datetime] = None,
    ) -> tuple[ApiKey, str]:
        """Create a new API key."""
        api_key = generate_api_key()
        key_hash = hash_api_key(api_key)

        async with AsyncSession(self.engine) as session:
            db_api_key = ApiKey(
                id=next(self.id_generator),
                name=name,
                description=description,
                key_hash=key_hash,
                role=role,
                associated_hotkey=associated_hotkey,
                expires_at=expires_at,
                is_active=True,
            )

            session.add(db_api_key)
            await session.commit()
            await session.refresh(db_api_key)

            return db_api_key, api_key

    async def list_api_keys(
        self, role: Optional[str] = None, active_only: bool = False
    ):
        """List all API keys."""
        async with AsyncSession(self.engine) as session:
            query = select(ApiKey)

            if role:
                query = query.where(ApiKey.role == role)

            if active_only:
                query = query.where(ApiKey.is_active.is_(True))

            result = await session.execute(query.order_by(ApiKey.created_at.desc()))
            return result.scalars().all()

    async def get_api_key(self, key_id: int) -> Optional[ApiKey]:
        """Get an API key by ID."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
            return result.scalar_one_or_none()

    async def deactivate_api_key(self, key_id: int) -> bool:
        """Deactivate an API key."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
            api_key = result.scalar_one_or_none()

            if not api_key:
                return False

            api_key.is_active = False
            await session.commit()
            return True

    async def activate_api_key(self, key_id: int) -> bool:
        """Activate an API key."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
            api_key = result.scalar_one_or_none()

            if not api_key:
                return False

            api_key.is_active = True
            await session.commit()
            return True

    async def delete_api_key(self, key_id: int) -> bool:
        """Delete an API key permanently."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
            api_key = result.scalar_one_or_none()

            if not api_key:
                return False

            await session.delete(api_key)
            await session.commit()
            return True

    async def check_admin_exists(self) -> bool:
        """Check if any admin API key exists."""
        async with AsyncSession(self.engine) as session:
            result = await session.execute(
                select(ApiKey).where(ApiKey.role == UserRole.ADMIN)
            )
            return result.scalar_one_or_none() is not None

    async def cleanup(self):
        """Cleanup database connections."""
        await self.engine.dispose()


@click.group()
@click.option(
    "--database-url",
    envvar="DATABASE_URL",
    help="Database URL (can also be set via DATABASE_URL env var)",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to backend.toml config file",
)
@click.pass_context
def cli(ctx, database_url, config):
    """Kinitro Backend CLI - Manage API keys and administration."""
    # Get database URL from config if not provided
    if not database_url:
        # Try to load from config file
        config_paths = []

        # If config path is specified, use it first
        if config:
            config_paths.append(config)

        # Add default locations
        config_paths.extend(
            [
                "backend.toml",
                "config/backend.toml",
                os.path.expanduser("~/.kinitro/backend.toml"),
            ]
        )

        for config_path in config_paths:
            if os.path.exists(config_path):
                try:
                    with open(config_path, "r") as f:
                        config_data = toml.load(f)
                        database_url = config_data.get("database_url")
                        if database_url:
                            console.print(
                                f"[dim]Using config from: {config_path}[/dim]"
                            )
                            break
                except Exception as e:
                    if config:  # Only show error if user explicitly specified this file
                        console.print(
                            f"[yellow]Warning:[/yellow] Failed to load config from {config_path}: {e}"
                        )
                    pass

    if not database_url:
        console.print(
            "[red]ERROR:[/red] No database URL configured. "
            "Set --database-url or DATABASE_URL environment variable."
        )
        sys.exit(1)

    ctx.obj = ApiKeyManager(database_url)


@cli.command()
@click.option("--name", prompt=True, help="Name for the API key")
@click.option(
    "--role",
    type=click.Choice(["admin", "validator", "viewer"], case_sensitive=False),
    prompt=True,
    help="Role for the API key",
)
@click.option("--description", help="Description of the API key")
@click.option("--hotkey", help="Associated hotkey (for validators)")
@click.option("--expires", help="Expiration date (YYYY-MM-DD)")
@click.option(
    "--force-admin",
    is_flag=True,
    help="Force creation of admin key even if one exists",
)
@click.pass_context
def create(ctx, name, role, description, hotkey, expires, force_admin):
    """Create a new API key."""
    manager = ctx.obj

    async def _create():
        # Check if creating admin and one already exists
        if role == "admin" and not force_admin:
            if await manager.check_admin_exists():
                console.print(
                    "[yellow]Warning:[/yellow] An admin API key already exists. "
                    "Use --force-admin to create another one."
                )
                if not click.confirm("Do you want to continue?"):
                    return

        # Parse expiration date if provided
        expires_at = None
        if expires:
            try:
                expires_at = datetime.strptime(expires, "%Y-%m-%d").replace(
                    tzinfo=timezone.utc
                )
            except ValueError:
                console.print(f"[red]ERROR:[/red] Invalid date format: {expires}")
                return

        # Create the API key
        try:
            api_key_obj, api_key = await manager.create_api_key(
                name=name,
                role=role,
                description=description,
                associated_hotkey=hotkey,
                expires_at=expires_at,
            )

            console.print("\n[green]✓[/green] API key created successfully!\n")
            console.print(f"  [bold]ID:[/bold]        {api_key_obj.id}")
            console.print(f"  [bold]Name:[/bold]      {api_key_obj.name}")
            console.print(f"  [bold]Role:[/bold]      {api_key_obj.role}")
            if api_key_obj.description:
                console.print(f"  [bold]Desc:[/bold]      {api_key_obj.description}")
            if api_key_obj.associated_hotkey:
                console.print(
                    f"  [bold]Hotkey:[/bold]    {api_key_obj.associated_hotkey}"
                )
            if api_key_obj.expires_at:
                console.print(f"  [bold]Expires:[/bold]   {api_key_obj.expires_at}")
            console.print(f"  [bold]Created:[/bold]   {api_key_obj.created_at}")

            console.print(f"\n  [bold cyan]API Key:[/bold cyan]   {api_key}\n")
            console.print(
                "[yellow]⚠ IMPORTANT:[/yellow] Save this API key securely! "
                "It will not be displayed again.\n"
            )

        except Exception as e:
            console.print(f"[red]ERROR:[/red] Failed to create API key: {e}")

        await manager.cleanup()

    asyncio.run(_create())


@cli.command(name="list")
@click.option(
    "--role",
    type=click.Choice(["admin", "validator", "viewer"], case_sensitive=False),
    help="Filter by role",
)
@click.option("--active-only", is_flag=True, help="Show only active keys")
@click.option("--show-id", is_flag=True, help="Show full ID numbers")
@click.pass_context
def list(ctx, role, active_only, show_id):
    """List all API keys."""
    manager = ctx.obj

    async def _list():
        keys = await manager.list_api_keys(role=role, active_only=active_only)

        if not keys:
            console.print("No API keys found.")
            await manager.cleanup()
            return

        table = Table(title="API Keys")

        if show_id:
            table.add_column("ID", style="cyan")
        else:
            table.add_column("ID (last 6)", style="cyan")

        table.add_column("Name", style="white")
        table.add_column("Role", style="magenta")
        table.add_column("Active", style="green")
        table.add_column("Hotkey", style="yellow")
        table.add_column("Last Used", style="blue")
        table.add_column("Created", style="dim")

        for key in keys:
            id_display = str(key.id) if show_id else f"...{str(key.id)[-6:]}"
            active_display = "✓" if key.is_active else "✗"
            hotkey_display = (
                key.associated_hotkey[:8] + "..." if key.associated_hotkey else "-"
            )
            last_used = (
                key.last_used_at.strftime("%Y-%m-%d %H:%M")
                if key.last_used_at
                else "Never"
            )
            created = key.created_at.strftime("%Y-%m-%d %H:%M")

            table.add_row(
                id_display,
                key.name,
                key.role,
                active_display,
                hotkey_display,
                last_used,
                created,
            )

        console.print(table)
        console.print(f"\nTotal: {len(keys)} key(s)")

        await manager.cleanup()

    asyncio.run(_list())


@cli.command()
@click.argument("key_id", type=int)
@click.pass_context
def show(ctx, key_id):
    """Show details of a specific API key."""
    manager = ctx.obj

    async def _show():
        api_key = await manager.get_api_key(key_id)

        if not api_key:
            console.print(f"[red]ERROR:[/red] API key with ID {key_id} not found.")
            await manager.cleanup()
            return

        console.print(f"\n[bold]API Key Details[/bold]\n")
        console.print(f"  [bold]ID:[/bold]              {api_key.id}")
        console.print(f"  [bold]Name:[/bold]            {api_key.name}")
        console.print(f"  [bold]Role:[/bold]            {api_key.role}")
        console.print(
            f"  [bold]Active:[/bold]          {'Yes' if api_key.is_active else 'No'}"
        )

        if api_key.description:
            console.print(f"  [bold]Description:[/bold]     {api_key.description}")

        if api_key.associated_hotkey:
            console.print(
                f"  [bold]Associated Key:[/bold]  {api_key.associated_hotkey}"
            )

        if api_key.last_used_at:
            console.print(f"  [bold]Last Used:[/bold]       {api_key.last_used_at}")
        else:
            console.print("  [bold]Last Used:[/bold]       Never")

        if api_key.expires_at:
            console.print(f"  [bold]Expires:[/bold]         {api_key.expires_at}")
            if api_key.expires_at < datetime.now(timezone.utc):
                console.print("  [red]Status:[/red]          EXPIRED")

        console.print(f"  [bold]Created:[/bold]         {api_key.created_at}")
        console.print(f"  [bold]Updated:[/bold]         {api_key.updated_at}")
        console.print()

        await manager.cleanup()

    asyncio.run(_show())


@cli.command()
@click.argument("key_id", type=int)
@click.pass_context
def deactivate(ctx, key_id):
    """Deactivate an API key."""
    manager = ctx.obj

    async def _deactivate():
        if await manager.deactivate_api_key(key_id):
            console.print(
                f"[green]✓[/green] API key {key_id} deactivated successfully."
            )
        else:
            console.print(f"[red]ERROR:[/red] API key with ID {key_id} not found.")

        await manager.cleanup()

    asyncio.run(_deactivate())


@cli.command()
@click.argument("key_id", type=int)
@click.pass_context
def activate(ctx, key_id):
    """Activate an API key."""
    manager = ctx.obj

    async def _activate():
        if await manager.activate_api_key(key_id):
            console.print(f"[green]✓[/green] API key {key_id} activated successfully.")
        else:
            console.print(f"[red]ERROR:[/red] API key with ID {key_id} not found.")

        await manager.cleanup()

    asyncio.run(_activate())


@cli.command()
@click.argument("key_id", type=int)
@click.confirmation_option(prompt="Are you sure you want to delete this API key?")
@click.pass_context
def delete(ctx, key_id):
    """Delete an API key permanently."""
    manager = ctx.obj

    async def _delete():
        if await manager.delete_api_key(key_id):
            console.print(f"[green]✓[/green] API key {key_id} deleted permanently.")
        else:
            console.print(f"[red]ERROR:[/red] API key with ID {key_id} not found.")

        await manager.cleanup()

    asyncio.run(_delete())


@cli.command()
@click.pass_context
def init(ctx):
    """Initialize the backend with an admin API key."""
    manager = ctx.obj

    async def _init():
        # Check if admin already exists
        if await manager.check_admin_exists():
            console.print(
                "[yellow]Admin API key already exists.[/yellow] "
                "Use 'create' command to create additional keys."
            )
            await manager.cleanup()
            return

        console.print("[bold]Initializing Kinitro Backend[/bold]\n")
        console.print("Creating initial admin API key...")

        try:
            api_key_obj, api_key = await manager.create_api_key(
                name="Initial Admin",
                role=UserRole.ADMIN,
                description="Initial admin API key created during setup",
            )

            console.print(
                "\n[green]✓[/green] Initial admin API key created successfully!\n"
            )
            console.print(f"  [bold]ID:[/bold]        {api_key_obj.id}")
            console.print(f"  [bold]Name:[/bold]      {api_key_obj.name}")
            console.print(f"  [bold]Role:[/bold]      {api_key_obj.role}")
            console.print(f"  [bold]Created:[/bold]   {api_key_obj.created_at}")

            console.print(f"\n  [bold cyan]API Key:[/bold cyan]   {api_key}\n")
            console.print(
                "[yellow]⚠ IMPORTANT:[/yellow] Save this API key securely! "
                "This is the only time it will be displayed.\n"
            )
            console.print("Next steps:")
            console.print("1. Save the API key in a secure location")
            console.print("2. Use it with the X-API-Key header for admin endpoints")
            console.print("3. Create additional API keys as needed using this CLI\n")

        except Exception as e:
            console.print(f"[red]ERROR:[/red] Failed to create admin API key: {e}")

        await manager.cleanup()

    asyncio.run(_init())


if __name__ == "__main__":
    cli()
