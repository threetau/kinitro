"""
Alembic environment configuration for evaluator database.
"""

import asyncio
import os
import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine
from sqlmodel import SQLModel

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the evaluator models to ensure they're registered with SQLModel
from evaluator.db import models  # noqa: E402, F401

# this is the Alembic Config object
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Add your model's MetaData object here for 'autogenerate' support
# Use SQLModel's metadata which includes all tables
target_metadata = SQLModel.metadata


def get_database_url():
    """Get database URL from configuration or environment."""
    # First try environment variable
    db_url = os.getenv("DATABASE_URL")
    if db_url:
        return db_url

    # Try to read from evaluator config file directly
    try:
        import toml  # noqa: PLC0415

        config_path = project_root / "config" / "evaluator.toml"
        if config_path.exists():
            config_data = toml.load(config_path)
            return config_data.get(
                "pg_database", "postgresql://postgres@localhost/evaluator_db"
            )
    except Exception:
        pass

    # Fallback to alembic config
    return config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = get_database_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    """Run migrations with database connection."""
    context.configure(connection=connection, target_metadata=target_metadata)

    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations asynchronously."""
    url = get_database_url()

    # Convert to async URL if needed
    if url.startswith("postgresql://"):
        url = url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif url.startswith("sqlite://"):
        url = url.replace("sqlite://", "sqlite+aiosqlite://", 1)

    connectable = create_async_engine(
        url,
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
