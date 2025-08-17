from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

from sqlalchemy import (
    Engine,
    create_engine,
    inspect,
    text,
)
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import QueuePool

from ..config import get_config
from .schema import Base

logger = logging.getLogger(__name__)


class DatabaseConfig:
    def __init__(self) -> None:
        self._database_url = str(get_config().database_url)

        self.pool_min_conn: int = 1
        self.pool_max_conn: int = 10
        self.pool_timeout: int = 30
        self.pool_recycle: int = 3600

    @property
    def database_url(self) -> str:
        """Get database URL from env or config."""
        return self._database_url


class DatabaseManager:
    """
    Manages PostgreSQL database connections using SQLAlchemy.

    Provides connection pooling and database operations for the
    evaluation system using SQLAlchemy ORM.
    """

    def __init__(self, config: Optional[DatabaseConfig] = None):
        self.config = config or DatabaseConfig()
        self._engine: Optional[Engine] = None
        self._session_factory: Optional[sessionmaker] = None

    def initialize_engine(self) -> None:
        """Initialize SQLAlchemy engine with connection pooling."""
        if self._engine is not None:
            return

        try:
            self._engine = create_engine(
                self.config.database_url,
                poolclass=QueuePool,
                pool_size=self.config.pool_min_conn,
                max_overflow=self.config.pool_max_conn - self.config.pool_min_conn,
                pool_timeout=self.config.pool_timeout,
                pool_recycle=self.config.pool_recycle,
                echo=False,  # Set to True for SQL debugging
            )

            self._session_factory = sessionmaker(bind=self._engine)

            logger.info(
                f"Initialized SQLAlchemy engine with pool (size={self.config.pool_min_conn}, max_overflow={self.config.pool_max_conn - self.config.pool_min_conn})"
            )
        except Exception as e:
            logger.error(f"Failed to initialize database engine: {e}")
            raise

    @contextmanager
    def get_session(self) -> Generator[Session, None, None]:
        """Get a database session from the session factory."""
        if self._session_factory is None:
            self.initialize_engine()

        if self._session_factory is None:
            raise RuntimeError("Session factory is not initialized")

        session = self._session_factory()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()

    def dispose_engine(self) -> None:
        """Dispose of the SQLAlchemy engine and close all connections."""
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            logger.info("Disposed SQLAlchemy engine")

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            if self._engine is None:
                self.initialize_engine()

            if self._engine is None:
                return False

            with self._engine.connect() as conn:
                conn.execute(text("SELECT 1"))

            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {e}")
            return False

    def ensure_tables_exist(self) -> None:
        """Ensure all tables defined in ORM metadata exist in the database.

        Raises RuntimeError with a helpful message if any tables are missing.
        """
        if self._engine is None:
            self.initialize_engine()

        if self._engine is None:
            raise RuntimeError(
                "Database engine is not initialized for table inspection"
            )

        inspector = inspect(self._engine)
        try:
            existing_tables = set(inspector.get_table_names())
        except Exception as e:
            logger.error(f"Failed to inspect database tables: {e}")
            raise

        expected_tables = set(Base.metadata.tables.keys())
        missing = expected_tables - existing_tables

        if missing:
            raise RuntimeError("Database is missing required tables: {}. ")

        logger.info("All expected tables present in database")


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_database_manager() -> DatabaseManager:
    """Get the global database manager instance."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def initialize_database() -> DatabaseManager:
    """Initialize and test database connection."""
    db_manager = get_database_manager()

    # Test connection
    if not db_manager.test_connection():
        raise RuntimeError("Failed to connect to database")

    # Ensure required tables exist
    db_manager.ensure_tables_exist()

    logger.info("Database initialized successfully")
    return db_manager
