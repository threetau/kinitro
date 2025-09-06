import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import duckdb
from snowflake import SnowflakeGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import (
    EvaluationJob,
    EvaluationResult,
    EvaluationStatus,
    SnowflakeId,
)
from .models import (
    EvaluationJob as PGEvaluationJob,
)
from .models import (
    EvaluationResult as PGEvaluationResult,
)
from .schema.duckdb import (
    # Episode,
    # EpisodeStep,
    setup_duckdb_database,
)


class DatabaseManager:
    """Manages connections and operations for both PostgreSQL and DuckDB."""

    def __init__(
        self,
        postgres_url: str,
        duckdb_path: str = "evaluation_data.duckdb",
        echo: bool = False,
    ):
        """Initialize database connections."""
        self.postgres_url = postgres_url
        self.duckdb_path = duckdb_path
        self.echo = echo

        self.snowflakeGen = SnowflakeGenerator(42)

        # PostgreSQL setup
        self.pg_engine = create_engine(postgres_url, echo=echo)
        self.pg_session_factory = sessionmaker(bind=self.pg_engine)

        # DuckDB setup
        self.duck_conn: Optional[duckdb.DuckDBPyConnection] = None

    def initialize_databases(self) -> None:
        """Initialize both databases with their schemas."""
        # Initialize DuckDB
        self.duck_conn = setup_duckdb_database(self.duckdb_path)

    @contextmanager
    def pg_session(self):
        """Context manager for PostgreSQL sessions."""
        session = self.pg_session_factory()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    @contextmanager
    def duck_session(self):
        """Context manager for DuckDB sessions."""
        if self.duck_conn is None:
            self.duck_conn = setup_duckdb_database(self.duckdb_path)
        try:
            yield self.duck_conn
        except Exception:
            # DuckDB auto-rollbacks on exceptions
            raise

    def close_connections(self) -> None:
        """Close all database connections."""
        if self.duck_conn:
            self.duck_conn.close()
            self.duck_conn = None
        self.pg_engine.dispose()

    # PostgreSQL Operations - Evaluation Jobs

    def create_evaluation_job(self, job_data: EvaluationJob) -> EvaluationJob:
        """Create a new evaluation job."""
        with self.pg_session() as session:
            pg_job = PGEvaluationJob(**job_data.model_dump())
            session.add(pg_job)
            session.flush()  # Get ID
            session.refresh(pg_job)
            return EvaluationJob.model_validate(pg_job)

    def get_evaluation_job(self, job_id: SnowflakeId) -> Optional[EvaluationJob]:
        """Get an evaluation job by ID."""
        with self.pg_session() as session:
            pg_job = (
                session.query(PGEvaluationJob)
                .filter(PGEvaluationJob.id == job_id)
                .first()
            )
            if pg_job:
                return EvaluationJob.model_validate(pg_job)
            return None

    def update_evaluation_job(
        self, job_id: SnowflakeId, updates: Dict[str, Any]
    ) -> Optional[EvaluationJob]:
        """Update an evaluation job."""
        with self.pg_session() as session:
            pg_job = (
                session.query(PGEvaluationJob)
                .filter(PGEvaluationJob.id == job_id)
                .first()
            )

            if not pg_job:
                return None

            # Update fields
            for key, value in updates.items():
                if hasattr(pg_job, key):
                    setattr(pg_job, key, value)

            session.flush()
            session.refresh(pg_job)
            return EvaluationJob.model_validate(pg_job)

    def get_evaluation_jobs_by_status(
        self, status: EvaluationStatus
    ) -> List[EvaluationJob]:
        """Get all evaluation jobs by status."""
        with self.pg_session() as session:
            pg_jobs = (
                session.query(PGEvaluationJob)
                .filter(PGEvaluationJob.status == status)
                .all()
            )
            return [EvaluationJob.model_validate(job) for job in pg_jobs]

    def get_evaluation_jobs_by_miner(self, miner_hotkey: str) -> List[EvaluationJob]:
        """Get all evaluation jobs for a specific miner."""
        with self.pg_session() as session:
            pg_jobs = (
                session.query(PGEvaluationJob)
                .filter(PGEvaluationJob.miner_hotkey == miner_hotkey)
                .order_by(PGEvaluationJob.created_at.desc())
                .all()
            )
            return [EvaluationJob.model_validate(job) for job in pg_jobs]

    def create_evaluation_result(
        self, result_data: EvaluationResult
    ) -> EvaluationResult:
        """Create a new evaluation result."""
        with self.pg_session() as session:
            pg_result = PGEvaluationResult(**result_data.model_dump())
            session.add(pg_result)
            session.flush()
            session.refresh(pg_result)
            return EvaluationResult.model_validate(pg_result)

    def get_evaluation_result(self, evaluation_id: int) -> Optional[EvaluationResult]:
        """Get evaluation result by evaluation ID."""
        with self.pg_session() as session:
            pg_result = (
                session.query(PGEvaluationResult)
                .filter(PGEvaluationResult.evaluation_id == evaluation_id)
                .first()
            )
            if pg_result:
                return EvaluationResult.model_validate(pg_result)
            return None

    def update_evaluation_result(
        self, evaluation_id: int, updates: Dict[str, Any]
    ) -> Optional[EvaluationResult]:
        """Update an evaluation result."""
        with self.pg_session() as session:
            pg_result = (
                session.query(PGEvaluationResult)
                .filter(PGEvaluationResult.evaluation_id == evaluation_id)
                .first()
            )

            if not pg_result:
                return None

            # Update fields
            for key, value in updates.items():
                if hasattr(pg_result, key):
                    setattr(pg_result, key, value)

            session.flush()
            session.refresh(pg_result)
            return EvaluationResult.model_validate(pg_result)


# TODO: bring back duckdb episode logging functions


# Factory function for easy setup
def create_database_manager(
    postgres_url: str,
    duckdb_path: str = "evaluation_data.duckdb",
    echo: bool = False,
    initialize: bool = True,
) -> DatabaseManager:
    """Create and optionally initialize a database manager."""
    manager = DatabaseManager(postgres_url, duckdb_path, echo)

    if initialize:
        manager.initialize_databases()

    return manager
