from contextlib import contextmanager
from typing import Any, Dict, List, Optional

from snowflake import SnowflakeGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from core.db.models import SnowflakeId

from .models import (
    EvaluationJob,
    EvaluationResult,
    EvaluationStatus,
)
from .models import (
    EvaluationJob as PGEvaluationJob,
)
from .models import (
    EvaluationResult as PGEvaluationResult,
)


class DatabaseManager:
    """Manages connections and operations for PostgreSQL."""

    def __init__(
        self,
        postgres_url: str,
        echo: bool = False,
    ):
        """Initialize database connections."""
        self.postgres_url = postgres_url
        self.echo = echo

        self.snowflakeGen = SnowflakeGenerator(42)

        # PostgreSQL setup
        self.pg_engine = create_engine(postgres_url, echo=echo)
        self.pg_session_factory = sessionmaker(bind=self.pg_engine)

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

    def close_connections(self) -> None:
        """Close all database connections."""
        self.pg_engine.dispose()

    # PostgreSQL Operations - Evaluation Jobs

    def create_evaluation_job(self, pg_job: EvaluationJob) -> None:
        """Create a new evaluation job."""
        with self.pg_session() as session:
            session.add(pg_job)
            session.flush()  # Get ID
            session.refresh(pg_job)

    def get_evaluation_job(self, job_id: SnowflakeId) -> Optional[EvaluationJob]:
        """Get an evaluation job by ID."""
        with self.pg_session() as session:
            pg_job = (
                session.query(PGEvaluationJob)
                .filter(PGEvaluationJob.id == job_id)
                .first()
            )
            return pg_job

    def update_evaluation_job(
        self, job_id: SnowflakeId, updates: Dict[str, Any]
    ) -> None:
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
