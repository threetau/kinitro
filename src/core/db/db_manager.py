import json
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import duckdb
from snowflake import SnowflakeGenerator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import (
    Episode,
    EpisodeStep,
    EvaluationJob,
    EvaluationJobBase,
    EvaluationResult,
    EvaluationResultBase,
    EvaluationStatus,
    SnowflakeId,
)
from .schema.duckdb import setup_duckdb_database
from .schema.pg import (
    EvaluationJob as PGEvaluationJob,
)
from .schema.pg import (
    EvaluationResult as PGEvaluationResult,
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

    def create_evaluation_job(self, job_data: EvaluationJobBase) -> EvaluationJob:
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
        self, result_data: EvaluationResultBase
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

    # DuckDB Operations - Episodes

    def create_episode(self, episode_data: Episode) -> Optional[Episode]:
        """Create a new episode."""
        with self.duck_session() as conn:
            episode_id = next(self.snowflakeGen)

            if episode_id:
                insert_data = episode_data.model_dump()
                insert_data["id"] = episode_id

                # Convert JSON fields to strings
                if insert_data.get("initial_state"):
                    insert_data["initial_state"] = json.dumps(
                        insert_data["initial_state"]
                    )
                if insert_data.get("final_state"):
                    insert_data["final_state"] = json.dumps(insert_data["final_state"])

                # Build INSERT statement
                columns = list(insert_data.keys())
                placeholders = ", ".join(["?" for _ in columns])
                values = [insert_data[col] for col in columns]

                query = f"INSERT INTO episodes ({', '.join(columns)}) VALUES ({placeholders})"
                conn.execute(query, values)

                # Return the created episode
                return self.get_episode(episode_id)

    def get_episode(self, episode_id: int) -> Optional[Episode]:
        """Get an episode by ID."""
        with self.duck_session() as conn:
            result = conn.execute(
                "SELECT * FROM episodes WHERE id = ?", [episode_id]
            ).fetchone()

            if result:
                # Convert result to dict
                columns = [desc[0] for desc in conn.description]  # pyright: ignore[reportOptionalIterable]
                episode_dict = dict(zip(columns, result))

                # Parse JSON fields
                if episode_dict.get("initial_state"):
                    episode_dict["initial_state"] = json.loads(
                        episode_dict["initial_state"]
                    )
                if episode_dict.get("final_state"):
                    episode_dict["final_state"] = json.loads(
                        episode_dict["final_state"]
                    )

                return Episode.model_validate(episode_dict)
            return None

    def update_episode(
        self, episode_id: int, updates: Dict[str, Any]
    ) -> Optional[Episode]:
        """Update an episode."""
        with self.duck_session() as conn:
            if not updates:
                return self.get_episode(episode_id)

            # Convert JSON fields to strings
            processed_updates = {}
            for key, value in updates.items():
                if key in ["initial_state", "final_state"] and value is not None:
                    processed_updates[key] = json.dumps(value)
                else:
                    processed_updates[key] = value

            # Build UPDATE statement
            set_clause = ", ".join([f"{key} = ?" for key in processed_updates.keys()])
            values = list(processed_updates.values()) + [episode_id]

            query = f"UPDATE episodes SET {set_clause} WHERE id = ?"
            conn.execute(query, values)

            return self.get_episode(episode_id)

    def get_episodes_by_eval(self, evaluation_id: int) -> List[Episode]:
        """Get all episodes for an evaluation."""
        with self.duck_session() as conn:
            results = conn.execute(
                "SELECT * FROM episodes WHERE evaluation_id = ? ORDER BY episode_index",
                [evaluation_id],
            ).fetchall()

            episodes = []
            columns = [desc[0] for desc in conn.description]  # pyright: ignore[reportOptionalIterable]

            for result in results:
                episode_dict = dict(zip(columns, result))

                # Parse JSON fields
                if episode_dict.get("initial_state"):
                    episode_dict["initial_state"] = json.loads(
                        episode_dict["initial_state"]
                    )
                if episode_dict.get("final_state"):
                    episode_dict["final_state"] = json.loads(
                        episode_dict["final_state"]
                    )

                episodes.append(Episode.model_validate(episode_dict))

            return episodes

    # DuckDB Operations - Episode Steps

    def create_episode_step(self, step_data: EpisodeStep) -> Optional[EpisodeStep]:
        """Create a new episode step."""
        with self.duck_session() as conn:
            # Generate ID
            step_id = next(self.snowflakeGen)

            if step_id:
                insert_data = step_data.model_dump()
                insert_data["id"] = step_id

                # Convert action JSON to string
                if insert_data.get("action"):
                    insert_data["action"] = json.dumps(insert_data["action"])

                # Build INSERT statement
                columns = list(insert_data.keys())
                placeholders = ", ".join(["?" for _ in columns])
                values = [insert_data[col] for col in columns]

                query = f"INSERT INTO episode_steps ({', '.join(columns)}) VALUES ({placeholders})"
                conn.execute(query, values)

                return self.get_episode_step(step_id)

    def get_episode_step(self, step_id: int) -> Optional[EpisodeStep]:
        """Get an episode step by ID."""
        with self.duck_session() as conn:
            result = conn.execute(
                "SELECT * FROM episode_steps WHERE id = ?", [step_id]
            ).fetchone()

            if result:
                columns = [desc[0] for desc in conn.description]  # pyright: ignore[reportOptionalIterable]
                step_dict = dict(zip(columns, result))

                # Parse action JSON
                if step_dict.get("action"):
                    step_dict["action"] = json.loads(step_dict["action"])

                return EpisodeStep.model_validate(step_dict)
            return None

    def update_episode_step(
        self, step_id: int, updates: Dict[str, Any]
    ) -> Optional[EpisodeStep]:
        """Update an episode step."""
        with self.duck_session() as conn:
            if not updates:
                return self.get_episode_step(step_id)

            # Convert action JSON to string
            processed_updates = {}
            for key, value in updates.items():
                if key == "action" and value is not None:
                    processed_updates[key] = json.dumps(value)
                else:
                    processed_updates[key] = value

            # Build UPDATE statement
            set_clause = ", ".join([f"{key} = ?" for key in processed_updates.keys()])
            values = list(processed_updates.values()) + [step_id]

            query = f"UPDATE episode_steps SET {set_clause} WHERE id = ?"
            conn.execute(query, values)

            return self.get_episode_step(step_id)

    def get_episode_steps_by_episode(self, episode_id: int) -> List[EpisodeStep]:
        """Get all steps for an episode."""
        with self.duck_session() as conn:
            results = conn.execute(
                "SELECT * FROM episode_steps WHERE episode_id = ? ORDER BY step_index",
                [episode_id],
            ).fetchall()

            steps = []
            columns = [desc[0] for desc in conn.description]  # pyright: ignore[reportOptionalIterable]

            for result in results:
                step_dict = dict(zip(columns, result))

                # Parse action JSON
                if step_dict.get("action"):
                    step_dict["action"] = json.loads(step_dict["action"])

                steps.append(EpisodeStep.model_validate(step_dict))

            return steps


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
