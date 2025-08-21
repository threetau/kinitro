"""
Court Clerk component for the kinitro evaluator.

The Court Clerk is responsible for filing evaluation results and managing
all database interactions. It provides a comprehensive CRUD interface for
all database models including evaluation jobs, results, episodes, and steps.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import duckdb
from snowflake import SnowflakeGenerator

from kinitro_eval.roullout.envs import BenchmarkSpec

from . import DatabaseManager, get_database_manager
from .models import (
    Episode as EpisodeModel,
)
from .models import (
    EpisodeStep as EpisodeStepModel,
)
from .models import (
    EvaluationJob as EvaluationJobModel,
)
from .models import (
    EvaluationResult as EvaluationResultModel,
)
from .models import (
    EvaluationStatus,
)
from .schema import DUCKDB_SCHEMA, EvaluationJob, EvaluationResult

logger = logging.getLogger(__name__)


class CourtClerk:
    """
    Court Clerk manages all database interactions for the evaluation system.

    Provides comprehensive CRUD operations for:
    - EvaluationJob: Evaluation job management (PostgreSQL)
    - EvaluationResult: Results storage and retrieval (PostgreSQL)
    - Episode: Episode data management (DuckDB)
    - EpisodeStep: Step-by-step episode data (DuckDB)
    """

    def __init__(
        self,
        db_manager: Optional[DatabaseManager] = None,
        duckdb_path: str = "episodes.duckdb",
    ):
        self.db_manager = db_manager or get_database_manager()
        self.duckdb_path = duckdb_path

        self._init_duckdb()

    def _init_duckdb(self):
        """Initialize DuckDB connection and create schema."""
        try:
            self.duck_conn = duckdb.connect(self.duckdb_path)
            # Create tables and indexes
            self.duck_conn.execute(DUCKDB_SCHEMA)
            logger.info(f"DuckDB initialized at {self.duckdb_path}")
        except Exception as e:
            logger.error(f"Failed to initialize DuckDB: {e}")
            raise

    # The evaluator will never create jobs
    def create_evaluation_job(
        self,
        submission_id: int,
        miner_hotkey: str,
        hf_repo_id: str,
        pgqueuer_job_id: str,
        logs_path: str,
        benchmarks: list[BenchmarkSpec],
        hf_repo_commit: str,
        container_id: str,
        ray_worker_id: Optional[str] = None,
        retry_count: int = 0,
        max_retries: int = 3,
        random_seed: int = 0,
        status: EvaluationStatus = EvaluationStatus.queued,
    ) -> EvaluationJobModel:
        """Create a new evaluation job."""
        try:
            with self.db_manager.get_session() as session:
                # Create the job object using dict constructor
                job_data = {
                    "id": SnowflakeGenerator(42),
                    "submission_id": submission_id,
                    "miner_hotkey": miner_hotkey,
                    "hf_repo_id": hf_repo_id,
                    "hf_repo_commit": hf_repo_commit,
                    "benchmarks": benchmarks,
                    "pgqueuer_job_id": pgqueuer_job_id,
                    "container_id": container_id,
                    "ray_worker_id": ray_worker_id,
                    "retry_count": retry_count,
                    "max_retries": max_retries,
                    "logs_path": logs_path,
                    "random_seed": random_seed,
                    "eval_start": None,
                    "eval_end": None,
                    "status": status,
                }

                # Filter out None values for optional fields
                job_data = {k: v for k, v in job_data.items() if v is not None}

                job_orm = EvaluationJob(**job_data)
                session.add(job_orm)
                session.flush()

                # Convert ORM to Pydantic model
                job_model = self._orm_to_pydantic(job_orm, EvaluationJobModel)
                logger.info(f"Created evaluation job: {job_model.id}")
                return job_model

        except Exception as e:
            logger.error(f"Failed to create evaluation job: {e}")
            raise

    def get_evaluation_job(self, job_id: int) -> Optional[EvaluationJobModel]:
        """Get evaluation job by ID."""
        try:
            with self.db_manager.get_session() as session:
                job_orm = (
                    session.query(EvaluationJob)
                    .filter(EvaluationJob.id == job_id)
                    .first()
                )

                if job_orm:
                    return self._orm_to_pydantic(job_orm, EvaluationJobModel)
                return None

        except Exception as e:
            logger.error(f"Failed to get evaluation job {job_id}: {e}")
            raise

    def update_evaluation_job(
        self, job_id: int, **updates: Any
    ) -> Optional[EvaluationJobModel]:
        """Update evaluation job fields."""
        try:
            with self.db_manager.get_session() as session:
                job_orm = (
                    session.query(EvaluationJob)
                    .filter(EvaluationJob.id == job_id)
                    .first()
                )

                if not job_orm:
                    logger.warning(f"No evaluation job found with ID {job_id}")
                    return None

                # Update fields
                for field, value in updates.items():
                    if hasattr(job_orm, field):
                        setattr(job_orm, field, value)

                session.flush()

                job_model = self._orm_to_pydantic(job_orm, EvaluationJobModel)
                logger.info(f"Updated evaluation job {job_id}")
                return job_model

        except Exception as e:
            logger.error(f"Failed to update evaluation job {job_id}: {e}")
            raise

    def delete_evaluation_job(self, job_id: int) -> bool:
        """Delete evaluation job (cascades to results and episodes)."""
        try:
            with self.db_manager.get_session() as session:
                job_orm = (
                    session.query(EvaluationJob)
                    .filter(EvaluationJob.id == job_id)
                    .first()
                )

                if not job_orm:
                    logger.warning(f"No evaluation job found with ID {job_id}")
                    return False

                session.delete(job_orm)
                logger.info(f"Deleted evaluation job {job_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete evaluation job {job_id}: {e}")
            raise

    def list_evaluation_jobs(
        self,
        submission_id: Optional[int] = None,
        miner_hotkey: Optional[str] = None,
        env_name: Optional[str] = None,
        status: Optional[EvaluationStatus] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[EvaluationJobModel]:
        """List evaluation jobs with optional filtering."""
        try:
            with self.db_manager.get_session() as session:
                query = session.query(EvaluationJob)

                if submission_id:
                    query = query.filter(EvaluationJob.submission_id == submission_id)
                if miner_hotkey:
                    query = query.filter(EvaluationJob.miner_hotkey == miner_hotkey)
                if env_name:
                    query = query.filter(EvaluationJob.env_name == env_name)
                if status:
                    query = query.filter(EvaluationJob.status == status)

                job_orms = (
                    query.order_by(EvaluationJob.created_at.desc())
                    .limit(limit)
                    .offset(offset)
                    .all()
                )

                return [
                    self._orm_to_pydantic(job, EvaluationJobModel) for job in job_orms
                ]

        except Exception as e:
            logger.error(f"Failed to list evaluation jobs: {e}")
            raise

    def create_evaluation_result(
        self,
        evaluation_id: int,
        avg_return: Optional[float] = None,
        success_rate: Optional[float] = None,
        total_reward: Optional[float] = None,
        num_episodes: Optional[int] = None,
    ) -> EvaluationResultModel:
        """Create evaluation result."""
        try:
            with self.db_manager.get_session() as session:
                result_data = {
                    "evaluation_id": evaluation_id,
                    "avg_return": avg_return,
                    "success_rate": success_rate,
                    "total_reward": total_reward,
                    "num_episodes": num_episodes,
                }

                # Filter out None values
                result_data = {k: v for k, v in result_data.items() if v is not None}

                result_orm = EvaluationResult(**result_data)
                session.add(result_orm)
                session.flush()

                result_model = self._orm_to_pydantic(result_orm, EvaluationResultModel)
                logger.info(f"Created evaluation result: {result_model.id}")
                return result_model

        except Exception as e:
            logger.error(f"Failed to create evaluation result: {e}")
            raise

    def get_evaluation_result(self, result_id: int) -> Optional[EvaluationResultModel]:
        """Get evaluation result by ID."""
        try:
            with self.db_manager.get_session() as session:
                result_orm = (
                    session.query(EvaluationResult)
                    .filter(EvaluationResult.id == result_id)
                    .first()
                )

                if result_orm:
                    return self._orm_to_pydantic(result_orm, EvaluationResultModel)
                return None

        except Exception as e:
            logger.error(f"Failed to get evaluation result {result_id}: {e}")
            raise

    def get_evaluation_result_by_job_id(
        self, evaluation_id: int
    ) -> Optional[EvaluationResultModel]:
        """Get evaluation result by evaluation job ID."""
        try:
            with self.db_manager.get_session() as session:
                result_orm = (
                    session.query(EvaluationResult)
                    .filter(EvaluationResult.evaluation_id == evaluation_id)
                    .first()
                )

                if result_orm:
                    return self._orm_to_pydantic(result_orm, EvaluationResultModel)
                return None

        except Exception as e:
            logger.error(
                f"Failed to get evaluation result for job {evaluation_id}: {e}"
            )
            raise

    def update_evaluation_result(
        self, result_id: int, **updates: Any
    ) -> Optional[EvaluationResultModel]:
        """Update evaluation result fields."""
        try:
            with self.db_manager.get_session() as session:
                result_orm = (
                    session.query(EvaluationResult)
                    .filter(EvaluationResult.id == result_id)
                    .first()
                )

                if not result_orm:
                    logger.warning(f"No evaluation result found with ID {result_id}")
                    return None

                # Update fields
                for field, value in updates.items():
                    if hasattr(result_orm, field):
                        setattr(result_orm, field, value)

                session.flush()

                result_model = self._orm_to_pydantic(result_orm, EvaluationResultModel)
                logger.info(f"Updated evaluation result {result_id}")
                return result_model

        except Exception as e:
            logger.error(f"Failed to update evaluation result {result_id}: {e}")
            raise

    def delete_evaluation_result(self, result_id: int) -> bool:
        """Delete evaluation result."""
        try:
            with self.db_manager.get_session() as session:
                result_orm = (
                    session.query(EvaluationResult)
                    .filter(EvaluationResult.id == result_id)
                    .first()
                )

                if not result_orm:
                    logger.warning(f"No evaluation result found with ID {result_id}")
                    return False

                session.delete(result_orm)
                logger.info(f"Deleted evaluation result {result_id}")
                return True

        except Exception as e:
            logger.error(f"Failed to delete evaluation result {result_id}: {e}")
            raise

    def create_episode(
        self,
        evaluation_id: int,
        episode_index: int,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        total_reward: Optional[float] = None,
        success: bool = False,
    ) -> EpisodeModel:
        """Create episode in DuckDB."""

        try:
            episode_id = SnowflakeGenerator(42)
            created_at = datetime.utcnow()
            updated_at = created_at

            # Insert into DuckDB
            self.duck_conn.execute(
                """
                INSERT INTO episodes (
                    id, evaluation_id, episode_index, start_time, end_time, 
                    total_reward, success, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    episode_id,
                    evaluation_id,
                    episode_index,
                    start_time,
                    end_time,
                    total_reward,
                    success,
                    created_at,
                    updated_at,
                ],
            )

            # Create and return model
            episode_model = EpisodeModel(
                id=episode_id,
                evaluation_id=evaluation_id,
                episode_index=episode_index,
                start_time=start_time,
                end_time=end_time,
                total_reward=total_reward,
                success=success,
                created_at=created_at,
                updated_at=updated_at,
            )

            logger.info(f"Created episode: {episode_model.id}")
            return episode_model

        except Exception as e:
            logger.error(f"Failed to create episode: {e}")
            raise

    def get_episode(self, episode_id: int) -> Optional[EpisodeModel]:
        """Get episode by ID from DuckDB."""
        try:
            result = self.duck_conn.execute(
                """
                SELECT id, evaluation_id, episode_index, start_time, end_time,
                       total_reward, success, created_at, updated_at
                FROM episodes WHERE id = ?
            """,
                [episode_id],
            ).fetchone()

            if result:
                return EpisodeModel(
                    id=result[0],
                    evaluation_id=result[1],
                    episode_index=result[2],
                    start_time=result[3],
                    end_time=result[4],
                    total_reward=result[5],
                    success=result[6],
                    created_at=result[7],
                    updated_at=result[8],
                )
            return None

        except Exception as e:
            logger.error(f"Failed to get episode {episode_id}: {e}")
            raise

    def list_episodes_by_evaluation(self, evaluation_id: int) -> List[EpisodeModel]:
        """List all episodes for an evaluation from DuckDB."""
        try:
            results = self.duck_conn.execute(
                """
                SELECT id, evaluation_id, episode_index, start_time, end_time,
                       total_reward, success, created_at, updated_at
                FROM episodes 
                WHERE evaluation_id = ?
                ORDER BY episode_index
            """,
                [evaluation_id],
            ).fetchall()

            episodes = []
            for result in results:
                episodes.append(
                    EpisodeModel(
                        id=result[0],
                        evaluation_id=result[1],
                        episode_index=result[2],
                        start_time=result[3],
                        end_time=result[4],
                        total_reward=result[5],
                        success=result[6],
                        created_at=result[7],
                        updated_at=result[8],
                    )
                )

            return episodes

        except Exception as e:
            logger.error(f"Failed to list episodes for evaluation {evaluation_id}: {e}")
            raise

    def update_episode(self, episode_id: int, **updates: Any) -> Optional[EpisodeModel]:
        """Update episode fields in DuckDB."""

        try:
            # Check if episode exists
            existing = self.get_episode(episode_id)
            if not existing:
                logger.warning(f"No episode found with ID {episode_id}")
                return None

            # Build update query
            set_clauses = []
            values = []

            valid_fields = {
                "episode_index",
                "start_time",
                "end_time",
                "total_reward",
                "success",
            }

            for field, value in updates.items():
                if field in valid_fields:
                    set_clauses.append(f"{field} = ?")
                    values.append(value)

            if not set_clauses:
                logger.warning("No valid fields to update")
                return existing

            # Add updated_at
            set_clauses.append("updated_at = ?")
            values.append(datetime.utcnow())
            values.append(episode_id)

            query = f"UPDATE episodes SET {', '.join(set_clauses)} WHERE id = ?"
            self.duck_conn.execute(query, values)

            # Return updated episode
            updated_episode = self.get_episode(episode_id)
            logger.info(f"Updated episode {episode_id}")
            return updated_episode

        except Exception as e:
            logger.error(f"Failed to update episode {episode_id}: {e}")
            raise

    def delete_episode(self, episode_id: int) -> bool:
        """Delete episode (cascades to steps) from DuckDB."""
        try:
            # Check if episode exists first
            existing = self.get_episode(episode_id)
            if not existing:
                logger.warning(f"No episode found with ID {episode_id}")
                return False

            # First delete all steps for this episode
            self.duck_conn.execute(
                "DELETE FROM episode_steps WHERE episode_id = ?", [episode_id]
            )

            # Then delete the episode
            self.duck_conn.execute("DELETE FROM episodes WHERE id = ?", [episode_id])

            logger.info(f"Deleted episode {episode_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete episode {episode_id}: {e}")
            raise

    def create_episode_step(
        self,
        episode_id: int,
        step_index: int,
        observation_path: Dict[str, Any],
        reward: float,
        action: Dict[str, Any],
    ) -> EpisodeStepModel:
        """Create episode step in DuckDB."""

        try:
            step_id = SnowflakeGenerator(42)
            created_at = datetime.utcnow()
            updated_at = created_at

            # Convert dictionaries to JSON strings for DuckDB
            observation_json = json.dumps(observation_path)
            action_json = json.dumps(action)

            # Insert into DuckDB
            self.duck_conn.execute(
                """
                INSERT INTO episode_steps (
                    id, episode_id, step_index, observation_path, reward, action,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
                [
                    step_id,
                    episode_id,
                    step_index,
                    observation_json,
                    reward,
                    action_json,
                    created_at,
                    updated_at,
                ],
            )

            # Create and return model
            step_model = EpisodeStepModel(
                id=step_id,
                episode_id=episode_id,
                step_index=step_index,
                observation_path=observation_path,
                reward=reward,
                action=action,
                created_at=created_at,
                updated_at=updated_at,
            )

            logger.info(f"Created episode step: {step_model.id}")
            return step_model

        except Exception as e:
            logger.error(f"Failed to create episode step: {e}")
            raise

    def get_episode_step(self, step_id: int) -> Optional[EpisodeStepModel]:
        """Get episode step by ID from DuckDB."""

        try:
            result = self.duck_conn.execute(
                """
                SELECT id, episode_id, step_index, observation_path, reward, action,
                       created_at, updated_at
                FROM episode_steps WHERE id = ?
            """,
                [step_id],
            ).fetchone()

            if result:
                # Parse JSON fields back to dictionaries
                observation_path = json.loads(result[3]) if result[3] else {}
                action = json.loads(result[5]) if result[5] else {}

                return EpisodeStepModel(
                    id=result[0],
                    episode_id=result[1],
                    step_index=result[2],
                    observation_path=observation_path,
                    reward=result[4],
                    action=action,
                    created_at=result[6],
                    updated_at=result[7],
                )
            return None

        except Exception as e:
            logger.error(f"Failed to get episode step {step_id}: {e}")
            raise

    def list_episode_steps(self, episode_id: int) -> List[EpisodeStepModel]:
        """List all steps for an episode from DuckDB."""

        try:
            results = self.duck_conn.execute(
                """
                SELECT id, episode_id, step_index, observation_path, reward, action,
                       created_at, updated_at
                FROM episode_steps
                WHERE episode_id = ?
                ORDER BY step_index
            """,
                [episode_id],
            ).fetchall()

            steps = []
            for result in results:
                # Parse JSON fields back to dictionaries
                observation_path = json.loads(result[3]) if result[3] else {}
                action = json.loads(result[5]) if result[5] else {}

                steps.append(
                    EpisodeStepModel(
                        id=result[0],
                        episode_id=result[1],
                        step_index=result[2],
                        observation_path=observation_path,
                        reward=result[4],
                        action=action,
                        created_at=result[6],
                        updated_at=result[7],
                    )
                )

            return steps

        except Exception as e:
            logger.error(f"Failed to list steps for episode {episode_id}: {e}")
            raise

    def update_episode_step(
        self, step_id: int, **updates: Any
    ) -> Optional[EpisodeStepModel]:
        """Update episode step fields in DuckDB."""

        try:
            # Check if step exists
            existing = self.get_episode_step(step_id)
            if not existing:
                logger.warning(f"No episode step found with ID {step_id}")
                return None

            # Build update query
            set_clauses = []
            values = []

            valid_fields = {"step_index", "observation_path", "reward", "action"}

            for field, value in updates.items():
                if field in valid_fields:
                    if field in ["observation_path", "action"]:
                        # Convert dict to JSON string
                        json_value = json.dumps(value)
                        set_clauses.append(f"{field} = ?")
                        values.append(json_value)
                    else:
                        set_clauses.append(f"{field} = ?")
                        values.append(value)

            if not set_clauses:
                logger.warning("No valid fields to update")
                return existing

            # Add updated_at
            set_clauses.append("updated_at = ?")
            values.append(datetime.utcnow())
            values.append(step_id)  # For WHERE clause

            query = f"UPDATE episode_steps SET {', '.join(set_clauses)} WHERE id = ?"
            self.duck_conn.execute(query, values)

            # Return updated step
            updated_step = self.get_episode_step(step_id)
            logger.info(f"Updated episode step {step_id}")
            return updated_step

        except Exception as e:
            logger.error(f"Failed to update episode step {step_id}: {e}")
            raise

    def delete_episode_step(self, step_id: int) -> bool:
        """Delete episode step from DuckDB."""

        try:
            # Check if step exists first
            existing = self.get_episode_step(step_id)
            if not existing:
                logger.warning(f"No episode step found with ID {step_id}")
                return False

            self.duck_conn.execute("DELETE FROM episode_steps WHERE id = ?", [step_id])
            logger.info(f"Deleted episode step {step_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to delete episode step {step_id}: {e}")
            raise

    def _orm_to_pydantic(self, orm_obj: Any, pydantic_class: type) -> Any:
        """Convert SQLAlchemy ORM object to Pydantic model."""
        # Get all field names from the Pydantic model
        field_data = {}
        for field_name in pydantic_class.model_fields.keys():
            if hasattr(orm_obj, field_name):
                field_data[field_name] = getattr(orm_obj, field_name)

        return pydantic_class(**field_data)

    def complete_evaluation_job(
        self,
        job_id: int,
        avg_return: Optional[float] = None,
        success_rate: Optional[float] = None,
        total_reward: Optional[float] = None,
        num_episodes: Optional[int] = None,
    ) -> tuple[EvaluationJobModel, EvaluationResultModel]:
        """Mark evaluation as completed and create results."""
        try:
            # Update job status to completed
            job_model = self.update_evaluation_job(
                job_id,
                status=EvaluationStatus.completed,
                eval_end=datetime.utcnow(),
            )

            if not job_model:
                raise ValueError(f"Evaluation job {job_id} not found")

            # Create evaluation result
            result_model = self.create_evaluation_result(
                evaluation_id=job_id,
                avg_return=avg_return,
                success_rate=success_rate,
                total_reward=total_reward,
                num_episodes=num_episodes,
            )

            logger.info(f"Completed evaluation job {job_id}")
            return job_model, result_model

        except Exception as e:
            logger.error(f"Failed to complete evaluation job {job_id}: {e}")
            raise

    def fail_evaluation_job(
        self, job_id: int, error_message: str
    ) -> Optional[EvaluationJobModel]:
        """Mark evaluation job as failed with error message."""
        return self.update_evaluation_job(
            job_id,
            status=EvaluationStatus.failed,
            eval_end=datetime.utcnow(),
        )

    def get_evaluation_summary(self, job_id: int) -> Dict[str, Any]:
        """Get comprehensive summary of an evaluation job."""
        try:
            job = self.get_evaluation_job(job_id)
            if not job:
                return {}

            result = self.get_evaluation_result_by_job_id(job_id)
            episodes = self.list_episodes_by_evaluation(job_id)

            return {
                "job": job.model_dump(),
                "result": result.model_dump() if result else None,
                "episodes": [ep.model_dump() for ep in episodes],
                "episode_count": len(episodes),
                "success_count": sum(1 for ep in episodes if ep.success),
            }

        except Exception as e:
            logger.error(f"Failed to get evaluation summary for {job_id}: {e}")
            raise
