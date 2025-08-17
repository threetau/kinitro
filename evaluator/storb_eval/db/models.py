from datetime import datetime
from typing import Annotated, Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field

from .schema import EvaluationStatus

# Snowflake-style integer ID (0 .. 2^63 - 1) used for DB BigInteger PKs
SnowflakeId = Annotated[int, Field(ge=0, le=(2**63 - 1))]


class SubmissionMixin(BaseModel):
    """Mixin for submission/provenance data."""

    submission_id: UUID
    miner_hotkey: str
    hf_repo_id: str
    hf_repo_commit: Optional[str] = None
    env_provider: str
    env_name: str


class StatusMixin(BaseModel):
    """Mixin for status data."""

    status: EvaluationStatus


class TimestampMixin(BaseModel):
    """Mixin for timestamp data."""

    created_at: datetime
    updated_at: datetime


class EvaluationJob(SubmissionMixin, StatusMixin, TimestampMixin, BaseModel):
    """Model for evaluation jobs."""

    # Snowflake-style integer ID (maps to BigInteger in DB)
    id: SnowflakeId
    pgqueuer_job_id: str
    container_id: Optional[str] = None
    ray_worker_id: Optional[str] = None
    retry_count: int
    max_retries: int
    last_error: Optional[str] = None
    exit_code: Optional[int] = None
    logs_path: str
    random_seed: Optional[int] = None
    eval_start: Optional[datetime] = None
    eval_end: Optional[datetime] = None


class EvaluationResult:
    """Model for evaluation results."""

    id: UUID
    # FK to EvaluationJob.id (SnowflakeId)
    evaluation_id: SnowflakeId
    avg_return: Optional[float] = None
    success_rate: Optional[float] = None
    num_episodes: Optional[int] = None
    eval_start: Optional[datetime] = None
    eval_end: Optional[datetime] = None
    metrics: Optional[dict[str, Any]] = None


class Episode:
    """Model for episodes."""

    id: UUID
    # FK to EvaluationJob.id (SnowflakeId)
    evaluation_id: SnowflakeId
    episode_index: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    length: Optional[int] = None
    total_reward: Optional[float] = None
    success: bool = False
