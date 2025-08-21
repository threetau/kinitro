from datetime import datetime
from typing import Annotated, Optional

from pydantic import BaseModel, Field

from .schema import EvaluationStatus

# Snowflake-style integer ID (0 .. 2^63 - 1) used for DB BigInteger PKs
SnowflakeId = Annotated[int, Field(ge=0, le=(2**63 - 1))]


class SubmissionMixin(BaseModel):
    """Mixin for submission/provenance data."""

    submission_id: SnowflakeId
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
    logs_path: str
    random_seed: Optional[int] = None
    eval_start: Optional[datetime] = None
    eval_end: Optional[datetime] = None


class EvaluationResult(TimestampMixin, BaseModel):
    """Model for evaluation results."""

    id: SnowflakeId
    # FK to EvaluationJob.id (SnowflakeId)
    evaluation_id: SnowflakeId
    avg_return: Optional[float] = None
    success_rate: Optional[float] = None
    total_reward: Optional[float] = None
    num_episodes: Optional[int] = None


class Episode(TimestampMixin, BaseModel):
    """Model for episodes."""

    id: SnowflakeId
    # FK to EvaluationJob.id (SnowflakeId)
    evaluation_id: SnowflakeId
    episode_index: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_reward: Optional[float] = None
    success: bool = False


class EpisodeStep(TimestampMixin, BaseModel):
    """Model for episode steps."""

    id: SnowflakeId
    # FK to Episode.id (SnowflakeId)
    episode_id: SnowflakeId
    step_index: int
    observation_path: dict  # JSON field
    reward: float
    action: dict  # JSON field
