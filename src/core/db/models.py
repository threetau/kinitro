from datetime import datetime
from enum import Enum
from typing import Annotated, Any, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field

# Snowflake-style integer ID (0 .. 2^63 - 1) used for DB BigInteger PKs
SnowflakeId = Annotated[int, Field(ge=0, le=(2**63 - 1))]


class EvaluationStatus(str, Enum):
    """Evaluation job status enum."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


# Base Models
class TimestampMixin(BaseModel):
    """Mixin for created/updated timestamps."""

    created_at: datetime
    updated_at: Optional[datetime] = None


class SubmissionMixin(BaseModel):
    """Mixin for submission/provenance metadata."""

    submission_id: int
    miner_hotkey: str = Field(..., max_length=48)
    hf_repo_id: str = Field(..., max_length=256)
    hf_repo_commit: Optional[str] = Field(None, max_length=40)
    env_provider: str = Field(..., max_length=64)
    env_name: str = Field(..., max_length=128)


class StatusMixin(BaseModel):
    """Mixin for status data."""

    status: EvaluationStatus


class EvaluationJobBase(BaseModel):
    """Base model for evaluation job data."""

    status: EvaluationStatus = EvaluationStatus.QUEUED
    container_id: Optional[str] = Field(None, max_length=128)
    ray_worker_id: Optional[str] = Field(None, max_length=128)
    max_retries: int = Field(default=3, ge=0)
    retry_count: int = Field(default=0, ge=0)
    random_seed: Optional[int] = None
    eval_start: Optional[datetime] = None
    eval_end: Optional[datetime] = None
    logs_path: str
    max_memory_mb: Optional[int] = Field(None, gt=0)
    max_cpu_percent: Optional[float] = Field(None, ge=0, le=100)

    model_config = ConfigDict(validate_assignment=True, use_enum_values=True)

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate evaluation duration."""
        if self.eval_start is not None and self.eval_end is not None:
            return (self.eval_end - self.eval_start).total_seconds()
        return None


class EvaluationJob(EvaluationJobBase, SubmissionMixin, TimestampMixin):
    """Full evaluation job model."""

    id: int

    model_config = ConfigDict(from_attributes=True)

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "EvaluationJob":
        return cls.model_validate_json(data)


class EvaluationResultBase(BaseModel):
    """Base model for evaluation results."""

    total_episodes: int = Field(..., gt=0)
    successful_episodes: int = Field(default=0, ge=0)
    failed_episodes: int = Field(default=0, ge=0)
    success_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    total_reward: Optional[float] = None
    avg_reward: Optional[float] = None
    min_reward: Optional[float] = None
    max_reward: Optional[float] = None
    std_reward: Optional[float] = None
    median_reward: Optional[float] = None
    total_steps: Optional[int] = Field(None, ge=0)
    avg_steps_per_episode: Optional[float] = Field(None, ge=0)
    max_steps_per_episode: Optional[int] = None
    avg_episode_duration_seconds: Optional[float] = None
    total_computation_time_seconds: Optional[float] = None
    computed_at: datetime = Field(default_factory=datetime.utcnow)


class EvaluationResult(EvaluationResultBase, TimestampMixin):
    """Full evaluation result model."""

    id: int
    evaluation_id: int

    model_config = ConfigDict(from_attributes=True)


# DuckDB Models
class EpisodeBase(BaseModel):
    """Base model for episodes."""

    evaluation_id: int
    episode_index: int
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_reward: Optional[float] = None
    success: bool = False
    num_steps: int = 0
    initial_state: Optional[dict[str, Any]] = None
    final_state: Optional[dict[str, Any]] = None
    memory_peak_mb: Optional[int] = None

    @computed_field
    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate episode duration."""
        if self.start_time is not None and self.end_time is not None:
            return (self.end_time - self.start_time).total_seconds()
        return None


class Episode(EpisodeBase):
    """Full episode model."""

    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class EpisodeStepBase(BaseModel):
    """Base model for episode steps."""

    episode_id: int
    step_index: int
    reward: float
    action: dict[str, Any]
    observation: str
    step_timestamp: Optional[datetime] = None
    duration: Optional[float] = None


class EpisodeStep(EpisodeStepBase):
    """Full episode step model."""

    id: int
    created_at: datetime
    partition_date: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)
