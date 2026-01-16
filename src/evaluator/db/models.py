"""
SQLModel models for validator database.

These models are used for validator-specific data storage including
evaluation jobs, results, and local state management.
"""

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from sqlalchemy import BigInteger, ForeignKey, Integer
from sqlalchemy import DateTime as SADateTime
from sqlalchemy import String as SAString
from sqlalchemy import Text as SAText
from sqlmodel import JSON, Column, Field, Relationship, SQLModel

from core.db.models import EvaluationStatus, TimestampMixin


class EvaluationJob(TimestampMixin, SQLModel, table=True):
    """Evaluation jobs received from backend."""

    __tablename__ = "validator_evaluation_jobs"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))

    # Submission metadata
    submission_id: int = Field(sa_column=Column(BigInteger, nullable=False, index=True))
    competition_id: str = Field(max_length=128, nullable=False, index=True)
    miner_hotkey: str = Field(max_length=48, nullable=False, index=True)
    hf_repo_id: str = Field(max_length=256, nullable=False)
    env_provider: str = Field(max_length=128, nullable=False)
    benchmark_name: str = Field(max_length=128, nullable=False)
    config: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))
    timeout_seconds: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    artifact_url: Optional[str] = Field(
        default=None, max_length=512, sa_column=Column(SAString(512), nullable=True)
    )
    artifact_expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )
    artifact_sha256: Optional[str] = Field(
        default=None, max_length=64, sa_column=Column(SAString(64), nullable=True)
    )
    artifact_size_bytes: Optional[int] = Field(default=None, nullable=True)

    # Job execution
    status: EvaluationStatus = Field(
        default=EvaluationStatus.QUEUED, nullable=False, index=True
    )
    received_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )
    started_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime, nullable=True)
    )
    completed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime, nullable=True)
    )

    # Resource limits
    max_memory_mb: Optional[int] = Field(default=None)
    max_cpu_percent: Optional[float] = Field(default=None)
    max_retries: int = Field(default=3, nullable=False)
    retry_count: int = Field(default=0, nullable=False)

    # Container/execution info
    container_id: Optional[str] = Field(default=None, max_length=128)
    error_message: Optional[str] = Field(
        default=None, sa_column=Column(SAText, nullable=True)
    )

    # Random seed for reproducibility
    random_seed: Optional[int] = Field(default=None)

    # Relationships
    results: List["EvaluationResult"] = Relationship(
        back_populates="job", cascade_delete=True
    )


class EvaluationResult(TimestampMixin, SQLModel, table=True):
    """Results of evaluation jobs."""

    __tablename__ = "validator_evaluation_results"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    job_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("validator_evaluation_jobs.id"),
            nullable=False,
            index=True,
        )
    )

    # Result metadata
    benchmark: str = Field(max_length=128, nullable=False, index=True)
    validator_hotkey: str = Field(max_length=48, nullable=False, index=True)
    miner_hotkey: str = Field(max_length=48, nullable=False, index=True)
    competition_id: str = Field(max_length=128, nullable=False, index=True)

    # Evaluation metrics
    score: float = Field(nullable=False)
    success_rate: Optional[float] = Field(default=None)
    avg_reward: Optional[float] = Field(default=None)
    total_episodes: Optional[int] = Field(default=None)

    # Execution info
    result_time: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc), nullable=False
    )
    computation_time_seconds: Optional[float] = Field(default=None)

    # Logs and metadata
    logs: Optional[str] = Field(default=None, sa_column=Column(SAText, nullable=True))
    error: Optional[str] = Field(default=None, sa_column=Column(SAText, nullable=True))
    extra_data: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )  # Additional benchmark-specific data
    env_specs: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    # Submission status
    submitted_to_backend: bool = Field(default=False, nullable=False)
    submission_error: Optional[str] = Field(
        default=None, sa_column=Column(SAText, nullable=True)
    )

    # Relationships
    job: Optional["EvaluationJob"] = Relationship(back_populates="results")


class ValidatorState(TimestampMixin, SQLModel, table=True):
    """Local validator state information."""

    __tablename__ = "validator_state"

    id: int = Field(primary_key=True)
    validator_hotkey: str = Field(
        max_length=48, nullable=False, unique=True, index=True
    )

    # Backend connection state
    backend_url: Optional[str] = Field(default=None, max_length=512)
    connected_to_backend: bool = Field(default=False, nullable=False)
    last_backend_connection: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime, nullable=True)
    )
    last_heartbeat_sent: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime, nullable=True)
    )
    last_heartbeat_ack: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime, nullable=True)
    )

    # Job processing stats
    total_jobs_received: int = Field(default=0, nullable=False)
    total_jobs_completed: int = Field(default=0, nullable=False)
    total_jobs_failed: int = Field(default=0, nullable=False)

    # Version and config
    validator_version: Optional[str] = Field(default=None, max_length=32)
    config_hash: Optional[str] = Field(
        default=None, max_length=64
    )  # Hash of current configuration

    # Performance metrics
    avg_job_duration_seconds: Optional[float] = Field(default=None)
    last_performance_update: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime, nullable=True)
    )
