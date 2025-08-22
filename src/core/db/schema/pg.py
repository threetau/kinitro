import enum
from typing import Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import declarative_base
from sqlalchemy.sql import func

Base = declarative_base()

SnowflakeId = BigInteger
SS58Address = String(48)


class EvaluationStatus(enum.Enum):
    """Evaluation job status enum."""

    QUEUED = "queued"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TimestampMixin:
    """Mixin for created/updated timestamps."""

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), index=True
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class SubmissionMixin:
    """Mixin for submission/provenance metadata."""

    submission_id = Column(SnowflakeId, nullable=False, index=True)
    miner_hotkey = Column(SS58Address, nullable=False, index=True)
    hf_repo_id = Column(String(256), nullable=False, index=True)
    hf_repo_commit = Column(String(40), nullable=True, index=True)  # Git SHA
    env_provider = Column(String(64), nullable=False)
    env_name = Column(String(128), nullable=False, index=True)


class EvaluationJob(SubmissionMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_jobs"

    id = Column(SnowflakeId, primary_key=True)

    status = Column(
        SAEnum(EvaluationStatus, name="evaluation_status", native_enum=False),
        nullable=False,
        server_default=EvaluationStatus.QUEUED.value,
        index=True,
    )

    container_id = Column(String(128), nullable=True, index=True)
    ray_worker_id = Column(String(128), nullable=True, index=True)

    max_retries = Column(Integer, nullable=False, server_default="3")
    retry_count = Column(Integer, nullable=False, server_default="0")
    random_seed = Column(Integer, nullable=True)

    eval_start = Column(DateTime(timezone=True), nullable=True, index=True)
    eval_end = Column(DateTime(timezone=True), nullable=True, index=True)

    logs_path = Column(Text, nullable=False)

    max_memory_mb = Column(Integer, nullable=True)
    max_cpu_percent = Column(Float, nullable=True)

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate evaluation duration."""
        if self.eval_start is not None and self.eval_end is not None:
            return (self.eval_end - self.eval_start).total_seconds()
        return None

    __table_args__ = (
        # Business logic constraints
        CheckConstraint("retry_count >= 0", name="ck_retry_count_non_negative"),
        CheckConstraint("max_retries >= 0", name="ck_max_retries_non_negative"),
        CheckConstraint("retry_count <= max_retries", name="ck_retry_within_max"),
        CheckConstraint(
            "eval_end IS NULL OR eval_start IS NULL OR eval_end >= eval_start",
            name="ck_eval_times_ordered",
        ),
        CheckConstraint(
            "max_memory_mb IS NULL OR max_memory_mb > 0", name="ck_memory_positive"
        ),
        CheckConstraint(
            "max_cpu_percent IS NULL OR (max_cpu_percent >= 0 AND max_cpu_percent <= 100)",
            name="ck_cpu_percent_range",
        ),
        Index("ix_eval_jobs_status_created", "status", "created_at"),
        Index("ix_eval_jobs_miner_env", "miner_hotkey", "env_name"),
        Index(
            "ix_eval_jobs_active",
            "status",
            "eval_start",
            postgresql_where=Column("status").in_(["starting", "running"]),
        ),
        Index(
            "ix_eval_jobs_recent_completed",
            "status",
            "eval_end",
            postgresql_where=Column("status") == "completed",
        ),
        # Unique constraint to prevent duplicate submissions
        UniqueConstraint(
            "submission_id", "miner_hotkey", "env_name", name="uq_submission_miner_env"
        ),
    )


class EvaluationResult(TimestampMixin, Base):
    __tablename__ = "evaluation_results"

    id = Column(SnowflakeId, primary_key=True)

    evaluation_id = Column(SnowflakeId, nullable=False, unique=True, index=True)

    total_episodes = Column(Integer, nullable=False)
    successful_episodes = Column(Integer, nullable=False, server_default="0")
    failed_episodes = Column(Integer, nullable=False, server_default="0")

    success_rate = Column(Float, nullable=False, server_default="0.0")

    total_reward = Column(Float, nullable=True)
    avg_reward = Column(Float, nullable=True)
    min_reward = Column(Float, nullable=True)
    max_reward = Column(Float, nullable=True)
    std_reward = Column(Float, nullable=True)
    median_reward = Column(Float, nullable=True)

    total_steps = Column(Integer, nullable=True)
    avg_steps_per_episode = Column(Float, nullable=True)
    max_steps_per_episode = Column(Integer, nullable=True)

    avg_episode_duration_seconds = Column(Float, nullable=True)
    total_computation_time_seconds = Column(Float, nullable=True)

    computed_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        # Data quality constraints
        CheckConstraint("total_episodes > 0", name="ck_total_episodes_positive"),
        CheckConstraint(
            "successful_episodes >= 0", name="ck_successful_episodes_non_negative"
        ),
        CheckConstraint("failed_episodes >= 0", name="ck_failed_episodes_non_negative"),
        CheckConstraint(
            "successful_episodes + failed_episodes <= total_episodes",
            name="ck_episode_counts_consistent",
        ),
        CheckConstraint(
            "success_rate >= 0.0 AND success_rate <= 1.0", name="ck_success_rate_range"
        ),
        CheckConstraint(
            "total_steps IS NULL OR total_steps >= 0",
            name="ck_total_steps_non_negative",
        ),
        CheckConstraint(
            "avg_steps_per_episode IS NULL OR avg_steps_per_episode >= 0",
            name="ck_avg_steps_non_negative",
        ),
        # Performance indexes
        Index("ix_eval_results_success_rate", "success_rate"),
        Index("ix_eval_results_avg_reward", "avg_reward"),
        Index("ix_eval_results_computed_at", "computed_at"),
    )
