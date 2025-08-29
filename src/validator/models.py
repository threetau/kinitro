"""
SQLAlchemy models for validator database.

These models are used for validator-specific data storage including
evaluation jobs, results, and local state management.
"""

from datetime import datetime, timezone
from typing import List

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from core.db.models import EvaluationStatus

# Base class for all validator models
Base = declarative_base()


class TimestampMixin:
    """Mixin for created/updated timestamps."""

    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class ValidatorEvaluationJob(TimestampMixin, Base):
    """Evaluation jobs received from backend."""

    __tablename__ = "validator_evaluation_jobs"

    id = Column(BigInteger, primary_key=True)
    job_id = Column(String(128), nullable=False, unique=True, index=True)

    # Submission metadata
    competition_id = Column(String(128), nullable=False, index=True)
    miner_hotkey = Column(String(48), nullable=False, index=True)
    hf_repo_id = Column(String(256), nullable=False)
    benchmarks = Column(JSON, nullable=False)  # List of benchmark names

    # Job execution
    status = Column(
        Enum(EvaluationStatus),
        default=EvaluationStatus.QUEUED,
        nullable=False,
        index=True,
    )
    received_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Resource limits
    max_memory_mb = Column(Integer, nullable=True)
    max_cpu_percent = Column(Float, nullable=True)
    max_retries = Column(Integer, default=3, nullable=False)
    retry_count = Column(Integer, default=0, nullable=False)

    # Container/execution info
    container_id = Column(String(128), nullable=True)
    logs_path = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)

    # Random seed for reproducibility
    random_seed = Column(Integer, nullable=True)

    # Relationships
    results = relationship(
        "ValidatorEvaluationResult", back_populates="job", cascade="all, delete-orphan"
    )


class ValidatorEvaluationResult(TimestampMixin, Base):
    """Results of evaluation jobs."""

    __tablename__ = "validator_evaluation_results"

    id = Column(BigInteger, primary_key=True)
    job_id = Column(
        String(128),
        ForeignKey("validator_evaluation_jobs.job_id"),
        nullable=False,
        index=True,
    )

    # Result metadata
    benchmark = Column(String(128), nullable=False, index=True)
    validator_hotkey = Column(String(48), nullable=False, index=True)
    miner_hotkey = Column(String(48), nullable=False, index=True)
    competition_id = Column(String(128), nullable=False, index=True)

    # Evaluation metrics
    score = Column(Float, nullable=False)
    success_rate = Column(Float, nullable=True)
    avg_reward = Column(Float, nullable=True)
    total_episodes = Column(Integer, nullable=True)

    # Execution info
    result_time = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False
    )
    computation_time_seconds = Column(Float, nullable=True)

    # Logs and metadata
    logs = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Additional benchmark-specific data

    # Submission status
    submitted_to_backend = Column(Boolean, default=False, nullable=False)
    submission_error = Column(Text, nullable=True)

    # Relationships
    job = relationship("ValidatorEvaluationJob", back_populates="results")


class ValidatorState(TimestampMixin, Base):
    """Local validator state information."""

    __tablename__ = "validator_state"

    id = Column(Integer, primary_key=True)
    validator_hotkey = Column(String(48), nullable=False, unique=True, index=True)

    # Backend connection state
    backend_url = Column(String(512), nullable=True)
    connected_to_backend = Column(Boolean, default=False, nullable=False)
    last_backend_connection = Column(DateTime, nullable=True)
    last_heartbeat_sent = Column(DateTime, nullable=True)
    last_heartbeat_ack = Column(DateTime, nullable=True)

    # Job processing stats
    total_jobs_received = Column(Integer, default=0, nullable=False)
    total_jobs_completed = Column(Integer, default=0, nullable=False)
    total_jobs_failed = Column(Integer, default=0, nullable=False)

    # Version and config
    validator_version = Column(String(32), nullable=True)
    config_hash = Column(String(64), nullable=True)  # Hash of current configuration

    # Performance metrics
    avg_job_duration_seconds = Column(Float, nullable=True)
    last_performance_update = Column(DateTime, nullable=True)


class ValidatorLog(TimestampMixin, Base):
    """Local validator log entries."""

    __tablename__ = "validator_logs"

    id = Column(BigInteger, primary_key=True)

    # Log metadata
    level = Column(String(10), nullable=False, index=True)  # DEBUG, INFO, WARN, ERROR
    component = Column(
        String(64), nullable=True, index=True
    )  # websocket_validator, evaluation, etc.

    # Log content
    message = Column(Text, nullable=False)
    exception = Column(Text, nullable=True)

    # Context
    job_id = Column(String(128), nullable=True, index=True)
    benchmark = Column(String(128), nullable=True)
    miner_hotkey = Column(String(48), nullable=True, index=True)

    # Additional data
    extra_data = Column(JSON, nullable=True)


class ValidatorMetrics(TimestampMixin, Base):
    """Performance and health metrics."""

    __tablename__ = "validator_metrics"

    id = Column(BigInteger, primary_key=True)

    # Metric metadata
    metric_name = Column(String(128), nullable=False, index=True)
    metric_type = Column(String(32), nullable=False)  # counter, gauge, histogram

    # Metric values
    value = Column(Float, nullable=False)
    count = Column(Integer, nullable=True)  # For counters/histograms

    # Context
    job_id = Column(String(128), nullable=True, index=True)
    benchmark = Column(String(128), nullable=True)

    # Tags/labels
    tags = Column(JSON, nullable=True)

    # Timestamp (separate from created_at for metric collection time)
    timestamp = Column(
        DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True
    )
