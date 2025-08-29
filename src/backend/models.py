"""
SQLAlchemy models for Kinitro Backend database.
"""

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

# Type aliases for consistency
SnowflakeId = BigInteger
SS58Address = String(48)


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


class Competition(TimestampMixin, Base):
    """Competition with benchmarks and point allocation."""

    __tablename__ = "competitions"

    id = Column(String(64), primary_key=True)
    name = Column(String(256), nullable=False, unique=True)
    description = Column(Text, nullable=True)

    # Store benchmarks as JSON array: ["benchmark_a", "benchmark_b"]
    benchmarks = Column(JSON, nullable=False)

    # Points allocated to this competition for reward distribution
    points = Column(Integer, nullable=False)

    # Competition status
    active = Column(Boolean, nullable=False, server_default="true", index=True)

    # Start and end times for the competition
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)

    # Relationships
    submissions = relationship(
        "MinerSubmission", back_populates="competition", cascade="all, delete-orphan"
    )
    evaluation_jobs = relationship(
        "BackendEvaluationJob",
        back_populates="competition",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        CheckConstraint("points > 0", name="ck_competition_points_positive"),
        CheckConstraint(
            "end_time IS NULL OR start_time IS NULL OR end_time > start_time",
            name="ck_competition_times_ordered",
        ),
        Index("ix_competitions_active", "active"),
        Index("ix_competitions_points", "points"),
    )


class MinerSubmission(TimestampMixin, Base):
    """Record of a miner's submission to a competition."""

    __tablename__ = "miner_submissions"

    id = Column(SnowflakeId, primary_key=True)

    # Miner and competition info
    miner_hotkey = Column(SS58Address, nullable=False, index=True)
    competition_id = Column(
        String(64), ForeignKey("competitions.id"), nullable=False, index=True
    )

    # Submission details from chain commitment
    hf_repo_id = Column(String(256), nullable=False)
    version = Column(String(32), nullable=False)
    commitment_block = Column(BigInteger, nullable=False, index=True)
    commitment_hash = Column(
        String(128), nullable=True
    )  # Optional hash of the commitment

    # Submission timestamp (when we processed it)
    submission_time = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    competition = relationship("Competition", back_populates="submissions")
    evaluation_jobs = relationship(
        "BackendEvaluationJob",
        back_populates="submission",
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        # Prevent duplicate submissions for the same miner/competition/version
        UniqueConstraint(
            "miner_hotkey",
            "competition_id",
            "version",
            name="uq_miner_competition_version",
        ),
        Index("ix_submissions_miner_competition", "miner_hotkey", "competition_id"),
        Index("ix_submissions_block", "commitment_block"),
        Index("ix_submissions_time", "submission_time"),
    )


class BackendEvaluationJob(TimestampMixin, Base):
    """Evaluation job created by backend and sent to validators."""

    __tablename__ = "backend_evaluation_jobs"

    id = Column(SnowflakeId, primary_key=True)
    job_id = Column(
        String(128), nullable=False, unique=True, index=True
    )  # External job ID for validators

    # Link to submission and competition
    submission_id = Column(
        SnowflakeId, ForeignKey("miner_submissions.id"), nullable=False, index=True
    )
    competition_id = Column(
        String(64), ForeignKey("competitions.id"), nullable=False, index=True
    )

    # Job details
    miner_hotkey = Column(SS58Address, nullable=False, index=True)
    hf_repo_id = Column(String(256), nullable=False)
    benchmarks = Column(JSON, nullable=False)  # List of benchmarks to run

    # Job status tracking
    broadcast_time = Column(DateTime(timezone=True), nullable=True)
    validators_sent = Column(Integer, nullable=False, server_default="0")
    validators_completed = Column(Integer, nullable=False, server_default="0")

    # Relationships
    submission = relationship("MinerSubmission", back_populates="evaluation_jobs")
    competition = relationship("Competition", back_populates="evaluation_jobs")
    results = relationship(
        "BackendEvaluationResult", back_populates="job", cascade="all, delete-orphan"
    )

    __table_args__ = (
        CheckConstraint("validators_sent >= 0", name="ck_validators_sent_non_negative"),
        CheckConstraint(
            "validators_completed >= 0", name="ck_validators_completed_non_negative"
        ),
        CheckConstraint(
            "validators_completed <= validators_sent",
            name="ck_validators_completed_within_sent",
        ),
        Index("ix_backend_jobs_broadcast", "broadcast_time"),
        Index("ix_backend_jobs_miner", "miner_hotkey"),
    )


class BackendEvaluationResult(TimestampMixin, Base):
    """Evaluation results received from validators."""

    __tablename__ = "backend_evaluation_results"

    id = Column(SnowflakeId, primary_key=True)

    # Job and validator info
    job_id = Column(String(128), nullable=False, index=True)
    backend_job_id = Column(
        SnowflakeId,
        ForeignKey("backend_evaluation_jobs.id"),
        nullable=False,
        index=True,
    )
    validator_hotkey = Column(SS58Address, nullable=False, index=True)

    # Result details
    miner_hotkey = Column(SS58Address, nullable=False, index=True)
    competition_id = Column(String(64), nullable=False, index=True)
    benchmark = Column(String(128), nullable=False, index=True)

    # Scores and metrics
    score = Column(Float, nullable=False)
    success_rate = Column(Float, nullable=True)
    avg_reward = Column(Float, nullable=True)
    total_episodes = Column(Integer, nullable=True)

    # Additional data
    logs = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    extra_data = Column(JSON, nullable=True)  # Additional metrics/data

    # Timing
    result_time = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Relationships
    job = relationship("BackendEvaluationJob", back_populates="results")

    __table_args__ = (
        # Prevent duplicate results from same validator for same job/benchmark
        UniqueConstraint(
            "job_id", "validator_hotkey", "benchmark", name="uq_job_validator_benchmark"
        ),
        CheckConstraint("score >= 0", name="ck_score_non_negative"),
        CheckConstraint(
            "success_rate IS NULL OR (success_rate >= 0 AND success_rate <= 1)",
            name="ck_success_rate_range",
        ),
        CheckConstraint(
            "total_episodes IS NULL OR total_episodes > 0", name="ck_episodes_positive"
        ),
        Index("ix_backend_results_validator", "validator_hotkey"),
        Index("ix_backend_results_miner", "miner_hotkey"),
        Index("ix_backend_results_competition", "competition_id"),
        Index("ix_backend_results_benchmark", "benchmark"),
        Index("ix_backend_results_score", "score"),
        Index("ix_backend_results_time", "result_time"),
    )


class ValidatorConnection(TimestampMixin, Base):
    """Track validator connections and statistics."""

    __tablename__ = "validator_connections"

    id = Column(SnowflakeId, primary_key=True)

    validator_hotkey = Column(SS58Address, nullable=False, unique=True, index=True)
    connection_id = Column(
        String(128), nullable=False, index=True
    )  # IP:port or other identifier

    # Connection tracking
    first_connected_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_connected_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_heartbeat = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # Statistics
    total_jobs_sent = Column(Integer, nullable=False, server_default="0")
    total_results_received = Column(Integer, nullable=False, server_default="0")
    total_errors = Column(Integer, nullable=False, server_default="0")

    # Current status
    is_connected = Column(Boolean, nullable=False, server_default="true", index=True)

    __table_args__ = (
        CheckConstraint("total_jobs_sent >= 0", name="ck_jobs_sent_non_negative"),
        CheckConstraint(
            "total_results_received >= 0", name="ck_results_received_non_negative"
        ),
        CheckConstraint("total_errors >= 0", name="ck_errors_non_negative"),
        Index("ix_validator_connections_heartbeat", "last_heartbeat"),
        Index("ix_validator_connections_connected", "is_connected"),
    )


class BackendState(TimestampMixin, Base):
    """Backend service state for persistence across restarts."""

    __tablename__ = "backend_state"

    id = Column(Integer, primary_key=True)

    # Chain monitoring state
    last_seen_block = Column(BigInteger, nullable=False, server_default="0")
    last_chain_scan = Column(DateTime(timezone=True), nullable=True)

    # Service info
    service_version = Column(String(32), nullable=True)
    service_start_time = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("last_seen_block >= 0", name="ck_backend_block_non_negative"),
        CheckConstraint(
            "id = 1", name="ck_backend_state_singleton"
        ),  # Ensure only one row
    )
