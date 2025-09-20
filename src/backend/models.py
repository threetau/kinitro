"""
SQLModel models for Kinitro Backend database.
"""

from datetime import datetime
from typing import Dict, List, Optional

from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy import (
    DateTime as SADateTime,
)
from sqlalchemy import (
    String as SAString,
)
from sqlalchemy import (
    Text as SAText,
)
from sqlalchemy.sql import func
from sqlmodel import JSON, Column, Field, Relationship, SQLModel

from backend.constants import (
    DEFAULT_MIN_AVG_REWARD,
    DEFAULT_MIN_SUCCESS_RATE,
    DEFAULT_WIN_MARGIN_PCT,
)
from core.db.models import EvaluationStatus, SnowflakeId, TimestampMixin

# Type aliases for consistency
SS58Address = str  # Will be constrained to 48 chars in field definition
Uuid = str  # UUID string


# Models for API requests/responses
class CompetitionCreateRequest(SQLModel):
    """Request model for creating a competition."""

    name: str
    description: Optional[str] = None
    benchmarks: List[dict]
    points: int = Field(gt=0)
    min_avg_reward: float = Field(default=DEFAULT_MIN_AVG_REWARD)
    win_margin_pct: float = Field(default=DEFAULT_WIN_MARGIN_PCT)
    min_success_rate: float = Field(default=DEFAULT_MIN_SUCCESS_RATE)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class CompetitionResponse(SQLModel):
    """Response model for competition data."""

    id: Uuid
    name: str
    description: Optional[str]
    benchmarks: List[dict]
    points: int
    min_avg_reward: float
    win_margin_pct: float
    min_success_rate: float
    current_leader_hotkey: Optional[SS58Address]
    current_leader_reward: Optional[float]
    leader_updated_at: Optional[datetime]
    active: bool
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ValidatorInfoResponse(SQLModel):
    """Response model for validator information."""

    validator_hotkey: str
    connection_id: str
    api_key_id: Optional[SnowflakeId]
    is_connected: bool
    first_connected_at: datetime
    last_heartbeat: datetime
    total_jobs_sent: int
    total_results_received: int
    total_errors: int

    class Config:
        from_attributes = True


class MinerSubmissionResponse(SQLModel):
    """Response model for miner submission data."""

    id: SnowflakeId
    miner_hotkey: str
    competition_id: str
    hf_repo_id: str
    version: str
    commitment_block: int
    submission_time: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class JobResponse(SQLModel):
    """Response model for job data."""

    id: SnowflakeId
    job_id: str
    submission_id: int
    competition_id: str
    miner_hotkey: str
    hf_repo_id: str
    env_provider: str
    benchmark_name: str
    config: dict
    broadcast_time: Optional[datetime]
    created_at: datetime

    class Config:
        from_attributes = True


class EvaluationResultResponse(SQLModel):
    """Response model for evaluation result data."""

    id: SnowflakeId
    job_id: str
    validator_hotkey: str
    miner_hotkey: str
    competition_id: str
    benchmark: str
    score: float
    success_rate: Optional[float]
    avg_reward: Optional[float]
    total_episodes: Optional[int]
    error: Optional[str]
    result_time: datetime

    class Config:
        from_attributes = True


class JobStatusResponse(SQLModel):
    """Response model for job status data."""

    id: SnowflakeId
    job_id: SnowflakeId
    validator_hotkey: str
    status: EvaluationStatus
    detail: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class BackendStatsResponse(SQLModel):
    """Response model for backend statistics."""

    total_competitions: int
    active_competitions: int
    total_points: int
    connected_validators: int
    total_submissions: int
    total_jobs: int
    total_results: int
    last_seen_block: int
    competition_percentages: Dict[str, float]


class ApiKeyCreateRequest(SQLModel):
    """Request model for creating an API key."""

    name: str
    description: Optional[str] = None
    role: str  # admin, validator, viewer
    associated_hotkey: Optional[str] = None
    expires_at: Optional[datetime] = None


class ApiKeyResponse(SQLModel):
    """Response model for API key data (without the actual key)."""

    id: SnowflakeId
    name: str
    description: Optional[str]
    role: str
    associated_hotkey: Optional[str]
    is_active: bool
    last_used_at: Optional[datetime]
    expires_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ApiKeyCreateResponse(SQLModel):
    """Response model for API key creation (includes the actual key)."""

    id: SnowflakeId
    name: str
    description: Optional[str]
    role: str
    associated_hotkey: Optional[str]
    is_active: bool
    expires_at: Optional[datetime]
    created_at: datetime
    api_key: str  # The actual key - only shown once

    class Config:
        from_attributes = True


# Models for DB tables
class Competition(TimestampMixin, SQLModel, table=True):
    """Competition with benchmarks and point allocation."""

    __tablename__ = "competitions"

    id: str = Field(primary_key=True, max_length=64)
    name: str = Field(max_length=256, nullable=False, unique=True)
    description: Optional[str] = Field(
        default=None, sa_column=Column(SAText, nullable=True)
    )

    # Store benchmarks as JSON array: ["benchmark_a", "benchmark_b"]
    benchmarks: dict = Field(sa_column=Column(JSON, nullable=False))

    # Points allocated to this competition for reward distribution
    points: int = Field(nullable=False)

    # Scoring thresholds
    min_avg_reward: float = Field(
        default=DEFAULT_MIN_AVG_REWARD,
        nullable=False,
        sa_column_kwargs={"server_default": str(DEFAULT_MIN_AVG_REWARD)},
    )
    win_margin_pct: float = Field(
        default=DEFAULT_WIN_MARGIN_PCT,
        nullable=False,
        sa_column_kwargs={"server_default": str(DEFAULT_WIN_MARGIN_PCT)},
    )
    min_success_rate: float = Field(
        default=DEFAULT_MIN_SUCCESS_RATE,
        nullable=False,
        sa_column_kwargs={"server_default": str(DEFAULT_MIN_SUCCESS_RATE)},
    )

    # Current leader tracking
    current_leader_hotkey: Optional[SS58Address] = Field(
        default=None, max_length=48, nullable=True
    )
    current_leader_reward: Optional[float] = Field(default=None, nullable=True)
    leader_updated_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )

    # Competition status
    active: bool = Field(
        default=True,
        nullable=False,
        index=True,
        sa_column_kwargs={"server_default": "true"},
    )

    # Start and end times for the competition
    start_time: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )
    end_time: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )

    # Relationships
    submissions: List["MinerSubmission"] = Relationship(
        back_populates="competition", cascade_delete=True
    )
    evaluation_jobs: List["BackendEvaluationJob"] = Relationship(
        back_populates="competition", cascade_delete=True
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


class MinerSubmission(TimestampMixin, SQLModel, table=True):
    """Record of a miner's submission to a competition."""

    __tablename__ = "miner_submissions"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))

    # Miner and competition info
    miner_hotkey: str = Field(max_length=48, nullable=False, index=True)
    competition_id: str = Field(
        sa_column=Column(
            SAString(64), ForeignKey("competitions.id"), nullable=False, index=True
        )
    )

    # Submission details from chain commitment
    hf_repo_id: str = Field(max_length=256, nullable=False)
    version: str = Field(max_length=32, nullable=False)
    commitment_block: int = Field(
        sa_column=Column(BigInteger, nullable=False, index=True)
    )
    commitment_hash: Optional[str] = Field(
        default=None, max_length=128
    )  # Optional hash of the commitment

    # Submission timestamp (when we processed it)
    submission_time: datetime = Field(
        sa_column=Column(
            SADateTime(timezone=True), nullable=False, server_default=func.now()
        )
    )

    # Relationships
    competition: Optional["Competition"] = Relationship(back_populates="submissions")
    evaluation_jobs: List["BackendEvaluationJob"] = Relationship(
        back_populates="submission", cascade_delete=True
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


class BackendEvaluationJob(TimestampMixin, SQLModel, table=True):
    """Evaluation job created by backend and sent to validators."""

    __tablename__ = "backend_evaluation_jobs"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    # Link to submission and competition
    submission_id: int = Field(
        sa_column=Column(
            BigInteger, ForeignKey("miner_submissions.id"), nullable=False, index=True
        )
    )
    competition_id: str = Field(
        sa_column=Column(
            SAString(64), ForeignKey("competitions.id"), nullable=False, index=True
        )
    )

    # Job details
    miner_hotkey: str = Field(max_length=48, nullable=False, index=True)
    hf_repo_id: str = Field(max_length=256, nullable=False)
    env_provider: str = Field(max_length=64, nullable=False)
    benchmark_name: str = Field(max_length=128, nullable=False)
    config: dict = Field(sa_column=Column(JSON, nullable=False))

    # Relationships
    submission: Optional["MinerSubmission"] = Relationship(
        back_populates="evaluation_jobs"
    )
    competition: Optional["Competition"] = Relationship(
        back_populates="evaluation_jobs"
    )
    results: List["BackendEvaluationResult"] = Relationship(
        back_populates="job", cascade_delete=True
    )
    status_updates: List["BackendEvaluationJobStatus"] = Relationship(
        back_populates="job", cascade_delete=True
    )

    __table_args__ = (Index("ix_backend_jobs_miner", "miner_hotkey"),)


class BackendEvaluationJobStatus(TimestampMixin, SQLModel, table=True):
    """Track status updates for backend evaluation jobs."""

    __tablename__ = "backend_evaluation_job_status"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))

    job_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("backend_evaluation_jobs.id"),
            nullable=False,
            index=True,
        )
    )
    validator_hotkey: str = Field(
        max_length=48, nullable=False, index=True
    )  # Validator reporting the status
    status: EvaluationStatus = Field(nullable=False, index=True)
    detail: Optional[str] = Field(default=None, sa_column=Column(SAText, nullable=True))

    # Relationship
    job: Optional["BackendEvaluationJob"] = Relationship(
        back_populates="status_updates"
    )

    __table_args__ = (
        Index("ix_backend_job_status_job", "job_id"),
        Index("ix_backend_job_status_time", "created_at"),
    )


class BackendEvaluationResult(TimestampMixin, SQLModel, table=True):
    """Evaluation results received from validators."""

    __tablename__ = "backend_evaluation_results"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))

    # Job and validator info
    job_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("backend_evaluation_jobs.id"),
            nullable=False,
            index=True,
        )
    )
    validator_hotkey: str = Field(max_length=48, nullable=False, index=True)

    # Result details
    miner_hotkey: str = Field(max_length=48, nullable=False, index=True)
    competition_id: str = Field(max_length=64, nullable=False, index=True)
    env_provider: str = Field(max_length=64, nullable=False, index=True)
    benchmark: str = Field(max_length=128, nullable=False, index=True)

    # Scores and metrics
    score: float = Field(nullable=False)
    success_rate: Optional[float] = Field(default=None)
    avg_reward: Optional[float] = Field(default=None)
    total_episodes: Optional[int] = Field(default=None)

    # Additional data
    logs: Optional[str] = Field(default=None, sa_column=Column(SAText, nullable=True))
    error: Optional[str] = Field(default=None, sa_column=Column(SAText, nullable=True))
    extra_data: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )  # Additional metrics/data

    # Timing
    result_time: datetime = Field(
        sa_column=Column(
            SADateTime(timezone=True), nullable=False, server_default=func.now()
        )
    )

    # Relationships
    job: Optional["BackendEvaluationJob"] = Relationship(back_populates="results")

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


class EpisodeData(TimestampMixin, SQLModel, table=True):
    """Episode-level data for evaluation runs."""

    __tablename__ = "episode_data"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))

    # Link to evaluation job
    job_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("backend_evaluation_jobs.id"),
            nullable=False,
            index=True,
        )
    )
    submission_id: str = Field(max_length=128, nullable=False, index=True)

    # Episode metadata
    task_id: str = Field(max_length=128, nullable=False, index=True)
    episode_id: int = Field(nullable=False, index=True)
    env_name: str = Field(max_length=128, nullable=False)
    benchmark_name: str = Field(max_length=128, nullable=False)

    # Episode results
    total_reward: float = Field(nullable=False)
    success: bool = Field(nullable=False, index=True)
    steps: int = Field(nullable=False)

    # Timing
    start_time: datetime = Field(
        sa_column=Column(SADateTime(timezone=True), nullable=False)
    )
    end_time: datetime = Field(
        sa_column=Column(SADateTime(timezone=True), nullable=False)
    )

    # Additional metrics
    extra_metrics: Optional[dict] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )

    # Relationships
    episode_steps: List["EpisodeStepData"] = Relationship(
        back_populates="episode", cascade_delete=True
    )

    __table_args__ = (
        Index("ix_episode_data_submission", "submission_id"),
        Index("ix_episode_data_task", "task_id"),
        Index("ix_episode_data_episode", "episode_id"),
        Index("ix_episode_data_success", "success"),
        Index("ix_episode_data_submission_task", "submission_id", "task_id"),
    )


class EpisodeStepData(TimestampMixin, SQLModel, table=True):
    """Step-level data within episodes."""

    __tablename__ = "episode_step_data"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))

    # Link to episode
    episode_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("episode_data.id"),
            nullable=False,
            index=True,
        )
    )
    submission_id: str = Field(max_length=128, nullable=False, index=True)

    # Step metadata
    task_id: str = Field(max_length=128, nullable=False, index=True)
    step: int = Field(nullable=False, index=True)

    # Action taken
    action: dict = Field(sa_column=Column(JSON, nullable=False))

    # Reward received
    reward: float = Field(nullable=False)

    # Terminal states
    done: bool = Field(nullable=False)
    truncated: bool = Field(default=False, nullable=False)

    # Observation storage references
    observation_refs: dict = Field(
        sa_column=Column(JSON, nullable=False)
    )  # {"camera_name": {"bucket": "...", "key": "...", "url": "..."}}

    # Additional info from environment
    info: Optional[dict] = Field(default=None, sa_column=Column(JSON, nullable=True))

    # Timing
    timestamp: datetime = Field(
        sa_column=Column(SADateTime(timezone=True), nullable=False)
    )

    # Relationships
    episode: Optional["EpisodeData"] = Relationship(back_populates="episode_steps")

    __table_args__ = (
        Index("ix_episode_step_submission", "submission_id"),
        Index("ix_episode_step_task", "task_id"),
        Index("ix_episode_step_step", "step"),
        UniqueConstraint(
            "episode_id", "step", name="uq_episode_step"
        ),  # Ensure unique steps per episode
    )


class ValidatorConnection(TimestampMixin, SQLModel, table=True):
    """Track validator connections and statistics."""

    __tablename__ = "validator_connections"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))

    validator_hotkey: str = Field(
        max_length=48, nullable=False, unique=True, index=True
    )
    # TODO: better naming?
    connection_id: str = Field(
        max_length=128, nullable=False, index=True
    )  # IP:port or other identifier

    # Link to API key used for authentication
    api_key_id: Optional[int] = Field(
        sa_column=Column(
            BigInteger, ForeignKey("api_keys.id"), nullable=True, index=True
        )
    )

    # Connection tracking
    first_connected_at: datetime = Field(
        sa_column=Column(
            SADateTime(timezone=True), nullable=False, server_default=func.now()
        )
    )
    last_connected_at: datetime = Field(
        sa_column=Column(
            SADateTime(timezone=True), nullable=False, server_default=func.now()
        )
    )
    last_heartbeat: datetime = Field(
        sa_column=Column(
            SADateTime(timezone=True), nullable=False, server_default=func.now()
        )
    )

    # Statistics
    total_jobs_sent: int = Field(
        default=0, nullable=False, sa_column_kwargs={"server_default": "0"}
    )
    total_results_received: int = Field(
        default=0, nullable=False, sa_column_kwargs={"server_default": "0"}
    )
    total_errors: int = Field(
        default=0, nullable=False, sa_column_kwargs={"server_default": "0"}
    )

    # Current status
    is_connected: bool = Field(
        default=True,
        nullable=False,
        index=True,
        sa_column_kwargs={"server_default": "true"},
    )

    # Relationships
    api_key: Optional["ApiKey"] = Relationship(back_populates="validator_connections")

    __table_args__ = (
        CheckConstraint("total_jobs_sent >= 0", name="ck_jobs_sent_non_negative"),
        CheckConstraint(
            "total_results_received >= 0", name="ck_results_received_non_negative"
        ),
        CheckConstraint("total_errors >= 0", name="ck_errors_non_negative"),
        Index("ix_validator_connections_heartbeat", "last_heartbeat"),
        Index("ix_validator_connections_connected", "is_connected"),
    )


class BackendState(TimestampMixin, SQLModel, table=True):
    """Backend service state for persistence across restarts."""

    __tablename__ = "backend_state"

    id: int = Field(primary_key=True)

    # Chain monitoring state
    last_seen_block: int = Field(
        default=0, sa_column=Column(BigInteger, nullable=False, server_default="0")
    )
    last_chain_scan: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )

    # Service info
    service_version: Optional[str] = Field(default=None, max_length=32)
    service_start_time: datetime = Field(
        sa_column=Column(
            SADateTime(timezone=True), nullable=False, server_default=func.now()
        )
    )

    __table_args__ = (
        CheckConstraint("last_seen_block >= 0", name="ck_backend_block_non_negative"),
        CheckConstraint(
            "id = 1", name="ck_backend_state_singleton"
        ),  # Ensure only one row
    )


class ApiKey(TimestampMixin, SQLModel, table=True):
    """API keys for authentication and authorization."""

    __tablename__ = "api_keys"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))

    # User identification
    name: str = Field(max_length=128, nullable=False)
    description: Optional[str] = Field(
        default=None, sa_column=Column(SAText, nullable=True)
    )

    # The hashed API key (SHA256)
    key_hash: str = Field(max_length=64, nullable=False, unique=True, index=True)

    # Role for authorization
    role: str = Field(max_length=32, nullable=False, index=True)

    # Optional association with a specific hotkey (for validators)
    associated_hotkey: Optional[SS58Address] = Field(
        default=None, max_length=48, nullable=True, index=True
    )

    # Status
    is_active: bool = Field(
        default=True,
        nullable=False,
        index=True,
        sa_column_kwargs={"server_default": "true"},
    )

    # Usage tracking
    last_used_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )

    # Expiration (optional)
    expires_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )

    # Relationships
    validator_connections: List["ValidatorConnection"] = Relationship(
        back_populates="api_key", cascade_delete=True
    )

    __table_args__ = (
        Index("ix_api_keys_active", "is_active"),
        Index("ix_api_keys_expires", "expires_at"),
        CheckConstraint(
            "role IN ('admin', 'validator', 'viewer')", name="ck_api_keys_valid_role"
        ),
    )
