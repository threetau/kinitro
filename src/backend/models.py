"""
SQLModel models for Kinitro Backend database.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, field_validator
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    UniqueConstraint,
)
from sqlalchemy import (
    DateTime as SADateTime,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy import (
    String as SAString,
)
from sqlalchemy import Text as SAText
from sqlalchemy.dialects.postgresql import ENUM as PGEnum  # noqa: N811
from sqlalchemy.sql import func
from sqlmodel import JSON, Column, Field, Relationship, SQLModel

from backend.constants import (
    DEFAULT_MIN_AVG_REWARD,
    DEFAULT_MIN_SUCCESS_RATE,
    DEFAULT_SUBMISSION_HOLDOUT_SECONDS,
    DEFAULT_WIN_MARGIN_PCT,
    EVAL_JOB_TIMEOUT,
)
from core.db.models import EvaluationStatus, TimestampMixin

# Type aliases for consistency
SS58Address = str  # Will be constrained to 48 chars in field definition
Uuid = str  # UUID string


class WeightsSnapshot(BaseModel):
    """Cached weight set broadcast to validators."""

    updated_at: datetime
    total_weight: float
    weights: Dict[int, float]


# Models for API requests/responses
class SubmissionUploadStatus(StrEnum):
    """Lifecycle states for direct-upload submissions."""

    PENDING = "pending"
    READY = "ready"
    PROCESSED = "processed"
    EXPIRED = "expired"


class LeaderCandidateStatus(StrEnum):
    """Review lifecycle for prospective competition leaders."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


class CompetitionCreateRequest(SQLModel):
    """Request model for creating a competition."""

    name: str
    description: Optional[str] = None
    benchmarks: List[dict]
    points: int = Field(gt=0)
    min_avg_reward: float = Field(default=DEFAULT_MIN_AVG_REWARD)
    win_margin_pct: float = Field(default=DEFAULT_WIN_MARGIN_PCT)
    min_success_rate: float = Field(default=DEFAULT_MIN_SUCCESS_RATE)
    job_timeout_seconds: int = Field(default=EVAL_JOB_TIMEOUT, ge=1)
    submission_holdout_seconds: int = Field(
        default=DEFAULT_SUBMISSION_HOLDOUT_SECONDS, ge=0
    )
    submission_max_size_bytes: Optional[int] = Field(default=None, ge=1)
    submission_upload_window_seconds: Optional[int] = Field(default=None, ge=1)
    submission_uploads_per_window: Optional[int] = Field(default=None, ge=1)
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
    job_timeout_seconds: int
    submission_holdout_seconds: int
    submission_max_size_bytes: Optional[int]
    submission_upload_window_seconds: Optional[int]
    submission_uploads_per_window: Optional[int]
    current_leader_hotkey: Optional[SS58Address]
    current_leader_reward: Optional[float]
    current_leader_success_rate: Optional[float] = None
    leader_updated_at: Optional[datetime]
    active: bool
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class LeaderCandidateReviewRequest(SQLModel):
    """Request payload for approving or rejecting a leader candidate."""

    reason: Optional[str] = None


class SubmissionRerunRequest(SQLModel):
    """Request payload for re-running submission evaluations."""

    benchmarks: Optional[List[str]] = None


class CompetitionLeaderCandidateResponse(SQLModel):
    """Response model representing a leader candidate entry."""

    id: str
    competition_id: str
    miner_hotkey: str
    evaluation_result_id: str
    avg_reward: float
    success_rate: Optional[float]
    score: Optional[float]
    total_episodes: Optional[int]
    status: LeaderCandidateStatus
    status_reason: Optional[str]
    reviewed_by_api_key_id: Optional[str]
    reviewed_at: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @field_validator("id", mode="before")
    @classmethod
    def _convert_id(cls, value):
        return str(value)

    @field_validator("evaluation_result_id", "reviewed_by_api_key_id", mode="before")
    @classmethod
    def _convert_optional_ids(cls, value):
        return None if value is None else str(value)


class ValidatorInfoResponse(SQLModel):
    """Response model for validator information."""

    validator_hotkey: str
    connection_id: str
    api_key_id: Optional[str]
    is_connected: bool
    first_connected_at: datetime
    last_heartbeat: datetime
    total_jobs_sent: int
    total_results_received: int
    total_errors: int

    class Config:
        from_attributes = True

    @field_validator("api_key_id", mode="before")
    @classmethod
    def _convert_api_key_id(cls, value):
        return None if value is None else str(value)


class MinerSubmissionResponse(SQLModel):
    """Response model for miner submission data."""

    id: str
    miner_hotkey: str
    competition_id: str
    hf_repo_id: str
    version: str
    commitment_block: int
    submission_time: datetime
    created_at: datetime
    evaluation_status: Optional[EvaluationStatus] = None

    class Config:
        from_attributes = True

    @field_validator("id", mode="before")
    @classmethod
    def _convert_id(cls, value):
        return str(value)


class RevealedSubmissionResponse(MinerSubmissionResponse):
    """Response model for submissions whose hold-out period has ended."""

    holdout_release_at: Optional[datetime]
    released_at: Optional[datetime]
    public_artifact_url: Optional[str]
    public_artifact_url_expires_at: Optional[datetime]
    artifact_sha256: Optional[str]
    artifact_size_bytes: Optional[int]

    class Config:
        from_attributes = True


class JobResponse(SQLModel):
    """Response model for job data."""

    id: str
    submission_id: str
    competition_id: str
    miner_hotkey: str
    hf_repo_id: str
    env_provider: str
    benchmark_name: str
    config: dict
    timeout_seconds: Optional[int]
    created_at: datetime

    class Config:
        from_attributes = True

    @field_validator("id", "submission_id", mode="before")
    @classmethod
    def _convert_ids(cls, value):
        return str(value)


class EvaluationResultResponse(SQLModel):
    """Response model for evaluation result data."""

    id: str
    job_id: str
    validator_hotkey: str
    miner_hotkey: str
    competition_id: str
    benchmark: str
    score: float
    success_rate: Optional[float]
    avg_reward: Optional[float]
    total_episodes: Optional[int]
    env_specs: Optional[List[Dict[str, Any]]] = None
    error: Optional[str]
    result_time: datetime

    class Config:
        from_attributes = True

    @field_validator("id", "job_id", mode="before")
    @classmethod
    def _convert_ids(cls, value):
        return str(value)


class EvaluationLogDownloadResponse(SQLModel):
    """Response model describing a downloadable evaluator log bundle."""

    url: str
    expires_at: Optional[datetime] = None

    class Config:
        from_attributes = True


class EvaluationResultLogResponse(SQLModel):
    """Detailed evaluator log information for a result."""

    result_id: str
    job_id: str
    summary: Optional[Dict[str, Any]] = None
    download: Optional[EvaluationLogDownloadResponse] = None
    artifact_metadata: Optional[Dict[str, Any]] = None
    inline_logs: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True

    @field_validator("result_id", "job_id", mode="before")
    @classmethod
    def _convert_ids(cls, value):
        return str(value)


class JobStatusResponse(SQLModel):
    """Response model for job status data."""

    id: str
    job_id: str
    validator_hotkey: str
    status: EvaluationStatus
    detail: Optional[str]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @field_validator("id", "job_id", mode="before")
    @classmethod
    def _convert_ids(cls, value):
        return str(value)


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


class CompetitionLeaderInfo(SQLModel):
    """Summary of an individual competition's active leader."""

    competition_id: str
    competition_name: str
    points: int
    current_leader_hotkey: Optional[str]
    current_leader_submission_id: Optional[str]
    current_leader_reward: Optional[float]
    current_leader_success_rate: Optional[float]
    leader_updated_at: Optional[datetime]


class AgentLeaderboardEntry(SQLModel):
    """Aggregated leaderboard entry for a miner across competitions."""

    rank: int
    miner_hotkey: str
    total_points: int
    normalized_score: float
    competitions: List[str]
    competition_submission_ids: Dict[str, str]


class CompetitionLeaderboardResponse(SQLModel):
    """Response model for the competition-wide agent leaderboard."""

    total_competitions: int
    total_points: int
    leaders: List[AgentLeaderboardEntry]
    competitions: List[CompetitionLeaderInfo]


class SubmissionLeaderboardEntry(SQLModel):
    """Aggregated submission performance across evaluation results."""

    rank: int
    submission_id: str
    competition_id: str
    miner_hotkey: str
    hf_repo_id: Optional[str]
    version: Optional[str]
    avg_reward: Optional[float]
    success_rate: Optional[float]
    score: Optional[float]
    total_results: int
    total_episodes: Optional[int]
    last_result_time: Optional[datetime]


class SubmissionLeaderboardResponse(SQLModel):
    """Response model for submission-focused leaderboards."""

    total_submissions: int
    offset: int
    limit: int
    sort_by: str
    sort_direction: str
    entries: List[SubmissionLeaderboardEntry]


class ApiKeyCreateRequest(SQLModel):
    """Request model for creating an API key."""

    name: str
    description: Optional[str] = None
    role: str  # admin, validator, viewer
    associated_hotkey: Optional[str] = None
    expires_at: Optional[datetime] = None


class ApiKeyResponse(SQLModel):
    """Response model for API key data (without the actual key)."""

    id: str
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

    @field_validator("id", mode="before")
    @classmethod
    def _convert_id(cls, value):
        return str(value)


class ApiKeyCreateResponse(SQLModel):
    """Response model for API key creation (includes the actual key)."""

    id: str
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

    @field_validator("id", mode="before")
    @classmethod
    def _convert_id(cls, value):
        return str(value)


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
    job_timeout_seconds: int = Field(
        default=EVAL_JOB_TIMEOUT,
        nullable=False,
        sa_column_kwargs={"server_default": str(EVAL_JOB_TIMEOUT)},
    )
    submission_holdout_seconds: int = Field(
        default=DEFAULT_SUBMISSION_HOLDOUT_SECONDS,
        nullable=False,
        sa_column_kwargs={"server_default": str(DEFAULT_SUBMISSION_HOLDOUT_SECONDS)},
    )
    submission_max_size_bytes: Optional[int] = Field(
        default=None, sa_column=Column(BigInteger, nullable=True)
    )
    submission_upload_window_seconds: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )
    submission_uploads_per_window: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
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
    leader_candidates: List["CompetitionLeaderCandidate"] = Relationship(
        back_populates="competition", cascade_delete=True
    )

    __table_args__ = (
        CheckConstraint("points > 0", name="ck_competition_points_positive"),
        CheckConstraint(
            "end_time IS NULL OR start_time IS NULL OR end_time > start_time",
            name="ck_competition_times_ordered",
        ),
        CheckConstraint(
            "submission_max_size_bytes IS NULL OR submission_max_size_bytes > 0",
            name="ck_competition_submission_max_size_positive",
        ),
        CheckConstraint(
            "submission_upload_window_seconds IS NULL OR submission_upload_window_seconds > 0",
            name="ck_competition_upload_window_positive",
        ),
        CheckConstraint(
            "submission_uploads_per_window IS NULL OR submission_uploads_per_window > 0",
            name="ck_competition_uploads_per_window_positive",
        ),
        Index("ix_competitions_active", "active"),
        Index("ix_competitions_points", "points"),
    )


class SubmissionUpload(TimestampMixin, SQLModel, table=True):
    """Pending submission artifact stored in the vault before commitment."""

    __tablename__ = "submission_uploads"

    submission_id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    miner_hotkey: str = Field(
        sa_column=Column(SAString(48), nullable=False, index=True)
    )
    competition_id: str = Field(
        sa_column=Column(
            SAString(64), ForeignKey("competitions.id"), nullable=False, index=True
        )
    )
    version: str = Field(sa_column=Column(SAString(32), nullable=False))
    artifact_object_key: str = Field(sa_column=Column(SAString(512), nullable=False))
    artifact_sha256: str = Field(sa_column=Column(SAString(64), nullable=False))
    artifact_size_bytes: int = Field(sa_column=Column(BigInteger, nullable=False))
    status: SubmissionUploadStatus = Field(
        default=SubmissionUploadStatus.PENDING,
        sa_column=Column(
            SAEnum(SubmissionUploadStatus, name="submission_upload_status"),
            nullable=False,
            default=SubmissionUploadStatus.PENDING,
        ),
    )
    upload_url_expires_at: datetime = Field(
        sa_column=Column(SADateTime(timezone=True), nullable=False)
    )
    uploaded_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )
    holdout_seconds: int = Field(
        default=DEFAULT_SUBMISSION_HOLDOUT_SECONDS, nullable=False
    )
    notes: Optional[str] = Field(default=None, sa_column=Column(SAText, nullable=True))

    __table_args__ = (
        Index("ix_submission_uploads_hotkey", "miner_hotkey"),
        Index("ix_submission_uploads_status", "status"),
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

    # Hold-out storage metadata
    artifact_object_key: Optional[str] = Field(
        default=None, sa_column=Column(SAString(512), nullable=True)
    )
    artifact_sha256: Optional[str] = Field(
        default=None, sa_column=Column(SAString(64), nullable=True)
    )
    artifact_size_bytes: Optional[int] = Field(
        default=None, sa_column=Column(BigInteger, nullable=True)
    )
    holdout_release_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SADateTime(timezone=True), nullable=True, index=True),
    )
    released_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )
    public_artifact_url: Optional[str] = Field(
        default=None, max_length=512, sa_column=Column(SAString(512), nullable=True)
    )
    public_artifact_url_expires_at: Optional[datetime] = Field(
        default=None,
        sa_column=Column(SADateTime(timezone=True), nullable=True),
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
    timeout_seconds: Optional[int] = Field(
        default=None, sa_column=Column(Integer, nullable=True)
    )

    artifact_object_key: Optional[str] = Field(
        default=None, sa_column=Column(SAString(512), nullable=True)
    )
    artifact_sha256: Optional[str] = Field(
        default=None, sa_column=Column(SAString(64), nullable=True)
    )
    artifact_size_bytes: Optional[int] = Field(
        default=None, sa_column=Column(BigInteger, nullable=True)
    )

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
    env_specs: Optional[List[Dict[str, Any]]] = Field(
        default=None, sa_column=Column(JSON, nullable=True)
    )  # Serialized environment specs used for this result

    # Timing
    result_time: datetime = Field(
        sa_column=Column(
            SADateTime(timezone=True), nullable=False, server_default=func.now()
        )
    )

    # Relationships
    job: Optional["BackendEvaluationJob"] = Relationship(back_populates="results")
    leader_candidates: List["CompetitionLeaderCandidate"] = Relationship(
        back_populates="evaluation_result"
    )

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
    validator_hotkey: Optional[SS58Address] = Field(
        default=None,
        sa_column=Column(SAString(48), nullable=True),
    )

    # Episode metadata
    task_id: str = Field(max_length=128, nullable=False, index=True)
    episode_id: int = Field(nullable=False, index=True)
    env_name: str = Field(max_length=128, nullable=False)
    benchmark_name: str = Field(max_length=128, nullable=False)

    # Episode results
    final_reward: float = Field(nullable=False)
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
        Index("ix_episode_data_validator", "validator_hotkey"),
        UniqueConstraint(
            "submission_id",
            "task_id",
            "episode_id",
            "validator_hotkey",
            name="uq_episode_data_submission_task_episode_validator",
        ),
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
    validator_hotkey: Optional[SS58Address] = Field(
        default=None,
        sa_column=Column(SAString(48), nullable=True),
    )

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
        Index("ix_episode_step_validator", "validator_hotkey"),
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
    reviewed_leader_candidates: List["CompetitionLeaderCandidate"] = Relationship(
        back_populates="reviewed_by"
    )

    __table_args__ = (
        Index("ix_api_keys_active", "is_active"),
        Index("ix_api_keys_expires", "expires_at"),
        CheckConstraint(
            "role IN ('admin', 'validator', 'viewer')", name="ck_api_keys_valid_role"
        ),
    )


class CompetitionLeaderCandidate(TimestampMixin, SQLModel, table=True):
    """Pending or reviewed contenders for competition leadership."""

    __tablename__ = "competition_leader_candidates"

    id: int = Field(sa_column=Column(BigInteger, primary_key=True))
    competition_id: str = Field(
        sa_column=Column(
            SAString(64), ForeignKey("competitions.id"), nullable=False, index=True
        )
    )
    miner_hotkey: str = Field(max_length=48, nullable=False, index=True)
    evaluation_result_id: int = Field(
        sa_column=Column(
            BigInteger,
            ForeignKey("backend_evaluation_results.id"),
            nullable=False,
            index=True,
        )
    )
    avg_reward: float = Field(nullable=False)
    success_rate: Optional[float] = Field(default=None)
    score: Optional[float] = Field(default=None)
    total_episodes: Optional[int] = Field(default=None)
    status: LeaderCandidateStatus = Field(
        default=LeaderCandidateStatus.PENDING,
        sa_column=Column(
            PGEnum(
                LeaderCandidateStatus,
                values_callable=lambda enum_cls: [member.value for member in enum_cls],
                name="leader_candidate_status",
                create_type=False,
            ),
            nullable=False,
            server_default="pending",
        ),
    )
    status_reason: Optional[str] = Field(
        default=None, sa_column=Column(SAText, nullable=True)
    )
    reviewed_by_api_key_id: Optional[int] = Field(
        default=None,
        sa_column=Column(
            BigInteger, ForeignKey("api_keys.id"), nullable=True, index=True
        ),
    )
    reviewed_at: Optional[datetime] = Field(
        default=None, sa_column=Column(SADateTime(timezone=True), nullable=True)
    )

    competition: Optional["Competition"] = Relationship(
        back_populates="leader_candidates"
    )
    evaluation_result: Optional["BackendEvaluationResult"] = Relationship(
        back_populates="leader_candidates"
    )
    reviewed_by: Optional["ApiKey"] = Relationship(
        back_populates="reviewed_leader_candidates"
    )

    __table_args__ = (
        UniqueConstraint(
            "evaluation_result_id", name="uq_leader_candidates_eval_result"
        ),
        Index("ix_leader_candidates_status", "status"),
        Index("ix_leader_candidates_created_at", "created_at"),
    )
