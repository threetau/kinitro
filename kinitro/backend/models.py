"""Database models for the evaluation backend."""

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import BigInteger, DateTime, Float, ForeignKey, Index, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship

from kinitro.types import EnvironmentId, Hotkey, MinerUID, Seed, TaskUUID


def generate_task_uuid() -> str:
    """Generate a unique task UUID."""
    return str(uuid.uuid4())


# =============================================================================
# SQLAlchemy ORM Models
# =============================================================================


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class EvaluationCycleStatus(str, Enum):
    """Status of an evaluation cycle."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TaskStatus(str, Enum):
    """Status of a task in the task pool."""

    PENDING = "pending"
    ASSIGNED = "assigned"
    COMPLETED = "completed"
    FAILED = "failed"


class EvaluationCycleORM(Base):
    """Database model for evaluation cycles."""

    __tablename__ = "evaluation_cycles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    block_number: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=EvaluationCycleStatus.PENDING.value
    )
    n_miners: Mapped[int | None] = mapped_column(Integer, nullable=True)
    n_environments: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships
    miner_scores: Mapped[list["MinerScoreORM"]] = relationship(
        "MinerScoreORM", back_populates="cycle", cascade="all, delete-orphan"
    )
    computed_weights: Mapped[list["ComputedWeightsORM"]] = relationship(
        "ComputedWeightsORM", back_populates="cycle", cascade="all, delete-orphan"
    )


class MinerScoreORM(Base):
    """Database model for per-miner, per-environment scores."""

    __tablename__ = "miner_scores"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evaluation_cycles.id", ondelete="CASCADE"), nullable=False
    )
    uid: Mapped[int] = mapped_column(Integer, nullable=False)
    hotkey: Mapped[str] = mapped_column(String(64), nullable=False)
    env_id: Mapped[str] = mapped_column(String(64), nullable=False)
    success_rate: Mapped[float] = mapped_column(Float, nullable=False)
    mean_reward: Mapped[float] = mapped_column(Float, nullable=False)
    episodes_completed: Mapped[int] = mapped_column(Integer, nullable=False)
    episodes_failed: Mapped[int] = mapped_column(Integer, nullable=False)

    # Relationships
    cycle: Mapped["EvaluationCycleORM"] = relationship(
        "EvaluationCycleORM", back_populates="miner_scores"
    )

    __table_args__ = (
        Index("idx_miner_scores_cycle", "cycle_id"),
        Index("idx_miner_scores_uid", "uid"),
        # Prevent duplicate scores for same miner/env in a cycle
        Index(
            "idx_miner_scores_unique",
            "cycle_id",
            "uid",
            "env_id",
            unique=True,
        ),
    )


class ComputedWeightsORM(Base):
    """Database model for pre-computed weights."""

    __tablename__ = "computed_weights"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    cycle_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evaluation_cycles.id", ondelete="CASCADE"), nullable=False
    )
    block_number: Mapped[int] = mapped_column(BigInteger, nullable=False)
    weights_json: Mapped[dict[str, float]] = mapped_column(JSONB, nullable=False)
    weights_u16_json: Mapped[dict[str, list[int]]] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )

    # Relationships
    cycle: Mapped["EvaluationCycleORM"] = relationship(
        "EvaluationCycleORM", back_populates="computed_weights"
    )

    __table_args__ = (Index("idx_weights_block", "block_number"),)


class TaskPoolORM(Base):
    """Database model for the task pool.

    Tasks are created by the scheduler and executed by executors.
    This enables horizontal scaling of evaluation workloads.

    The two-tier ID system:
    - task_uuid: Random UUID for task tracking/API calls (unpredictable)
    - seed: Deterministic seed for environment reproducibility
    """

    __tablename__ = "task_pool"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    task_uuid: Mapped[str] = mapped_column(
        String(36), nullable=False, unique=True, default=generate_task_uuid
    )
    cycle_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("evaluation_cycles.id", ondelete="CASCADE"), nullable=False
    )
    miner_uid: Mapped[int] = mapped_column(Integer, nullable=False)
    miner_hotkey: Mapped[str] = mapped_column(String(64), nullable=False)
    miner_endpoint: Mapped[str] = mapped_column(Text, nullable=False)
    miner_repo: Mapped[str | None] = mapped_column(String(256), nullable=True)
    miner_revision: Mapped[str | None] = mapped_column(String(64), nullable=True)
    env_id: Mapped[str] = mapped_column(String(64), nullable=False)
    seed: Mapped[int] = mapped_column(Integer, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default=TaskStatus.PENDING.value
    )
    assigned_to: Mapped[str | None] = mapped_column(String(64), nullable=True)
    assigned_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    # Any: JSONB column — schema varies by environment/task type
    result: Mapped[dict[str, Any] | None] = mapped_column(JSONB, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=func.now()
    )

    # Relationships
    cycle: Mapped["EvaluationCycleORM"] = relationship("EvaluationCycleORM")

    __table_args__ = (
        Index("idx_task_pool_status", "status"),
        Index("idx_task_pool_cycle", "cycle_id"),
        Index("idx_task_pool_miner", "miner_uid"),
        Index("idx_task_pool_uuid", "task_uuid"),
    )


# =============================================================================
# Pydantic API Models
# =============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "ok"
    version: str = "0.1.0"
    database: str = "connected"


class MinerScore(BaseModel):
    """Score for one miner on one environment."""

    uid: MinerUID
    hotkey: Hotkey
    env_id: EnvironmentId
    success_rate: float
    mean_reward: float
    episodes_completed: int
    episodes_failed: int


class EvaluationCycle(BaseModel):
    """Evaluation cycle summary."""

    id: int
    block_number: int
    started_at: datetime
    completed_at: datetime | None
    status: EvaluationCycleStatus
    n_miners: int | None
    n_environments: int | None
    duration_seconds: float | None

    class Config:
        from_attributes = True


class ScoresResponse(BaseModel):
    """Response for /scores endpoints."""

    cycle: EvaluationCycle
    scores: list[MinerScore]

    # Aggregated by miner
    miner_summary: dict[MinerUID, dict[EnvironmentId, float]] = Field(
        default_factory=dict,
        description="Aggregated scores per miner: {uid: {env_id: success_rate}}",
    )


class WeightsU16(BaseModel):
    """Weights in u16 format for chain submission."""

    uids: list[MinerUID]
    values: list[int]


class WeightsResponse(BaseModel):
    """Response for /weights endpoints."""

    cycle_id: int
    block_number: int
    timestamp: datetime
    weights: dict[MinerUID, float] = Field(description="Normalized weights: {uid: weight}")
    weights_u16: WeightsU16 = Field(description="Weights formatted for chain submission")
    # Any: open-ended metadata for extensibility (timestamps, debug info, etc.)
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    """Current backend status."""

    current_cycle: EvaluationCycle | None
    last_completed_cycle: EvaluationCycle | None
    total_cycles: int
    total_miners_evaluated: int
    environments: list[EnvironmentId]
    is_evaluating: bool


class MinerInfo(BaseModel):
    """Information about a miner."""

    uid: MinerUID
    hotkey: Hotkey
    last_evaluated_block: int | None
    avg_success_rate: float | None
    environments_evaluated: list[EnvironmentId]


class EnvironmentInfo(BaseModel):
    """Information about an evaluation environment."""

    env_id: EnvironmentId
    env_name: str
    task_name: str
    n_evaluations: int
    avg_success_rate: float | None


# =============================================================================
# Task Pool API Models
# =============================================================================


class Task(BaseModel):
    """A single evaluation task from the task pool."""

    task_uuid: TaskUUID  # Unique identifier for API calls
    cycle_id: int
    miner_uid: MinerUID
    miner_hotkey: Hotkey
    miner_endpoint: str
    miner_repo: str | None = None  # HuggingFace repo for verification
    miner_revision: str | None = None  # HuggingFace revision for verification
    env_id: EnvironmentId
    seed: Seed  # Deterministic seed for reproducibility
    status: TaskStatus
    created_at: datetime

    class Config:
        from_attributes = True


class TaskFetchRequest(BaseModel):
    """Request to fetch tasks from the pool."""

    executor_id: str = Field(description="Unique identifier for the executor")
    batch_size: int = Field(default=10, ge=1, le=100, description="Number of tasks to fetch")
    env_ids: list[EnvironmentId] | None = Field(
        default=None, description="Filter by environment IDs"
    )


class TaskFetchResponse(BaseModel):
    """Response containing fetched tasks."""

    tasks: list[Task]
    total_pending: int = Field(description="Total pending tasks in pool")


class TaskResult(BaseModel):
    """Result of a single task execution."""

    task_uuid: TaskUUID = Field(description="UUID of the task")
    success: bool
    score: float = Field(default=0.0)
    total_reward: float = Field(default=0.0)
    timesteps: int = Field(default=0)
    error: str | None = Field(default=None)
    verification_passed: bool | None = Field(
        default=None,
        description="Whether miner passed model verification (None if not checked)",
    )
    verification_score: float | None = Field(
        default=None,
        description="Match score between deployed and HuggingFace model (0.0 to 1.0)",
    )


class TaskSubmitRequest(BaseModel):
    """Request to submit task results."""

    executor_id: str = Field(description="Executor that completed the tasks")
    results: list[TaskResult]


class TaskSubmitResponse(BaseModel):
    """Response for task submission."""

    accepted: int = Field(description="Number of results accepted")
    rejected: int = Field(description="Number of results rejected")
    errors: list[str] = Field(default_factory=list)


class TaskPoolStats(BaseModel):
    """Statistics about the task pool."""

    total_tasks: int
    pending_tasks: int
    assigned_tasks: int
    completed_tasks: int
    failed_tasks: int
    active_executors: list[str]
    current_cycle_id: int | None
