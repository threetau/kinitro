"""Database models for the evaluation backend."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, relationship

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


class EvaluationCycleORM(Base):
    """Database model for evaluation cycles."""

    __tablename__ = "evaluation_cycles"

    id = Column(Integer, primary_key=True, autoincrement=True)
    block_number = Column(BigInteger, nullable=False, index=True)
    started_at = Column(DateTime, nullable=False, default=func.now())
    completed_at = Column(DateTime, nullable=True)
    status = Column(String(20), nullable=False, default=EvaluationCycleStatus.PENDING.value)
    n_miners = Column(Integer, nullable=True)
    n_environments = Column(Integer, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    error_message = Column(Text, nullable=True)

    # Relationships
    miner_scores = relationship(
        "MinerScoreORM", back_populates="cycle", cascade="all, delete-orphan"
    )
    computed_weights = relationship(
        "ComputedWeightsORM", back_populates="cycle", cascade="all, delete-orphan"
    )


class MinerScoreORM(Base):
    """Database model for per-miner, per-environment scores."""

    __tablename__ = "miner_scores"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(
        Integer, ForeignKey("evaluation_cycles.id", ondelete="CASCADE"), nullable=False
    )
    uid = Column(Integer, nullable=False)
    hotkey = Column(String(64), nullable=False)
    env_id = Column(String(64), nullable=False)
    success_rate = Column(Float, nullable=False)
    mean_reward = Column(Float, nullable=False)
    episodes_completed = Column(Integer, nullable=False)
    episodes_failed = Column(Integer, nullable=False)

    # Relationships
    cycle = relationship("EvaluationCycleORM", back_populates="miner_scores")

    __table_args__ = (
        Index("idx_miner_scores_cycle", "cycle_id"),
        Index("idx_miner_scores_uid", "uid"),
    )


class ComputedWeightsORM(Base):
    """Database model for pre-computed weights."""

    __tablename__ = "computed_weights"

    id = Column(Integer, primary_key=True, autoincrement=True)
    cycle_id = Column(
        Integer, ForeignKey("evaluation_cycles.id", ondelete="CASCADE"), nullable=False
    )
    block_number = Column(BigInteger, nullable=False)
    weights_json = Column(JSONB, nullable=False)  # {uid: weight_float}
    weights_u16_json = Column(JSONB, nullable=False)  # {uids: [], values: []}
    created_at = Column(DateTime, nullable=False, default=func.now())

    # Relationships
    cycle = relationship("EvaluationCycleORM", back_populates="computed_weights")

    __table_args__ = (Index("idx_weights_block", "block_number"),)


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

    uid: int
    hotkey: str
    env_id: str
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
    status: str
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
    miner_summary: dict[int, dict[str, float]] = Field(
        default_factory=dict,
        description="Aggregated scores per miner: {uid: {env_id: success_rate}}",
    )


class WeightsU16(BaseModel):
    """Weights in u16 format for chain submission."""

    uids: list[int]
    values: list[int]


class WeightsResponse(BaseModel):
    """Response for /weights endpoints."""

    cycle_id: int
    block_number: int
    timestamp: datetime
    weights: dict[int, float] = Field(description="Normalized weights: {uid: weight}")
    weights_u16: WeightsU16 = Field(description="Weights formatted for chain submission")
    metadata: dict[str, Any] = Field(default_factory=dict)


class StatusResponse(BaseModel):
    """Current backend status."""

    current_cycle: EvaluationCycle | None
    last_completed_cycle: EvaluationCycle | None
    total_cycles: int
    total_miners_evaluated: int
    environments: list[str]
    is_evaluating: bool


class MinerInfo(BaseModel):
    """Information about a miner."""

    uid: int
    hotkey: str
    last_evaluated_block: int | None
    avg_success_rate: float | None
    environments_evaluated: list[str]


class EnvironmentInfo(BaseModel):
    """Information about an evaluation environment."""

    env_id: str
    env_name: str
    task_name: str
    n_evaluations: int
    avg_success_rate: float | None
