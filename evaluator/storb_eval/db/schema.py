import enum

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy.sql import func

Base = declarative_base()

SnowflakeId = BigInteger
SS58Address = String(48)


class EvaluationStatus(enum.Enum):
    queued = "queued"
    pending = "pending"
    starting = "starting"
    running = "running"
    completed = "completed"
    failed = "failed"


class SubmissionMixin:
    """Columns describing the submission / provenance."""

    submission_id = Column(SnowflakeId, nullable=False, index=True)
    miner_hotkey = Column(SS58Address, nullable=False, index=True)
    hf_repo_id = Column(String, nullable=False, index=True)
    hf_repo_commit = Column(String(40), nullable=True, index=True)
    env_provider = Column(String, nullable=False, index=True)
    env_name = Column(String, nullable=False, index=True)


class StatusMixin:
    """Status enums"""

    status = Column(
        SAEnum(EvaluationStatus, name="evaluation_status", native_enum=False),
        nullable=False,
        server_default=EvaluationStatus.pending.value,
        index=True,
    )


class TimestampMixin:
    """Common timestamps (timezone-aware)."""

    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )


class EvaluationJob(SubmissionMixin, StatusMixin, TimestampMixin, Base):
    __tablename__ = "evaluation_jobs"

    id = Column(SnowflakeId, primary_key=True)
    pgqueuer_job_id = Column(String(128), nullable=False, index=True)
    container_id = Column(String(128), nullable=True, index=True)
    ray_worker_id = Column(String(128), nullable=True, index=True)
    retry_count = Column(Integer, nullable=False, server_default="0")
    max_retries = Column(Integer, nullable=False, server_default="3")
    logs_path = Column(Text, nullable=False)
    random_seed = Column(Integer, nullable=True)
    eval_start = Column(DateTime(timezone=True), nullable=True, index=True)
    eval_end = Column(DateTime(timezone=True), nullable=True, index=True)

    # relationships
    results = relationship(
        "EvaluationResult", back_populates="evaluation", cascade="all, delete-orphan"
    )
    episodes = relationship(
        "Episode", back_populates="evaluation", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index(
            "ix_queued_eval_miner_env_status_created",
            "miner_hotkey",
            "env_name",
            "status",
            "created_at",
        ),
    )


class EvaluationResult(TimestampMixin, Base):
    __tablename__ = "evaluation_results"

    id = Column(SnowflakeId, primary_key=True)
    evaluation_id = Column(
        SnowflakeId,
        ForeignKey("evaluation_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    avg_return = Column(Float, nullable=True)
    success_rate = Column(Float, nullable=True)
    total_reward = Column(Float, nullable=True)
    num_episodes = Column(Integer, nullable=True)

    evaluation = relationship("EvaluationJob", back_populates="results")


class Episode(TimestampMixin, Base):
    __tablename__ = "episodes"

    id = Column(SnowflakeId, primary_key=True)
    evaluation_id = Column(
        SnowflakeId,
        ForeignKey("evaluation_jobs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    episode_index = Column(Integer, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=True)
    end_time = Column(DateTime(timezone=True), nullable=True)
    total_reward = Column(Float, nullable=True)
    success = Column(Boolean, nullable=False, server_default="false")

    evaluation = relationship("EvaluationJob", back_populates="episodes")
    steps = relationship(
        "EpisodeStep", back_populates="episode", cascade="all, delete-orphan"
    )


class EpisodeStep(TimestampMixin, Base):
    __tablename__ = "episode_steps"

    id = Column(SnowflakeId, primary_key=True)
    episode_id = Column(
        SnowflakeId,
        ForeignKey("episodes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    step_index = Column(Integer, nullable=False)
    observation_path = Column(JSONB, nullable=False)
    reward = Column(Float, nullable=False)
    action = Column(JSONB, nullable=False)

    episode = relationship("Episode", back_populates="steps")


DUCKDB_SCHEMA = """
-- Episodes table
CREATE TABLE IF NOT EXISTS episodes (
    id BIGINT PRIMARY KEY,
    evaluation_id BIGINT NOT NULL,
    episode_index INTEGER NOT NULL,
    start_time TIMESTAMP WITH TIME ZONE,
    end_time TIMESTAMP WITH TIME ZONE,
    total_reward DOUBLE,
    success BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Episode steps table
CREATE TABLE IF NOT EXISTS episode_steps (
    id BIGINT PRIMARY KEY,
    episode_id BIGINT NOT NULL,
    step_index INTEGER NOT NULL,
    observation_path JSON NOT NULL,
    reward DOUBLE NOT NULL,
    action JSON NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_episodes_evaluation_id ON episodes(evaluation_id);
CREATE INDEX IF NOT EXISTS idx_episodes_success ON episodes(success);
CREATE INDEX IF NOT EXISTS idx_episodes_episode_index ON episodes(episode_index);
CREATE INDEX IF NOT EXISTS idx_episode_steps_episode_id ON episode_steps(episode_id);
CREATE INDEX IF NOT EXISTS idx_episode_steps_step_index ON episode_steps(step_index);
"""
