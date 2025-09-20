"""
Shared message models for backend-validator communication in Kinitro.

These SQLModel models define the message formats used for WebSocket
communication between the Kinitro Backend and Validators.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Optional

from sqlmodel import Field, SQLModel

from core.db.models import EvaluationStatus, SnowflakeId


class MessageType(StrEnum):
    """Enumeration of all message types used in Kinitro communication."""

    EVAL_JOB = "eval_job"
    EVAL_RESULT = "eval_result"
    REGISTER = "register"
    HEARTBEAT = "heartbeat"
    HEARTBEAT_ACK = "heartbeat_ack"
    REGISTRATION_ACK = "registration_ack"
    RESULT_ACK = "result_ack"
    SET_WEIGHTS = "set_weights"
    EPISODE_DATA = "episode_data"
    EPISODE_STEP_DATA = "episode_step_data"
    ERROR = "error"


class EvalJobMessage(SQLModel):
    """Message for broadcasting evaluation jobs from backend to validators."""

    message_type: MessageType = MessageType.EVAL_JOB
    job_id: SnowflakeId
    competition_id: str
    submission_id: int
    miner_hotkey: str
    hf_repo_id: str
    env_provider: str
    benchmark_name: str
    config: dict
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "EvalJobMessage":
        return cls.model_validate_json(data)


class EvalResultMessage(SQLModel):
    """Message for sending evaluation results from validators to backend."""

    message_type: MessageType = MessageType.EVAL_RESULT
    job_id: SnowflakeId
    validator_hotkey: str
    miner_hotkey: str
    competition_id: str
    env_provider: str
    benchmark_name: str
    config: dict
    score: float
    success_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    total_episodes: Optional[int] = None
    logs: Optional[str] = None
    error: Optional[str] = None
    extra_data: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SetWeightsMessage(SQLModel):
    """Message for sending model weights from backend to validators.

    weights maps miner UIDs to their corresponding weights.
    """

    message_type: MessageType = MessageType.SET_WEIGHTS
    weights: dict[int, float]  # Maps miner UID to weight
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ValidatorRegisterMessage(SQLModel):
    """Message for validator registration with backend."""

    message_type: MessageType = MessageType.REGISTER
    hotkey: str
    api_key: str  # API key for authentication
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HeartbeatMessage(SQLModel):
    """Message for validator heartbeat."""

    message_type: MessageType = MessageType.HEARTBEAT
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HeartbeatAckMessage(SQLModel):
    """Acknowledgment message for heartbeat."""

    message_type: MessageType = MessageType.HEARTBEAT_ACK
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegistrationAckMessage(SQLModel):
    """Acknowledgment message for validator registration."""

    message_type: MessageType = MessageType.REGISTRATION_ACK
    status: EvaluationStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResultAckMessage(SQLModel):
    """Acknowledgment message for result submission."""

    message_type: MessageType = MessageType.RESULT_ACK
    job_id: SnowflakeId
    status: EvaluationStatus
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EpisodeDataMessage(SQLModel):
    """Message for sending episode data from validators to backend."""

    message_type: MessageType = MessageType.EPISODE_DATA
    job_id: SnowflakeId
    submission_id: str
    task_id: str  # Unique identifier for the task within the job
    episode_id: int
    env_name: str
    benchmark_name: str
    total_reward: float
    success: bool
    steps: int
    start_time: datetime
    end_time: datetime
    extra_metrics: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EpisodeStepDataMessage(SQLModel):
    """Message for sending episode step data from validators to backend."""

    message_type: MessageType = MessageType.EPISODE_STEP_DATA
    submission_id: str
    task_id: str  # Unique identifier for the task within the job
    episode_id: int
    step: int
    action: dict
    reward: float
    done: bool
    truncated: bool
    observation_refs: dict
    info: Optional[dict] = None
    step_timestamp: datetime
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorMessage(SQLModel):
    """Error message for communication issues."""

    message_type: MessageType = MessageType.ERROR
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
