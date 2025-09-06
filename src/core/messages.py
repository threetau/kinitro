"""
Shared message models for backend-validator communication in Kinitro.

These Pydantic models define the message formats used for WebSocket
communication between the Kinitro Backend and Validators.
"""

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from validator.db.models import SnowflakeId


class EvalJobMessage(BaseModel):
    """Message for broadcasting evaluation jobs from backend to validators."""

    message_type: str = "eval_job"
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


class EvalResultMessage(BaseModel):
    """Message for sending evaluation results from validators to backend."""

    message_type: str = "eval_result"
    job_id: SnowflakeId
    validator_hotkey: str
    miner_hotkey: str
    competition_id: str
    benchmark: dict
    score: float
    success_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    total_episodes: Optional[int] = None
    logs: Optional[str] = None
    error: Optional[str] = None
    extra_data: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ValidatorRegisterMessage(BaseModel):
    """Message for validator registration with backend."""

    message_type: str = "register"
    hotkey: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HeartbeatMessage(BaseModel):
    """Message for validator heartbeat."""

    message_type: str = "heartbeat"
    queue_size: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HeartbeatAckMessage(BaseModel):
    """Acknowledgment message for heartbeat."""

    message_type: str = "heartbeat_ack"
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RegistrationAckMessage(BaseModel):
    """Acknowledgment message for validator registration."""

    message_type: str = "registration_ack"
    status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ResultAckMessage(BaseModel):
    """Acknowledgment message for result submission."""

    message_type: str = "result_ack"
    job_id: SnowflakeId
    status: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ErrorMessage(BaseModel):
    """Error message for communication issues."""

    message_type: str = "error"
    error: str
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
