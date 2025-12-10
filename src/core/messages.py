"""
Shared message models for backend-validator communication in Kinitro.

These SQLModel models define the message formats used for WebSocket
communication between the Kinitro Backend and Validators.
"""

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Dict, List, Optional

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
    JOB_STATUS_UPDATE = "job_status_update"
    ERROR = "error"

    # Client-Backend WebSocket messages
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    PING = "ping"
    PONG = "pong"
    SUBSCRIPTION_ACK = "subscription_ack"
    UNSUBSCRIPTION_ACK = "unsubscription_ack"
    EVENT = "event"


class EventType(StrEnum):
    """Types of real-time events that can be broadcast to clients."""

    # Job events
    JOB_CREATED = "job_created"
    JOB_STATUS_CHANGED = "job_status_changed"
    JOB_COMPLETED = "job_completed"

    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"

    # Episode events
    EPISODE_STARTED = "episode_started"
    EPISODE_STEP = "episode_step"
    EPISODE_COMPLETED = "episode_completed"

    # Competition events
    COMPETITION_CREATED = "competition_created"
    COMPETITION_UPDATED = "competition_updated"
    COMPETITION_ACTIVATED = "competition_activated"
    COMPETITION_DEACTIVATED = "competition_deactivated"

    # Submission events
    SUBMISSION_RECEIVED = "submission_received"

    # Validator events
    VALIDATOR_CONNECTED = "validator_connected"
    VALIDATOR_DISCONNECTED = "validator_disconnected"

    # Stats events
    STATS_UPDATED = "stats_updated"


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
    timeout_seconds: Optional[int] = None
    benchmark_spec: Optional[dict] = None
    artifact_url: Optional[str] = None
    artifact_expires_at: Optional[datetime] = None
    artifact_sha256: Optional[str] = None
    artifact_size_bytes: Optional[int] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def to_bytes(self) -> bytes:
        return self.model_dump_json().encode("utf-8")

    @classmethod
    def from_bytes(cls, data: bytes) -> "EvalJobMessage":
        return cls.model_validate_json(data)


class EvalResultMessage(SQLModel):
    """Message for sending evaluation results from validators to backend."""

    message_type: MessageType = MessageType.EVAL_RESULT
    status: EvaluationStatus = EvaluationStatus.COMPLETED
    job_id: SnowflakeId
    validator_hotkey: str
    miner_hotkey: str
    competition_id: str
    env_provider: str
    benchmark_name: str
    config: dict
    benchmark_spec: Optional[dict] = None
    score: float
    success_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    total_episodes: Optional[int] = None
    env_specs: Optional[List[Dict[str, Any]]] = None
    logs: Optional[str] = None
    error: Optional[str] = None
    extra_data: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobStatusUpdateMessage(SQLModel):
    """Message for notifying backend about job status changes."""

    message_type: MessageType = MessageType.JOB_STATUS_UPDATE
    job_id: SnowflakeId
    validator_hotkey: str
    status: EvaluationStatus
    detail: Optional[str] = None
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
    final_reward: float
    success: bool
    steps: int
    start_time: datetime
    end_time: datetime
    extra_metrics: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EpisodeStepDataMessage(SQLModel):
    """Message for sending episode step data from validators to backend."""

    message_type: MessageType = MessageType.EPISODE_STEP_DATA
    job_id: SnowflakeId
    env_name: str
    benchmark_name: str
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


# ============================================================================
# Client-Backend WebSocket Messages
# ============================================================================


class SubscriptionRequest(SQLModel):
    """Request to subscribe to specific events."""

    event_types: List[EventType]
    filters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    # Filter examples:
    # {"job_id": 123} - only events for specific job
    # {"competition_id": "abc"} - only events for specific competition
    # {"miner_hotkey": "5xyz..."} - only events for specific miner
    # {"validator_hotkey": "5abc..."} - only events for specific validator


class ClientMessage(SQLModel):
    """Base message from client to backend."""

    message_type: MessageType
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SubscribeMessage(ClientMessage):
    """Subscribe to events."""

    message_type: MessageType = MessageType.SUBSCRIBE
    subscription: SubscriptionRequest


class UnsubscribeMessage(ClientMessage):
    """Unsubscribe from events."""

    message_type: MessageType = MessageType.UNSUBSCRIBE
    subscription_id: str


class PingMessage(ClientMessage):
    """Ping message for keepalive."""

    message_type: MessageType = MessageType.PING


class ServerMessage(SQLModel):
    """Base message from backend to client."""

    message_type: MessageType
    request_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SubscriptionAckMessage(ServerMessage):
    """Acknowledgment of subscription."""

    message_type: MessageType = MessageType.SUBSCRIPTION_ACK
    subscription_id: str
    subscribed_events: List[EventType]


class UnsubscriptionAckMessage(ServerMessage):
    """Acknowledgment of unsubscription."""

    message_type: MessageType = MessageType.UNSUBSCRIPTION_ACK
    subscription_id: str


class PongMessage(ServerMessage):
    """Pong response to ping."""

    message_type: MessageType = MessageType.PONG


class EventMessage(ServerMessage):
    """Event notification to client."""

    message_type: MessageType = MessageType.EVENT
    event_type: EventType
    event_data: Dict[str, Any]
    subscription_id: Optional[str] = None
