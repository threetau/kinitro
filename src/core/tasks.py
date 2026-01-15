"""
Core task abstraction layer for Kinitro.

This module defines the interfaces for task execution that decouple
evaluation logic from RL-specific assumptions. New task types can be
added by implementing the TaskExecutor protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Optional, Protocol, runtime_checkable

if TYPE_CHECKING:
    pass


class TaskType(StrEnum):
    """Enumeration of supported task types."""

    RL_ROLLOUT = "rl_rollout"
    # Future task types:
    # TRAINING_RUN = "training_run"


@dataclass
class ResourceSpec:
    """Resource requirements for task execution.

    Validators use this to schedule tasks on appropriate hardware
    and manage resource allocation.
    """

    cpu_cores: float = 1.0
    memory_mb: int = 2048
    gpu_count: int = 0
    gpu_memory_mb: int = 0
    storage_mb: int = 1024

    def __post_init__(self):
        if self.cpu_cores < 0:
            raise ValueError("cpu_cores must be non-negative")
        if self.memory_mb < 0:
            raise ValueError("memory_mb must be non-negative")
        if self.gpu_count < 0:
            raise ValueError("gpu_count must be non-negative")


@dataclass
class TaskSpec:
    """Base specification for any evaluable work unit.

    TaskSpec is the universal contract between the backend (which creates jobs)
    and validators (which execute them). All task-type-specific configuration
    is stored in the `config` dict.
    """

    task_type: TaskType
    task_id: str
    config: dict[str, Any]
    timeout: timedelta
    resources: ResourceSpec

    # Execution context
    submission_id: int
    competition_id: str
    miner_hotkey: str

    # Artifact information
    artifact_url: str
    artifact_sha256: Optional[str] = None
    artifact_size_bytes: Optional[int] = None
    artifact_expires_at: Optional[Any] = None  # datetime

    # Optional metadata
    job_id: Optional[int] = None
    hf_repo_id: Optional[str] = None
    env_provider: Optional[str] = None
    benchmark_name: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize TaskSpec to a dictionary for message passing."""
        return {
            "task_type": self.task_type.value,
            "task_id": self.task_id,
            "config": self.config,
            "timeout_seconds": self.timeout.total_seconds(),
            "resources": {
                "cpu_cores": self.resources.cpu_cores,
                "memory_mb": self.resources.memory_mb,
                "gpu_count": self.resources.gpu_count,
                "gpu_memory_mb": self.resources.gpu_memory_mb,
                "storage_mb": self.resources.storage_mb,
            },
            "submission_id": self.submission_id,
            "competition_id": self.competition_id,
            "miner_hotkey": self.miner_hotkey,
            "artifact_url": self.artifact_url,
            "artifact_sha256": self.artifact_sha256,
            "artifact_size_bytes": self.artifact_size_bytes,
            "job_id": self.job_id,
            "hf_repo_id": self.hf_repo_id,
            "env_provider": self.env_provider,
            "benchmark_name": self.benchmark_name,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskSpec":
        """Deserialize TaskSpec from a dictionary."""
        resources = ResourceSpec(**data.get("resources", {}))
        timeout_seconds = data.get("timeout_seconds", 3600)

        return cls(
            task_type=TaskType(data["task_type"]),
            task_id=data["task_id"],
            config=data.get("config", {}),
            timeout=timedelta(seconds=timeout_seconds),
            resources=resources,
            submission_id=data["submission_id"],
            competition_id=data["competition_id"],
            miner_hotkey=data["miner_hotkey"],
            artifact_url=data["artifact_url"],
            artifact_sha256=data.get("artifact_sha256"),
            artifact_size_bytes=data.get("artifact_size_bytes"),
            job_id=data.get("job_id"),
            hf_repo_id=data.get("hf_repo_id"),
            env_provider=data.get("env_provider"),
            benchmark_name=data.get("benchmark_name"),
        )


@dataclass
class TaskResult:
    """Result from task execution.

    TaskResult is the universal output format that all executors produce.
    Task-type-specific metrics are stored in the `metrics` dict.
    """

    task_id: str
    success: bool
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)  # name -> S3 key
    logs: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0

    # For RL tasks, we include episode-level details
    total_episodes: Optional[int] = None
    env_results: Optional[list[Any]] = None  # List of EnvResult for RL tasks

    def to_dict(self) -> dict[str, Any]:
        """Serialize TaskResult to a dictionary."""
        return {
            "task_id": self.task_id,
            "success": self.success,
            "metrics": self.metrics,
            "artifacts": self.artifacts,
            "logs": self.logs,
            "error": self.error,
            "duration_seconds": self.duration_seconds,
            "total_episodes": self.total_episodes,
        }


@dataclass
class TaskContext:
    """Execution context passed between setup/execute/teardown phases.

    This holds all the state needed for a task execution, including
    references to resources that need cleanup.
    """

    spec: TaskSpec
    work_dir: str
    env_vars: dict[str, str] = field(default_factory=dict)

    # Container and infrastructure references
    container_name: Optional[str] = None
    container_host: Optional[str] = None
    container_port: Optional[int] = None

    # Executor-specific state (e.g., RolloutCluster, RolloutWorker)
    state: dict[str, Any] = field(default_factory=dict)

    # Timing
    start_time: Optional[Any] = None  # datetime


@runtime_checkable
class TaskExecutor(Protocol):
    """Interface for executing different task types.

    Implement this protocol to add support for new task types.
    The executor lifecycle is:
    1. validate_spec() - Check if the task spec is valid
    2. setup() - Prepare execution environment
    3. execute() - Run the task
    4. teardown() - Clean up resources
    """

    @property
    def task_type(self) -> TaskType:
        """The task type this executor handles."""
        ...

    async def validate_spec(self, spec: TaskSpec) -> list[str]:
        """Validate a task specification.

        Args:
            spec: The task specification to validate

        Returns:
            List of validation error messages. Empty list means valid.
        """
        ...

    async def setup(self, spec: TaskSpec) -> TaskContext:
        """Prepare execution environment.

        This method should:
        - Create containers/pods
        - Initialize any required infrastructure
        - Return a TaskContext with all state needed for execution

        Args:
            spec: The task specification

        Returns:
            TaskContext with execution state
        """
        ...

    async def execute(self, context: TaskContext) -> TaskResult:
        """Run the task.

        This is the main execution method. It should:
        - Run the actual task logic
        - Collect metrics and results
        - Return a TaskResult

        Args:
            context: The execution context from setup()

        Returns:
            TaskResult with execution results
        """
        ...

    async def teardown(self, context: TaskContext) -> None:
        """Clean up resources.

        This method should release all resources allocated in setup(),
        including containers, workers, etc.

        Args:
            context: The execution context
        """
        ...


class ExecutorNotFoundError(Exception):
    """Raised when no executor is registered for a task type."""

    pass


class TaskValidationError(Exception):
    """Raised when task spec validation fails."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__(f"Task validation failed: {'; '.join(errors)}")
