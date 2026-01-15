"""
Orchestrator V2 - Executor-based task orchestration.

This is a refactored orchestrator that uses the TaskExecutor pattern
for pluggable task type support. It maintains backward compatibility
with the existing EvalJobMessage format while adding support for
the new TaskSpec-based approach.

Connects directly to the backend via WebSocket to receive jobs and
send results.
"""

from __future__ import annotations

import asyncio
import copy
import gc
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, Optional

import ray
from fiber.chain.chain_utils import load_hotkey_keypair
from snowflake import SnowflakeGenerator

from core.db.models import EvaluationStatus
from core.log import configure_logging, get_logger
from core.messages import EvalJobMessage, EvalResultMessage, JobStatusUpdateMessage
from core.tasks import (
    ExecutorNotFoundError,
    ResourceSpec,
    TaskResult,
    TaskSpec,
    TaskType,
)
from evaluator.backend_client import BackendClient
from evaluator.config import EvaluatorConfig
from evaluator.constants import (
    EVAL_TIMEOUT,
    MIN_CONCURRENT_JOBS,
)
from evaluator.containers import Containers, PodSchedulingError
from evaluator.executors import ExecutorRegistry, RLRolloutExecutor
from evaluator.log_uploader import EvaluationLogUploader
from validator.db.db_manager import DatabaseManager
from validator.db.models import EvaluationJob

logger = get_logger(__name__)


def _sanitize_for_json(value: Any) -> Any:
    """Best-effort conversion of Python objects to JSON-safe types."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _sanitize_for_json(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_sanitize_for_json(item) for item in value]
    try:
        dumped = value.model_dump()
        return _sanitize_for_json(dumped)
    except Exception:
        return str(value)


class OrchestratorV2:
    """
    Executor-based orchestrator for task evaluation.

    This orchestrator uses the TaskExecutor pattern to support multiple
    task types. It delegates setup, execution, and teardown to registered
    executors, allowing new task types to be added without modifying
    the orchestrator itself.

    Connects directly to the backend via WebSocket to receive jobs and
    send results.

    Usage:
        config = EvaluatorConfig()
        orchestrator = OrchestratorV2(config)
        await orchestrator.start()
    """

    def __init__(self, config: EvaluatorConfig):
        self.config = config
        logger.info("OrchestratorV2 initialized with db: %s", self.config.pg_database)

        # Backend client for direct WebSocket connection
        self.backend_client: Optional[BackendClient] = None

        # Database and identity
        self.db = DatabaseManager(self.config.pg_database)
        self.id_generator = SnowflakeGenerator(42)
        self.keypair = load_hotkey_keypair(
            wallet_name=config.settings["wallet_name"],
            hotkey_name=config.settings["hotkey_name"],
        )

        # Log uploader
        self.log_uploader: Optional[EvaluationLogUploader] = (
            EvaluationLogUploader(self.config.s3_config)
            if self.config.s3_config
            else None
        )

        # Concurrency control
        if config.max_concurrent_jobs < MIN_CONCURRENT_JOBS:
            logger.warning(
                "Configured max_concurrent_jobs (%s) below minimum (%s); clamping.",
                config.max_concurrent_jobs,
                MIN_CONCURRENT_JOBS,
            )
        self.max_concurrent_jobs = max(MIN_CONCURRENT_JOBS, config.max_concurrent_jobs)
        self.default_job_timeout = self._resolve_job_timeout(
            config.settings.get("job_timeout"), int(EVAL_TIMEOUT.total_seconds())
        )
        self.concurrent_slots = asyncio.Semaphore(self.max_concurrent_jobs)

        # Running jobs tracking
        self.running_jobs: Dict[str, Dict[str, Any]] = {}

        # Initialize Ray
        self._init_ray()

        # Register default executors
        self._register_default_executors()

        logger.info("OrchestratorV2 initialized with config: %s", self.config)

    def _init_ray(self) -> None:
        """Initialize Ray with explicit configuration."""
        if not ray.is_initialized():
            init_kwargs = {
                "num_cpus": self.config.ray_num_cpus,
                "num_gpus": self.config.ray_num_gpus,
                "logging_level": "info",
            }

            if self.config.ray_object_store_memory is not None:
                init_kwargs["object_store_memory"] = self.config.ray_object_store_memory

            if self.config.ray_memory is not None:
                init_kwargs["_memory"] = self.config.ray_memory

            ray.init(**init_kwargs)
            logger.info("Ray initialized with explicit configuration")
        else:
            logger.info("Ray already initialized")

    def _register_default_executors(self) -> None:
        """Register the default task executors."""
        # Register RL rollout executor
        rl_executor = RLRolloutExecutor(self.config)
        ExecutorRegistry.register(rl_executor)
        logger.info("Registered default executors: %s", ExecutorRegistry.list_types())

    def _init_backend_client(self) -> BackendClient:
        """Initialize the BackendClient for direct WebSocket connection."""
        backend_ws_url = getattr(self.config, "backend_ws_url", None)
        if not backend_ws_url:
            raise ValueError(
                "backend_ws_url not configured. Set backend_ws_url in evaluator config."
            )

        evaluator_id = getattr(self.config, "evaluator_id", None)
        if not evaluator_id:
            # Generate a default evaluator ID from hotkey
            evaluator_id = f"evaluator-{self.keypair.ss58_address[:16]}"
            logger.warning(
                "No evaluator_id configured, using generated ID: %s", evaluator_id
            )

        # Get supported task types from registered executors
        supported_task_types = [t.value for t in ExecutorRegistry.list_types()]

        return BackendClient(
            backend_url=backend_ws_url,
            evaluator_id=evaluator_id,
            supported_task_types=supported_task_types,
            max_concurrent_jobs=self.max_concurrent_jobs,
            capabilities={
                "ray_num_cpus": self.config.ray_num_cpus,
                "ray_num_gpus": self.config.ray_num_gpus,
                "hotkey": self.keypair.ss58_address,
            },
            on_job_received=self._handle_job_from_backend,
        )

    @staticmethod
    def _resolve_job_timeout(value: Any, fallback: int) -> int:
        """Return a positive integer timeout, falling back when unset/invalid."""
        try:
            timeout_seconds = int(value)
        except (TypeError, ValueError):
            return fallback

        if timeout_seconds <= 0:
            return fallback

        return timeout_seconds

    def _task_spec_from_eval_job(self, eval_job_msg: EvalJobMessage) -> TaskSpec:
        """Convert an EvalJobMessage to a TaskSpec.

        This provides backward compatibility with the existing message format.
        """
        # Extract config from benchmark_spec or config field
        if eval_job_msg.benchmark_spec:
            config_payload = copy.deepcopy(eval_job_msg.benchmark_spec)
        else:
            config_payload = (
                copy.deepcopy(eval_job_msg.config) if eval_job_msg.config else {}
            )

        # Determine timeout
        timeout_seconds = self._resolve_job_timeout(
            getattr(eval_job_msg, "timeout", None).total_seconds()
            if getattr(eval_job_msg, "timeout", None)
            else None,
            self.default_job_timeout,
        )

        return TaskSpec(
            task_type=TaskType.RL_ROLLOUT,  # Default to RL for backward compatibility
            task_id=str(eval_job_msg.job_id),
            config=config_payload,
            timeout=timedelta(seconds=timeout_seconds),
            resources=ResourceSpec(),  # Use defaults
            submission_id=eval_job_msg.submission_id,
            competition_id=eval_job_msg.competition_id,
            miner_hotkey=eval_job_msg.miner_hotkey,
            artifact_url=eval_job_msg.artifact_url or "",
            artifact_sha256=eval_job_msg.artifact_sha256,
            artifact_size_bytes=eval_job_msg.artifact_size_bytes,
            artifact_expires_at=eval_job_msg.artifact_expires_at,
            job_id=eval_job_msg.job_id,
            hf_repo_id=eval_job_msg.hf_repo_id,
            env_provider=eval_job_msg.env_provider,
            benchmark_name=eval_job_msg.benchmark_name,
        )

    def _create_evaluation_job_record(
        self, eval_job_msg: EvalJobMessage, timeout_seconds: int
    ) -> EvaluationJob:
        """Create an EvaluationJob database record from an EvalJobMessage."""
        config_payload = (
            copy.deepcopy(eval_job_msg.benchmark_spec)
            if eval_job_msg.benchmark_spec
            else {}
        )

        return EvaluationJob(
            id=eval_job_msg.job_id,
            competition_id=eval_job_msg.competition_id,
            submission_id=eval_job_msg.submission_id,
            miner_hotkey=eval_job_msg.miner_hotkey,
            hf_repo_id=eval_job_msg.hf_repo_id,
            env_provider=eval_job_msg.env_provider,
            benchmark_name=eval_job_msg.benchmark_name,
            config=config_payload,
            timeout_seconds=timeout_seconds,
            artifact_url=eval_job_msg.artifact_url,
            artifact_expires_at=eval_job_msg.artifact_expires_at,
            artifact_sha256=eval_job_msg.artifact_sha256,
            artifact_size_bytes=eval_job_msg.artifact_size_bytes,
            created_at=datetime.now(timezone.utc),
        )

    async def _handle_job_from_backend(self, eval_job_msg: EvalJobMessage) -> None:
        """
        Handle a job received via WebSocket from the backend.

        This method is called by the BackendClient when a new job is received.
        """
        logger.info(
            "Received job %s from backend via WebSocket for competition %s",
            eval_job_msg.job_id,
            eval_job_msg.competition_id,
        )

        # Process the job using existing infrastructure
        await self.process_job(eval_job_msg)

    async def setup_job(self, eval_job_msg: EvalJobMessage) -> Optional[Dict[str, Any]]:
        """Set up job infrastructure using the appropriate executor.

        Returns a job context dict for monitoring, or None if setup fails.
        """
        logger.info("Setting up job: %s", eval_job_msg.job_id)

        # Check for expired artifact
        if (
            eval_job_msg.artifact_url
            and eval_job_msg.artifact_expires_at
            and eval_job_msg.artifact_expires_at <= datetime.now(timezone.utc)
        ):
            logger.warning(
                "Received expired artifact URL for job %s (expires_at=%s)",
                eval_job_msg.job_id,
                eval_job_msg.artifact_expires_at,
            )

        # Convert to TaskSpec
        task_spec = self._task_spec_from_eval_job(eval_job_msg)

        # Get the appropriate executor
        try:
            executor = ExecutorRegistry.get(task_spec.task_type)
        except ExecutorNotFoundError as e:
            logger.error("No executor for task type %s: %s", task_spec.task_type, e)
            return None

        # Validate the task spec
        validation_errors = await executor.validate_spec(task_spec)
        if validation_errors:
            logger.error(
                "Task spec validation failed for job %s: %s",
                eval_job_msg.job_id,
                validation_errors,
            )
            await self._handle_validation_failure(eval_job_msg, validation_errors)
            return None

        # Create/update database record
        timeout_seconds = int(task_spec.timeout.total_seconds())
        evaluation_job = self._create_evaluation_job_record(
            eval_job_msg, timeout_seconds
        )

        existing_job = self.db.get_evaluation_job(eval_job_msg.job_id)
        if existing_job:
            logger.info(
                "Existing evaluation job record found for %s; updating config",
                eval_job_msg.job_id,
            )
            self.db.update_evaluation_job(
                eval_job_msg.job_id,
                {
                    "config": evaluation_job.config,
                    "timeout_seconds": timeout_seconds,
                },
            )
        else:
            try:
                self.db.create_evaluation_job(evaluation_job)
            except Exception as e:
                logger.error(
                    "Failed to create evaluation job %s in DB: %s",
                    eval_job_msg.job_id,
                    e,
                )
                return None

        # Update status to STARTING
        await self._update_job_status(
            eval_job_msg,
            EvaluationStatus.STARTING,
            "Evaluator is preparing the environment",
        )

        # Delegate setup to executor
        try:
            task_context = await executor.setup(task_spec)
        except Exception as e:
            logger.exception("Executor setup failed for job %s", eval_job_msg.job_id)
            await self._handle_setup_failure(eval_job_msg, str(e))
            return None

        # Update status to RUNNING
        await self._update_job_status(
            eval_job_msg,
            EvaluationStatus.RUNNING,
            "Evaluator started processing the job",
        )

        # Return job context for monitoring
        return {
            "job_id": eval_job_msg.job_id,
            "submission_id": eval_job_msg.submission_id,
            "eval_job_msg": eval_job_msg,
            "task_spec": task_spec,
            "task_context": task_context,
            "executor": executor,
            "start_time": datetime.now(timezone.utc),
            "timeout_seconds": timeout_seconds,
        }

    async def monitor_job(self, job_context: Dict[str, Any]) -> bool:
        """Monitor a running job and handle completion.

        Returns True if the job is complete (success, failure, or timeout).
        """
        job_id = job_context["job_id"]
        task_context = job_context["task_context"]
        executor = job_context["executor"]
        start_time = job_context["start_time"]
        timeout_seconds = job_context["timeout_seconds"]

        # Check for timeout
        elapsed = (datetime.now(timezone.utc) - start_time).total_seconds()
        if elapsed > timeout_seconds:
            logger.error("Job %s timed out after %.1f seconds", job_id, elapsed)
            await self._handle_timeout(job_context, elapsed, timeout_seconds)
            return True

        try:
            # Execute the task (this may return immediately if already complete)
            result = await executor.execute(task_context)

            if result.success:
                await self._handle_success(job_context, result)
            else:
                await self._handle_failure(job_context, result)

            return True

        except Exception as e:
            logger.exception("Error monitoring job %s", job_id)
            await self._handle_error(job_context, str(e))
            return True

    async def _handle_validation_failure(
        self, eval_job_msg: EvalJobMessage, errors: list[str]
    ) -> None:
        """Handle task spec validation failure."""
        error_message = f"Task validation failed: {'; '.join(errors)}"
        completed_at = datetime.now(timezone.utc)

        self.db.update_evaluation_job(
            eval_job_msg.job_id,
            {
                "status": EvaluationStatus.FAILED,
                "error_message": error_message,
                "completed_at": completed_at,
            },
        )

        await self._publish_failure_result(
            eval_job_msg=eval_job_msg,
            status=EvaluationStatus.FAILED,
            completed_at=completed_at,
            error_message=error_message,
        )

    async def _handle_setup_failure(
        self, eval_job_msg: EvalJobMessage, error: str
    ) -> None:
        """Handle executor setup failure."""
        completed_at = datetime.now(timezone.utc)
        error_message = f"Setup failed: {error}"

        self.db.update_evaluation_job(
            eval_job_msg.job_id,
            {
                "status": EvaluationStatus.FAILED,
                "error_message": error_message,
                "completed_at": completed_at,
            },
        )

        await self._publish_failure_result(
            eval_job_msg=eval_job_msg,
            status=EvaluationStatus.FAILED,
            completed_at=completed_at,
            error_message=error_message,
        )

    async def _handle_success(
        self, job_context: Dict[str, Any], result: TaskResult
    ) -> None:
        """Handle successful task completion."""
        eval_job_msg = job_context["eval_job_msg"]
        executor = job_context["executor"]
        task_context = job_context["task_context"]
        completed_at = datetime.now(timezone.utc)

        logger.info(
            "Job %s completed successfully: %s",
            job_context["job_id"],
            result.metrics,
        )

        # Update database
        self.db.update_evaluation_job(
            eval_job_msg.job_id,
            {
                "status": EvaluationStatus.COMPLETED,
                "completed_at": completed_at,
            },
        )

        # Publish result
        await self._publish_success_result(
            eval_job_msg=eval_job_msg,
            result=result,
            completed_at=completed_at,
        )

        # Teardown
        try:
            await executor.teardown(task_context)
        except Exception as e:
            logger.warning("Teardown failed for job %s: %s", job_context["job_id"], e)

    async def _handle_failure(
        self, job_context: Dict[str, Any], result: TaskResult
    ) -> None:
        """Handle task execution failure."""
        eval_job_msg = job_context["eval_job_msg"]
        executor = job_context["executor"]
        task_context = job_context["task_context"]
        completed_at = datetime.now(timezone.utc)

        logger.error("Job %s failed: %s", job_context["job_id"], result.error)

        # Update database
        self.db.update_evaluation_job(
            eval_job_msg.job_id,
            {
                "status": EvaluationStatus.FAILED,
                "error_message": result.error,
                "completed_at": completed_at,
            },
        )

        # Publish result
        await self._publish_failure_result(
            eval_job_msg=eval_job_msg,
            status=EvaluationStatus.FAILED,
            completed_at=completed_at,
            error_message=result.error or "Unknown error",
        )

        # Teardown
        try:
            await executor.teardown(task_context)
        except Exception as e:
            logger.warning("Teardown failed for job %s: %s", job_context["job_id"], e)

    async def _handle_timeout(
        self,
        job_context: Dict[str, Any],
        elapsed: float,
        timeout_seconds: int,
    ) -> None:
        """Handle task timeout."""
        eval_job_msg = job_context["eval_job_msg"]
        executor = job_context["executor"]
        task_context = job_context["task_context"]
        completed_at = datetime.now(timezone.utc)

        error_message = (
            f"Job timed out after {elapsed:.1f} seconds (limit {timeout_seconds}s)"
        )

        # Update database
        self.db.update_evaluation_job(
            eval_job_msg.job_id,
            {
                "status": EvaluationStatus.TIMEOUT,
                "error_message": error_message,
                "completed_at": completed_at,
            },
        )

        # Publish result
        await self._publish_failure_result(
            eval_job_msg=eval_job_msg,
            status=EvaluationStatus.TIMEOUT,
            completed_at=completed_at,
            error_message=error_message,
        )

        # Teardown
        try:
            await executor.teardown(task_context)
        except Exception as e:
            logger.warning("Teardown failed for job %s: %s", job_context["job_id"], e)

    async def _handle_error(self, job_context: Dict[str, Any], error: str) -> None:
        """Handle unexpected error during monitoring."""
        eval_job_msg = job_context["eval_job_msg"]
        executor = job_context["executor"]
        task_context = job_context["task_context"]
        completed_at = datetime.now(timezone.utc)

        # Update database
        self.db.update_evaluation_job(
            eval_job_msg.job_id,
            {
                "status": EvaluationStatus.FAILED,
                "error_message": error,
                "completed_at": completed_at,
            },
        )

        # Publish result
        await self._publish_failure_result(
            eval_job_msg=eval_job_msg,
            status=EvaluationStatus.FAILED,
            completed_at=completed_at,
            error_message=error,
        )

        # Teardown
        try:
            await executor.teardown(task_context)
        except Exception as e:
            logger.warning("Teardown failed for job %s: %s", job_context["job_id"], e)

    async def _update_job_status(
        self,
        eval_job_msg: EvalJobMessage,
        status: EvaluationStatus,
        detail: str,
    ) -> None:
        """Update job status in database and notify backend via WebSocket."""
        try:
            update_fields: Dict[str, Any] = {"status": status}
            if status == EvaluationStatus.STARTING:
                update_fields["started_at"] = datetime.now(timezone.utc)
                update_fields["error_message"] = None
                update_fields["completed_at"] = None

            self.db.update_evaluation_job(eval_job_msg.job_id, update_fields)
        except Exception as e:
            logger.error("Failed to update job status to %s: %s", status, e)
            return

        # Create and send status message via WebSocket
        status_msg = JobStatusUpdateMessage(
            job_id=eval_job_msg.job_id,
            validator_hotkey=self.keypair.ss58_address,
            status=status,
            detail=detail,
        )

        try:
            if self.backend_client:
                await self.backend_client.send_status_update(status_msg)
        except Exception as e:
            logger.error("Failed to send job status update: %s", e)

    async def _publish_success_result(
        self,
        eval_job_msg: EvalJobMessage,
        result: TaskResult,
        completed_at: datetime,
    ) -> None:
        """Publish a success result to the backend via WebSocket."""
        metrics = result.metrics

        # Extract benchmark spec
        spec_payload = (
            copy.deepcopy(eval_job_msg.benchmark_spec)
            if eval_job_msg.benchmark_spec
            else {}
        )
        base_config = spec_payload.get("config", {})

        eval_result_msg = EvalResultMessage(
            job_id=eval_job_msg.job_id,
            status=EvaluationStatus.COMPLETED,
            validator_hotkey=self.keypair.ss58_address,
            miner_hotkey=eval_job_msg.miner_hotkey,
            competition_id=eval_job_msg.competition_id,
            env_provider=eval_job_msg.env_provider,
            benchmark_name=eval_job_msg.benchmark_name,
            config=base_config,
            benchmark_spec=spec_payload,
            score=metrics.get("avg_reward", 0.0),
            success_rate=metrics.get("success_rate"),
            avg_reward=metrics.get("avg_reward"),
            total_episodes=result.total_episodes,
            logs=result.logs or "Evaluation completed successfully",
            error=None,
            extra_data={
                "summary": {
                    "metrics": metrics,
                    "duration_seconds": result.duration_seconds,
                    "completed_at": completed_at.isoformat(),
                }
            },
        )

        # Send via WebSocket
        if self.backend_client:
            await self.backend_client.send_result(eval_result_msg)

    async def _publish_failure_result(
        self,
        eval_job_msg: EvalJobMessage,
        status: EvaluationStatus,
        completed_at: datetime,
        error_message: str,
    ) -> None:
        """Publish a failure result to the backend via WebSocket."""
        spec_payload = (
            copy.deepcopy(eval_job_msg.benchmark_spec)
            if eval_job_msg.benchmark_spec
            else {}
        )
        base_config = spec_payload.get("config", {})

        eval_result_msg = EvalResultMessage(
            job_id=eval_job_msg.job_id,
            status=status,
            validator_hotkey=self.keypair.ss58_address,
            miner_hotkey=eval_job_msg.miner_hotkey,
            competition_id=eval_job_msg.competition_id,
            env_provider=eval_job_msg.env_provider,
            benchmark_name=eval_job_msg.benchmark_name,
            config=base_config,
            benchmark_spec=spec_payload,
            score=0.0,
            success_rate=None,
            avg_reward=None,
            total_episodes=None,
            logs=f"Evaluation {status.value.lower()}: {error_message}",
            error=error_message,
        )

        # Send via WebSocket
        if self.backend_client:
            await self.backend_client.send_result(eval_result_msg)

    async def process_job(self, eval_job_msg: EvalJobMessage) -> None:
        """Process a job asynchronously."""
        job_id = eval_job_msg.job_id

        if self.concurrent_slots.locked():
            logger.warning(
                "Max concurrent jobs (%s) reached. Job %s waiting for a free slot.",
                self.max_concurrent_jobs,
                job_id,
            )

        await self.concurrent_slots.acquire()

        try:
            job_context = await self.setup_job(eval_job_msg)
        except PodSchedulingError as e:
            logger.warning(
                "Insufficient cluster resources for job %s: %s",
                job_id,
                e,
            )
            self.concurrent_slots.release()
            return
        except Exception as e:
            logger.error("Failed to process job %s: %s", job_id, e)
            self.concurrent_slots.release()
            return

        if not job_context:
            logger.warning(
                "Setup for job %s returned no context; releasing slot.",
                job_id,
            )
            self.concurrent_slots.release()
            return

        self.running_jobs[job_id] = job_context
        logger.info(
            "Job %s added to running jobs. Total running: %s",
            job_id,
            len(self.running_jobs),
        )

    async def monitor_running_jobs(self) -> None:
        """Background task to monitor all running jobs."""
        while True:
            if self.running_jobs:
                completed_jobs = []
                for job_id, job_context in list(self.running_jobs.items()):
                    is_complete = await self.monitor_job(job_context)
                    if is_complete:
                        completed_jobs.append(job_id)

                # Remove completed jobs
                for job_id in completed_jobs:
                    job_context = self.running_jobs.pop(job_id, None)
                    if job_context is not None:
                        self.concurrent_slots.release()
                        job_context.clear()
                    logger.info(
                        "Job %s removed from running jobs. Remaining: %s",
                        job_id,
                        len(self.running_jobs),
                    )

            await asyncio.sleep(1)

    async def recover_stale_jobs(self) -> None:
        """Recover jobs that were running when orchestrator crashed."""
        logger.info("Checking for stale jobs from previous orchestrator run...")

        stale_statuses = [EvaluationStatus.STARTING, EvaluationStatus.RUNNING]
        for status in stale_statuses:
            stale_jobs = self.db.get_evaluation_jobs_by_status(status)
            logger.info("Found %s jobs in %s state", len(stale_jobs), status.value)

            for job in stale_jobs:
                logger.info(
                    "Resetting stale job %s (previous status %s) for restart",
                    job.id,
                    status.value,
                )
                try:
                    # Clean up any orphaned containers
                    if job.submission_id:
                        try:
                            containers = Containers()
                            containers.cleanup_container(
                                job.submission_id, job.id, wait=True
                            )
                        except Exception as e:
                            logger.warning(
                                "Failed to cleanup container for job %s: %s", job.id, e
                            )

                    # Reset job status
                    self.db.update_evaluation_job(
                        job.id,
                        {
                            "status": EvaluationStatus.QUEUED,
                            "error_message": None,
                            "started_at": None,
                            "completed_at": None,
                        },
                    )

                    logger.info("Stale job %s reset to QUEUED", job.id)

                except Exception as e:
                    logger.error("Failed to reset stale job %s: %s", job.id, e)
                    self.db.update_evaluation_job(
                        job.id,
                        {
                            "status": EvaluationStatus.FAILED,
                            "error_message": f"Failed to restart: {e}",
                            "completed_at": datetime.now(timezone.utc),
                        },
                    )

        logger.info("Stale job recovery completed")

    async def periodic_cleanup(self) -> None:
        """Periodic cleanup of orphaned containers and resources."""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                logger.info("Starting periodic cleanup...")

                containers = Containers()

                # Get completed/failed jobs and clean up old containers
                for status in [
                    EvaluationStatus.FAILED,
                    EvaluationStatus.TIMEOUT,
                    EvaluationStatus.COMPLETED,
                ]:
                    jobs = self.db.get_evaluation_jobs_by_status(status)
                    for job in jobs:
                        if job.completed_at and job.submission_id:
                            job_timeout = self._resolve_job_timeout(
                                getattr(job, "timeout_seconds", None),
                                self.default_job_timeout,
                            )
                            completed_time = job.completed_at
                            if completed_time.tzinfo is None:
                                completed_time = completed_time.replace(
                                    tzinfo=timezone.utc
                                )

                            age = (
                                datetime.now(timezone.utc) - completed_time
                            ).total_seconds()
                            if age > job_timeout:
                                try:
                                    containers.cleanup_container(
                                        job.submission_id, job.id
                                    )
                                except Exception:
                                    pass

                gc.collect()
                logger.info("Periodic cleanup completed")

            except Exception as e:
                logger.error("Error during periodic cleanup: %s", e)

    async def start(self) -> None:
        """Start the orchestrator with direct backend connection."""
        logger.info("Starting OrchestratorV2...")

        # Recover stale jobs
        await self.recover_stale_jobs()

        # Initialize the backend client
        self.backend_client = self._init_backend_client()

        # Start background tasks
        monitor_task = asyncio.create_task(self.monitor_running_jobs())
        logger.info("Started job monitoring task")

        cleanup_task = asyncio.create_task(self.periodic_cleanup())
        logger.info("Started periodic cleanup task")

        logger.info(
            "OrchestratorV2 connecting to backend (max concurrent: %s)...",
            self.max_concurrent_jobs,
        )

        try:
            # Connect and run - this blocks until stop() is called
            await self.backend_client.connect_and_run()
        finally:
            monitor_task.cancel()
            cleanup_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass
            try:
                await cleanup_task
            except asyncio.CancelledError:
                pass

    async def stop(self) -> None:
        """Stop the orchestrator."""
        logger.info("Stopping OrchestratorV2...")

        # Stop backend client
        if self.backend_client:
            try:
                await self.backend_client.stop()
                logger.info("Backend client stopped")
            except Exception as e:
                logger.error("Error stopping backend client: %s", e)
            self.backend_client = None

        # Clean up all running jobs
        for job_id in list(self.running_jobs.keys()):
            job_context = self.running_jobs.pop(job_id, None)
            if job_context is None:
                continue

            try:
                executor = job_context.get("executor")
                task_context = job_context.get("task_context")
                if executor and task_context:
                    try:
                        await executor.teardown(task_context)
                    except Exception as e:
                        logger.error("Error tearing down job %s: %s", job_id, e)
            except Exception as e:
                logger.error("Error cleaning up job %s: %s", job_id, e)
            finally:
                self.concurrent_slots.release()

        self.running_jobs.clear()

        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")


if __name__ == "__main__":
    evaluator_config = EvaluatorConfig()
    configure_logging(evaluator_config.log_file)
    orc = OrchestratorV2(evaluator_config)
    asyncio.run(orc.start())
