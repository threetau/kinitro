import asyncio
import copy
import functools
import gc
import threading
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

import asyncpg
import ray
from fiber.chain.chain_utils import load_hotkey_keypair
from kubernetes import client, config
from pgqueuer import PgQueuer, Queries
from pgqueuer.db import AsyncpgDriver
from pgqueuer.models import Job
from ray.util.queue import Queue
from snowflake import SnowflakeGenerator

from core.db.models import EvaluationStatus
from core.log import configure_logging, get_logger
from core.messages import EvalJobMessage, EvalResultMessage, JobStatusUpdateMessage
from evaluator.config import EvaluatorConfig
from evaluator.constants import (
    EVAL_TIMEOUT,
    MIN_CONCURRENT_JOBS,
    POD_LOG_TAIL_LINES,
    PROCESS_JOB_WAIT_TIME,
    QUEUE_MAXSIZE,
    RAY_WAIT_TIMEOUT,
    RESOURCE_BACKOFF_SECONDS,
    WAIT_TIME,
)
from evaluator.containers import Containers, PodSchedulingError
from evaluator.log_uploader import EvaluationLogUploader
from evaluator.rollout import BenchmarkSpec, EnvManager, RolloutCluster
from evaluator.rollout.envs import EnvResult, EnvSpec
from evaluator.rpc.rpc_process import RPCProcess
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
        dumped = value.model_dump()  # type: ignore[attr-defined]
        return _sanitize_for_json(dumped)
    except Exception:
        return str(value)


class Orchestrator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        logger.info(f"Orchestrator initialized with db: {self.config.pg_database}")
        self.db = DatabaseManager(self.config.pg_database)
        self.id_generator = SnowflakeGenerator(42)
        self.keypair = load_hotkey_keypair(
            wallet_name=config.settings["wallet_name"],
            hotkey_name=config.settings["hotkey_name"],
        )

        self.log_uploader: Optional[EvaluationLogUploader] = (
            EvaluationLogUploader(self.config.s3_config)
            if self.config.s3_config
            else None
        )

        # Track running jobs for concurrent execution
        self.running_jobs: Dict[str, Dict] = {}  # job_id -> job_info
        if config.max_concurrent_jobs < MIN_CONCURRENT_JOBS:
            logger.warning(
                "Configured max_concurrent_jobs (%s) below minimum (%s); clamping.",
                config.max_concurrent_jobs,
                MIN_CONCURRENT_JOBS,
            )
        self.max_concurrent_jobs = max(MIN_CONCURRENT_JOBS, config.max_concurrent_jobs)
        self.default_job_timeout = self._resolve_job_timeout(
            config.settings.get("job_timeout"), EVAL_TIMEOUT
        )
        logger.info(
            "Default job timeout fallback set to %s seconds (backend may override)",
            self.default_job_timeout,
        )
        self.concurrent_slots = asyncio.Semaphore(self.max_concurrent_jobs)

        # Initialize Ray with explicit configuration
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

        logger.info(f"Orchestrator initialized with config: {self.config}")

    @staticmethod
    def _normalize_camera_names(value: Any) -> tuple[str, ...]:
        """Convert incoming camera name payloads into a tuple of strings."""
        if value is None:
            return tuple()
        if isinstance(value, str):
            return (value,)
        if isinstance(value, (list, tuple, set)):
            names: list[str] = []
            for item in value:
                if item is None:
                    continue
                names.append(str(item))
            return tuple(names)
        return tuple()

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

    def _split_config_data(
        self, config_value: Dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split a stored configuration into (spec, base_config) copies."""
        spec_copy = copy.deepcopy(config_value)
        try:
            base_config_source = config_value["config"]
        except KeyError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Benchmark spec is missing 'config'") from exc
        try:
            base_config = copy.deepcopy(dict(base_config_source))
        except TypeError as exc:  # pragma: no cover - defensive guard
            raise ValueError("Benchmark spec 'config' must be a mapping") from exc
        return spec_copy, base_config

    def _extract_job_spec_payloads(
        self, eval_job_msg: EvalJobMessage
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Return the benchmark spec payload and the underlying config dict."""
        if eval_job_msg.benchmark_spec is None:
            raise ValueError("EvalJobMessage is missing required benchmark_spec data")
        return self._split_config_data(eval_job_msg.benchmark_spec)

    def _config_payload_for_storage(self, eval_job_msg: EvalJobMessage) -> dict:
        """Choose the representation of the job configuration to persist locally."""
        spec_payload, _ = self._extract_job_spec_payloads(eval_job_msg)
        return spec_payload

    def _build_benchmark_spec_from_job(
        self, eval_job_msg: EvalJobMessage
    ) -> BenchmarkSpec:
        """Construct a BenchmarkSpec using job metadata and optional spec payload."""
        spec_payload, base_config = self._extract_job_spec_payloads(eval_job_msg)

        defaults = BenchmarkSpec.__dataclass_fields__
        default_render_mode = defaults["render_mode"].default
        default_camera_names = defaults["camera_names"].default
        default_camera_attribute = defaults["camera_attribute"].default

        render_mode = spec_payload.get("render_mode")
        camera_attribute = spec_payload.get("camera_attribute")
        camera_names = self._normalize_camera_names(spec_payload.get("camera_names"))

        return BenchmarkSpec(
            provider=eval_job_msg.env_provider,
            benchmark_name=eval_job_msg.benchmark_name,
            config=base_config,
            render_mode=render_mode or default_render_mode,
            camera_names=camera_names or default_camera_names,
            camera_attribute=camera_attribute or default_camera_attribute,
        )

    def _serialize_env_spec(self, env_spec: EnvSpec) -> Dict[str, Any]:
        """Convert an EnvSpec into a JSON-safe payload for storage."""
        return {
            "env_name": env_spec.env_name,
            "benchmark_name": env_spec.benchmark_name,
            "provider": env_spec.provider,
            "config": _sanitize_for_json(getattr(env_spec, "config", {})),
            "episodes_per_task": env_spec.episodes_per_task,
            "max_episode_steps": env_spec.max_episode_steps,
            "render_mode": env_spec.render_mode,
            "camera_attribute": env_spec.camera_attribute,
            "camera_names": list(env_spec.camera_names),
        }

    def _build_env_specs_payload(
        self,
        *,
        eval_job_msg: EvalJobMessage,
        env_results: Optional[List[EnvResult]] = None,
    ) -> List[Dict[str, Any]]:
        """Capture the environment specs used (or intended) for a job."""
        env_specs_payload: List[Dict[str, Any]] = []

        if env_results:
            for env_result in env_results:
                try:
                    env_specs_payload.append(
                        self._serialize_env_spec(env_result.env_spec)
                    )
                except Exception as exc:  # pragma: no cover - defensive guard
                    logger.warning(
                        "Failed to serialize env spec for job %s: %s",
                        eval_job_msg.job_id,
                        exc,
                    )

        if env_specs_payload:
            return env_specs_payload

        try:
            benchmark_spec = self._build_benchmark_spec_from_job(eval_job_msg)
            env_manager = EnvManager()
            derived_env_specs = env_manager.get_benchmark_envs(benchmark_spec)
            env_specs_payload = [
                self._serialize_env_spec(env_spec) for env_spec in derived_env_specs
            ]
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning(
                "Unable to derive env specs for job %s: %s",
                eval_job_msg.job_id,
                exc,
            )

        return env_specs_payload

    async def setup_job(self, job: Job) -> Optional[Dict]:
        """Setup job infrastructure and return job context for monitoring."""
        logger.info(f"Setting up job: {job.id}")
        if not job.payload:
            return None

        eval_job_msg = EvalJobMessage.from_bytes(job.payload)
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

        job_timeout_seconds = self._resolve_job_timeout(
            getattr(eval_job_msg, "timeout_seconds", None), self.default_job_timeout
        )

        job_config_payload = self._config_payload_for_storage(eval_job_msg)

        evaluation_job = EvaluationJob(
            id=eval_job_msg.job_id,
            competition_id=eval_job_msg.competition_id,
            submission_id=eval_job_msg.submission_id,
            miner_hotkey=eval_job_msg.miner_hotkey,
            hf_repo_id=eval_job_msg.hf_repo_id,
            env_provider=eval_job_msg.env_provider,
            benchmark_name=eval_job_msg.benchmark_name,
            config=job_config_payload,
            timeout_seconds=job_timeout_seconds,
            artifact_url=eval_job_msg.artifact_url,
            artifact_expires_at=eval_job_msg.artifact_expires_at,
            artifact_sha256=eval_job_msg.artifact_sha256,
            artifact_size_bytes=eval_job_msg.artifact_size_bytes,
            created_at=datetime.now(timezone.utc),
        )

        existing_job = self.db.get_evaluation_job(eval_job_msg.job_id)

        # Create job entry in the database with QUEUED status
        if existing_job:
            logger.info(
                "Existing evaluation job record found for %s; skipping creation",
                eval_job_msg.job_id,
            )
            try:
                self.db.update_evaluation_job(
                    eval_job_msg.job_id,
                    {
                        "config": job_config_payload,
                        "timeout_seconds": job_timeout_seconds,
                    },
                )
            except Exception as e:
                logger.warning(
                    "Failed to refresh stored timeout for job %s: %s",
                    eval_job_msg.job_id,
                    e,
                )
        else:
            try:
                self.db.create_evaluation_job(evaluation_job)
            except Exception as e:
                logger.error(
                    f"Failed to create evaluation job {eval_job_msg.job_id} in DB: {e}"
                )
                return None

        # Update status to STARTING
        try:
            self.db.update_evaluation_job(
                eval_job_msg.job_id,
                {
                    "status": EvaluationStatus.STARTING,
                    "started_at": datetime.now(timezone.utc),
                    "error_message": None,
                    "completed_at": None,
                },
            )
        except Exception as e:
            logger.error(f"Failed to update job status to STARTING: {e}")
        else:
            try:
                status_msg = JobStatusUpdateMessage(
                    job_id=eval_job_msg.job_id,
                    validator_hotkey=self.keypair.ss58_address,
                    status=EvaluationStatus.STARTING,
                    detail="Evaluator is preparing the environment",
                )
                await self.db.queue_job_status_update_msg(status_msg)
            except Exception as e:
                logger.error(
                    f"Failed to queue job status update for STARTING state: {e}"
                )

        if not eval_job_msg.artifact_url:
            raise RuntimeError(
                f"Job {eval_job_msg.job_id} missing artifact URL; cannot start container"
            )

        logger.info(
            "Creating container for job %s using artifact %s",
            eval_job_msg.job_id,
            eval_job_msg.artifact_url,
        )

        containers = Containers()
        cluster: Optional[RolloutCluster] = None
        worker: Optional[ray.actor.ActorHandle] = None
        worker_to_rpc_queue: Optional[Queue] = None
        rpc_to_worker_queue: Optional[Queue] = None
        container_ready = False

        try:
            pod = containers.create_container(
                eval_job_msg.submission_id,
                eval_job_msg.job_id,
                archive_url=eval_job_msg.artifact_url,
                archive_sha256=eval_job_msg.artifact_sha256,
            )
            container_ready = True
            logger.info(f"Created pod: {pod}")

            # Get NodePort and Node IP for direct TCP connection
            config.load_kube_config()
            k8v1api = client.CoreV1Api()
            v1 = client.CoreV1Api()
            service_name = pod
            svc = k8v1api.read_namespaced_service(service_name, "default")
            node_port = None
            for port in svc.spec.ports:
                if port.node_port:
                    node_port = port.node_port
                    break
            if not node_port:
                raise RuntimeError(f"No nodePort found for service {service_name}")

            # Get the first node's external IP (or internal if not available)
            nodes = v1.list_node().items
            node_ip = None
            for node in nodes:
                for addr in node.status.addresses:
                    if addr.type == "ExternalIP":
                        node_ip = addr.address
                        break
                if not node_ip:
                    for addr in node.status.addresses:
                        if addr.type == "InternalIP":
                            node_ip = addr.address
                            break
                if node_ip:
                    break
            if not node_ip:
                raise RuntimeError("No node IP found in cluster")

            # Wait for container to be ready
            await asyncio.sleep(WAIT_TIME)

            # Create a benchmark spec for the job, honoring backend-provided settings
            benchmark_spec = self._build_benchmark_spec_from_job(eval_job_msg)

            worker_to_rpc_queue = Queue(maxsize=QUEUE_MAXSIZE)
            rpc_to_worker_queue = Queue(maxsize=QUEUE_MAXSIZE)

            logger.info(
                f"Creating rollout cluster with config: {self.config.worker_remote_options}"
            )
            cluster = RolloutCluster(
                "eval-cluster",
                worker_remote_options=self.config.worker_remote_options,
            )
            worker = cluster.create_worker(
                eval_job_msg.job_id,
                [benchmark_spec],
                node_ip,
                node_port,
                eval_job_msg.submission_id,
                s3_config=self.config.s3_config,
                episode_log_interval=self.config.episode_log_interval,
                step_log_interval=self.config.step_log_interval,
                database_url=self.config.pg_database,
            )

            rpc_thread = threading.Thread(
                target=RPCProcess,
                args=(node_ip, node_port, rpc_to_worker_queue, worker_to_rpc_queue),
                daemon=True,
            )
            rpc_thread.start()

            await asyncio.sleep(PROCESS_JOB_WAIT_TIME)

            await self._wait_for_rpc_handshake(
                job_id=eval_job_msg.job_id,
                worker=worker,
                worker_to_rpc_queue=worker_to_rpc_queue,
                rpc_to_worker_queue=rpc_to_worker_queue,
            )
        except Exception:
            logger.exception(
                "Failed to prepare evaluation runtime for job %s",
                eval_job_msg.job_id,
            )
            self._teardown_failed_job_setup(
                job_id=eval_job_msg.job_id,
                submission_id=eval_job_msg.submission_id,
                cluster=cluster,
                worker=worker,
                worker_to_rpc_queue=worker_to_rpc_queue,
                rpc_to_worker_queue=rpc_to_worker_queue,
                containers=containers,
                container_ready=container_ready,
            )
            raise

        assert worker is not None
        assert cluster is not None
        assert worker_to_rpc_queue is not None
        assert rpc_to_worker_queue is not None

        # Update status to RUNNING
        try:
            self.db.update_evaluation_job(
                eval_job_msg.job_id, {"status": EvaluationStatus.RUNNING}
            )
        except Exception as e:
            logger.error(f"Failed to update job status to RUNNING: {e}")
        else:
            try:
                status_msg = JobStatusUpdateMessage(
                    job_id=eval_job_msg.job_id,
                    validator_hotkey=self.keypair.ss58_address,
                    status=EvaluationStatus.RUNNING,
                    detail="Evaluator started processing the job",
                )
                await self.db.queue_job_status_update_msg(status_msg)
            except Exception as e:
                logger.error(
                    f"Failed to queue job status update for RUNNING state: {e}"
                )

        # Start the evaluation (non-blocking)
        logger.info(f"Starting evaluation for job {eval_job_msg.job_id}")
        evaluation_future = worker.run_all_benchmark_tasks.remote(
            worker_to_rpc_queue, rpc_to_worker_queue
        )

        job_context = {
            "job_id": eval_job_msg.job_id,
            "submission_id": eval_job_msg.submission_id,
            "eval_job_msg": eval_job_msg,
            "worker": worker,
            "cluster": cluster,
            "evaluation_future": evaluation_future,
            "worker_to_rpc_queue": worker_to_rpc_queue,
            "rpc_to_worker_queue": rpc_to_worker_queue,
            "start_time": datetime.now(timezone.utc),
            "timeout_seconds": job_timeout_seconds,
        }

        logger.info(
            f"Created job context for {eval_job_msg.job_id} with queues: worker_to_rpc={worker_to_rpc_queue is not None}, rpc_to_worker={rpc_to_worker_queue is not None}"
        )

        return job_context

    async def _wait_for_rpc_handshake(
        self,
        *,
        job_id: Any,
        worker,
        worker_to_rpc_queue: Queue,
        rpc_to_worker_queue: Queue,
    ) -> None:
        """Ensure the RPC client can reach the submission container before proceeding."""
        max_attempts = max(1, getattr(self.config, "rpc_handshake_max_attempts", 5))
        retry_seconds = max(
            0.0, getattr(self.config, "rpc_handshake_retry_seconds", 2.0)
        )

        last_error = "no response received from RPC process"
        for attempt in range(1, max_attempts + 1):
            try:
                response = await worker.test_rpc.remote(
                    worker_to_rpc_queue, rpc_to_worker_queue
                )
            except Exception as exc:
                last_error = str(exc)
                logger.warning(
                    "RPC handshake attempt %d/%d for job %s failed: %s",
                    attempt,
                    max_attempts,
                    job_id,
                    exc,
                )
            else:
                if response and getattr(response, "success", False):
                    logger.info(
                        "RPC handshake succeeded for job %s on attempt %d",
                        job_id,
                        attempt,
                    )
                    return

                response_error = getattr(response, "error_message", None)
                last_error = response_error or "RPC response reported failure"
                logger.warning(
                    "RPC handshake attempt %d/%d for job %s reported error: %s",
                    attempt,
                    max_attempts,
                    job_id,
                    last_error,
                )

            if attempt < max_attempts and retry_seconds > 0:
                delay = retry_seconds * attempt
                logger.info("Retrying RPC handshake for job %s in %.1fs", job_id, delay)
                await asyncio.sleep(delay)

        raise RuntimeError(
            f"Unable to establish RPC connection for job {job_id} after {max_attempts} attempts: {last_error}"
        )

    def _teardown_failed_job_setup(
        self,
        *,
        job_id: Any,
        submission_id: Optional[Any],
        cluster: Optional[RolloutCluster],
        worker: Optional[Any],
        worker_to_rpc_queue: Optional[Queue],
        rpc_to_worker_queue: Optional[Queue],
        containers: Containers,
        container_ready: bool,
    ) -> None:
        """Best-effort cleanup for failures that occur before a job starts running."""
        if worker_to_rpc_queue is not None or rpc_to_worker_queue is not None:
            temp_context = {
                "job_id": job_id,
                "worker_to_rpc_queue": worker_to_rpc_queue,
                "rpc_to_worker_queue": rpc_to_worker_queue,
            }
            try:
                self._cleanup_queues(temp_context)
            except Exception as exc:
                logger.warning(
                    "Queue cleanup failed during setup teardown for job %s: %s",
                    job_id,
                    exc,
                )

        if cluster and worker:
            try:
                cluster.delete_worker(worker)
            except Exception as exc:
                logger.warning(
                    "Failed to delete rollout worker during setup teardown for job %s: %s",
                    job_id,
                    exc,
                )

        if submission_id is not None and container_ready:
            try:
                containers.cleanup_container(submission_id, job_id)
            except Exception as exc:
                logger.warning(
                    "Failed to cleanup container for job %s during setup teardown: %s",
                    job_id,
                    exc,
                )

        gc.collect()

    def _build_log_payload(
        self,
        *,
        job_context: Dict[str, Any],
        eval_job_msg: EvalJobMessage,
        status: EvaluationStatus,
        summary: Dict[str, Any],
        completed_at: datetime,
        error: Optional[str] = None,
        pod_logs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Create a structured payload describing an evaluation job."""
        start_time = job_context.get("start_time")
        if isinstance(start_time, datetime):
            duration_seconds = (completed_at - start_time).total_seconds()
            started_at = start_time
        else:
            duration_seconds = None
            started_at = None

        spec_payload, base_config = self._extract_job_spec_payloads(eval_job_msg)

        payload: Dict[str, Any] = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "job": {
                # NOTE: str these ids for JS compat.
                "id": str(job_context.get("job_id")),
                "submission_id": str(eval_job_msg.submission_id),
                "competition_id": eval_job_msg.competition_id,
                "miner_hotkey": eval_job_msg.miner_hotkey,
                "validator_hotkey": self.keypair.ss58_address,
                "hf_repo_id": eval_job_msg.hf_repo_id,
                "benchmark_name": eval_job_msg.benchmark_name,
                "env_provider": eval_job_msg.env_provider,
                "config": _sanitize_for_json(base_config),
                "status": status.value
                if isinstance(status, EvaluationStatus)
                else status,
                "started_at": _sanitize_for_json(started_at),
                "completed_at": completed_at.astimezone(timezone.utc).isoformat(),
            },
            "summary": _sanitize_for_json(summary),
        }

        payload["job"]["benchmark_spec"] = _sanitize_for_json(spec_payload)

        if duration_seconds is not None:
            payload["job"]["duration_seconds"] = duration_seconds

        if pod_logs:
            payload["pod_logs"] = _sanitize_for_json(pod_logs)

        if error:
            payload["error"] = error

        return payload

    async def _upload_job_log_bundle(
        self,
        *,
        job_context: Dict[str, Any],
        eval_job_msg: EvalJobMessage,
        status: EvaluationStatus,
        summary: Dict[str, Any],
        completed_at: datetime,
        error: Optional[str] = None,
        pod_logs: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Serialize and upload the job log bundle, returning storage metadata."""
        if not self.log_uploader:
            return None

        payload = self._build_log_payload(
            job_context=job_context,
            eval_job_msg=eval_job_msg,
            status=status,
            summary=summary,
            completed_at=completed_at,
            error=error,
            pod_logs=pod_logs,
        )

        loop = asyncio.get_running_loop()
        try:
            metadata = await loop.run_in_executor(
                None,
                functools.partial(
                    self.log_uploader.upload_log_bundle,
                    submission_id=eval_job_msg.submission_id,
                    job_id=eval_job_msg.job_id,
                    payload=payload,
                ),
            )
            return metadata
        except Exception:
            logger.exception(
                "Failed to upload evaluation log bundle for job %s",
                eval_job_msg.job_id,
            )
            return None

    async def _enqueue_job_for_processing(self, eval_job_msg: EvalJobMessage) -> None:
        """Requeue an evaluation job onto PgQueuer for processing."""
        conn = await asyncpg.connect(dsn=self.config.pg_database)
        try:
            driver = AsyncpgDriver(conn)
            q = Queries(driver)
            await q.enqueue(["add_job"], [eval_job_msg.to_bytes()], [0])
            logger.info("Requeued job %s onto add_job queue", eval_job_msg.job_id)
        except Exception:
            logger.exception(
                "Failed to enqueue job %s for restart", eval_job_msg.job_id
            )
            raise
        finally:
            await conn.close()

    def _cleanup_queues(self, job_context: Dict):
        """Clean up Ray Queue actors for a job."""
        job_id = job_context.get("job_id")

        # Track cleanup calls
        if not hasattr(self, "_cleanup_calls"):
            self._cleanup_calls = {}
        self._cleanup_calls[job_id] = self._cleanup_calls.get(job_id, 0) + 1

        logger.info(f"Cleanup call #{self._cleanup_calls[job_id]} for job {job_id}")

        # Debug: Log what's in job_context
        logger.info(f"Job context keys for job {job_id}: {list(job_context.keys())}")

        logger.info(f"Starting queue cleanup for job {job_id}")
        try:
            # Get queue actors
            worker_to_rpc = job_context.get("worker_to_rpc_queue")
            rpc_to_worker = job_context.get("rpc_to_worker_queue")

            logger.info(
                f"Retrieved from context - worker_to_rpc type: {type(worker_to_rpc)}, rpc_to_worker type: {type(rpc_to_worker)}"
            )

            # Shutdown queue actors
            if worker_to_rpc is not None:
                try:
                    if (
                        hasattr(worker_to_rpc, "actor")
                        and worker_to_rpc.actor is not None
                    ):
                        logger.info(
                            f"Shutting down worker_to_rpc queue for job {job_id}"
                        )
                        worker_to_rpc.shutdown(force=True)
                        logger.info(
                            f"Successfully shutdown worker_to_rpc queue for job {job_id}"
                        )
                    else:
                        logger.info(
                            f"worker_to_rpc queue already shutdown for job {job_id}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to shutdown worker_to_rpc queue: {e}")
            else:
                logger.warning(f"worker_to_rpc queue is None for job {job_id}")

            if rpc_to_worker is not None:
                try:
                    if (
                        hasattr(rpc_to_worker, "actor")
                        and rpc_to_worker.actor is not None
                    ):
                        logger.info(
                            f"Shutting down rpc_to_worker queue for job {job_id}"
                        )
                        rpc_to_worker.shutdown(force=True)
                        logger.info(
                            f"Successfully shutdown rpc_to_worker queue for job {job_id}"
                        )
                    else:
                        logger.info(
                            f"rpc_to_worker queue already shutdown for job {job_id}"
                        )
                except Exception as e:
                    logger.warning(f"Failed to shutdown rpc_to_worker queue: {e}")
            else:
                logger.warning(f"rpc_to_worker queue is None for job {job_id}")

            logger.info(f"Completed Ray Queue actors cleanup for job {job_id}")

        except Exception as e:
            logger.warning(f"Failed to cleanup queues: {e}")

    def _log_cached_runner_output(
        self, job_id: Any, pod_logs: Optional[Dict[str, Any]]
    ) -> None:
        """Log runner pod output for setup failures to aid debugging."""

        if not pod_logs:
            return

        containers = pod_logs.get("containers") or {}
        if not containers:
            error_detail = pod_logs.get("error") or pod_logs.get("warning")
            if error_detail:
                logger.error(
                    "Runner logs unavailable for job %s: %s",
                    job_id,
                    error_detail,
                )
            return

        source_note = (
            "cached before deletion"
            if pod_logs.get("cached_before_deletion")
            else "live fetch"
        )

        for container_name, entry in containers.items():
            log_text = entry.get("log")
            if log_text:
                logger.error(
                    "Runner logs for job %s (%s, %s):\n%s",
                    job_id,
                    container_name,
                    source_note,
                    log_text,
                )
            elif entry.get("error"):
                logger.error(
                    "Runner logs for job %s (%s, %s) unavailable: %s",
                    job_id,
                    container_name,
                    source_note,
                    entry.get("error"),
                )

    async def _publish_failure_result(
        self,
        *,
        eval_job_msg: EvalJobMessage,
        status: EvaluationStatus,
        completed_at: datetime,
        summary: Dict[str, Any],
        error_message: str,
        log_artifact: Optional[Dict[str, Any]],
        pod_logs: Optional[Dict[str, Any]],
    ) -> None:
        """Send an evaluation result message for failed or timed-out jobs."""

        spec_payload, base_config = self._extract_job_spec_payloads(eval_job_msg)
        env_specs_payload = self._build_env_specs_payload(
            eval_job_msg=eval_job_msg, env_results=None
        )
        extra_data: Dict[str, Any] = {"summary": summary}
        extra_data.setdefault("benchmark_spec", copy.deepcopy(spec_payload))

        logs_message = f"Evaluation {status.value.lower()}"
        if error_message:
            logs_message = f"{logs_message}: {error_message}"

        if log_artifact:
            extra_data["log_artifact"] = log_artifact
            artifact_ref = log_artifact.get("public_url") or log_artifact.get(
                "object_key"
            )
            if artifact_ref:
                logs_message = f"{logs_message}. Log bundle: {artifact_ref}"
        elif pod_logs:
            extra_data["pod_logs"] = pod_logs
            logs_message = f"{logs_message}. Container logs attached to result payload."

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
            env_specs=env_specs_payload or None,
            logs=logs_message,
            error=error_message,
            extra_data=extra_data,
        )

        await self.db.queue_evaluation_result_msg(eval_result_msg)

    async def monitor_job(self, job_context: Dict):
        """Monitor a running job and handle completion."""
        job_id = job_context["job_id"]
        submission_id = job_context["submission_id"]
        eval_job_msg = job_context["eval_job_msg"]
        evaluation_future = job_context["evaluation_future"]

        try:
            # Use ray.wait with timeout to check if job is done without blocking
            ready, not_ready = ray.wait([evaluation_future], timeout=RAY_WAIT_TIMEOUT)

            if ready:
                # Job completed, get results
                results: List[EnvResult] = ray.get(evaluation_future)

                logger.info(
                    f"Evaluation completed for job {job_id} with {len(results)} results"
                )

                # Calculate metrics
                if results:
                    total_episodes = sum(len(result.episodes) for result in results)
                    if total_episodes == 0:
                        total_episodes = None
                    avg_success_rate = sum(
                        result.success_rate for result in results
                    ) / len(results)
                    avg_reward = sum(result.mean_reward for result in results) / len(
                        results
                    )

                    logger.info(f"Job {job_id} - Total episodes: {total_episodes or 0}")
                    logger.info(
                        f"Job {job_id} - Average success rate: {avg_success_rate:.3f}"
                    )
                    logger.info(f"Job {job_id} - Average reward: {avg_reward:.3f}")
                else:
                    total_episodes = None
                    avg_success_rate = 0.0
                    avg_reward = 0.0

                completed_at = datetime.now(timezone.utc)
                summary_data: Dict[str, Any] = {
                    "results_count": len(results),
                    "total_episodes": total_episodes,
                    "avg_success_rate": avg_success_rate,
                    "avg_reward": avg_reward,
                    "completed_at": completed_at.isoformat(),
                }
                start_time = job_context.get("start_time")
                if isinstance(start_time, datetime):
                    summary_data["started_at"] = start_time.astimezone(
                        timezone.utc
                    ).isoformat()
                    summary_data["duration_seconds"] = (
                        completed_at - start_time
                    ).total_seconds()

                containers = Containers()
                pod_logs: Optional[Dict[str, Any]] = None
                try:
                    pod_logs = containers.collect_container_logs(
                        submission_id, job_id, tail_lines=POD_LOG_TAIL_LINES
                    )
                except Exception as log_exc:
                    logger.warning(
                        "Failed to collect pod logs for submission %s (job %s): %s",
                        submission_id,
                        job_id,
                        log_exc,
                    )

                log_artifact = await self._upload_job_log_bundle(
                    job_context=job_context,
                    eval_job_msg=eval_job_msg,
                    status=EvaluationStatus.COMPLETED,
                    summary=summary_data,
                    completed_at=completed_at,
                    error=None,
                    pod_logs=pod_logs,
                )

                # Update database
                try:
                    self.db.update_evaluation_job(
                        job_id,
                        {
                            "status": EvaluationStatus.COMPLETED,
                            "completed_at": completed_at,
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to update job status for job {job_id}: {e}")

                # Queue result message
                logs_message = "Evaluation completed successfully"
                if log_artifact:
                    artifact_ref = log_artifact.get("public_url") or log_artifact.get(
                        "object_key"
                    )
                    if artifact_ref:
                        logs_message = f"{logs_message}. Log bundle: {artifact_ref}"
                extra_data: Dict[str, Any] = {"summary": summary_data}
                if log_artifact:
                    extra_data["log_artifact"] = log_artifact
                elif pod_logs:
                    extra_data["pod_logs"] = pod_logs
                    logs_message = (
                        f"{logs_message}. Container logs attached to result payload."
                    )

                spec_payload, base_config = self._extract_job_spec_payloads(
                    eval_job_msg
                )
                extra_data.setdefault("benchmark_spec", copy.deepcopy(spec_payload))
                env_specs_payload = self._build_env_specs_payload(
                    eval_job_msg=eval_job_msg, env_results=results
                )

                eval_result_msg = EvalResultMessage(
                    job_id=job_id,
                    status=EvaluationStatus.COMPLETED,
                    validator_hotkey=self.keypair.ss58_address,
                    miner_hotkey=eval_job_msg.miner_hotkey,
                    competition_id=eval_job_msg.competition_id,
                    env_provider=eval_job_msg.env_provider,
                    benchmark_name=eval_job_msg.benchmark_name,
                    config=base_config,
                    benchmark_spec=spec_payload,
                    score=avg_reward,
                    success_rate=avg_success_rate,
                    avg_reward=avg_reward,
                    total_episodes=total_episodes,
                    env_specs=env_specs_payload or None,
                    logs=logs_message,
                    error=None,
                    extra_data=extra_data,
                )
                await self.db.queue_evaluation_result_msg(eval_result_msg)

                # Clean up Ray worker and container resources
                try:
                    # Clean up queues first
                    self._cleanup_queues(job_context)

                    # Clean up Ray worker
                    cluster = job_context.get("cluster")
                    worker = job_context.get("worker")
                    if cluster and worker:
                        # Call cleanup on the worker before killing it
                        try:
                            ray.get(worker.cleanup.remote(), timeout=5)
                        except Exception as e:
                            logger.warning(f"Worker cleanup failed: {e}")
                        cluster.delete_worker(worker)
                        logger.info(f"Cleaned up Ray worker for job {job_id}")

                    # Then clean up container
                    containers.cleanup_container(submission_id, job_id)
                    logger.info(
                        "Cleaned up container resources for submission %s (job %s)",
                        submission_id,
                        job_id,
                    )

                    # Clear references
                    del results
                    gc.collect()
                except Exception as e:
                    logger.error(
                        f"Failed to clean up resources for submission {submission_id}: {e}"
                    )

                return True  # Job completed

            # Check for timeout
            elapsed = (
                datetime.now(timezone.utc) - job_context["start_time"]
            ).total_seconds()
            timeout_seconds = job_context.get(
                "timeout_seconds", self.default_job_timeout
            )
            if elapsed > timeout_seconds:
                logger.error(f"Job {job_id} timed out after {elapsed} seconds")
                ray.cancel(evaluation_future)
                completed_at = datetime.now(timezone.utc)
                timeout_detail = (
                    f"Job timed out after {elapsed:.1f} seconds "
                    f"(limit {timeout_seconds}s)"
                )
                summary_data: Dict[str, Any] = {
                    "elapsed_seconds": elapsed,
                    "timeout_seconds": timeout_seconds,
                    "completed_at": completed_at.isoformat(),
                }
                start_time = job_context.get("start_time")
                if isinstance(start_time, datetime):
                    summary_data["started_at"] = start_time.astimezone(
                        timezone.utc
                    ).isoformat()
                    summary_data["duration_seconds"] = (
                        completed_at - start_time
                    ).total_seconds()

                containers = Containers()
                pod_logs: Optional[Dict[str, Any]] = None
                try:
                    pod_logs = containers.collect_container_logs(
                        submission_id, job_id, tail_lines=POD_LOG_TAIL_LINES
                    )
                except Exception as log_exc:
                    logger.warning(
                        "Failed to collect pod logs for timed out submission %s (job %s): %s",
                        submission_id,
                        job_id,
                        log_exc,
                    )

                log_artifact = await self._upload_job_log_bundle(
                    job_context=job_context,
                    eval_job_msg=eval_job_msg,
                    status=EvaluationStatus.TIMEOUT,
                    summary=summary_data,
                    completed_at=completed_at,
                    error=timeout_detail,
                    pod_logs=pod_logs,
                )

                error_message = timeout_detail
                if log_artifact:
                    artifact_ref = log_artifact.get("public_url") or log_artifact.get(
                        "object_key"
                    )
                    if artifact_ref:
                        error_message = f"{error_message}. Log bundle: {artifact_ref}"
                elif pod_logs:
                    error_message = (
                        f"{error_message}. Container logs attached to result payload."
                    )

                await self._publish_failure_result(
                    eval_job_msg=eval_job_msg,
                    status=EvaluationStatus.TIMEOUT,
                    completed_at=completed_at,
                    summary=summary_data,
                    error_message=timeout_detail,
                    log_artifact=log_artifact,
                    pod_logs=pod_logs,
                )

                self.db.update_evaluation_job(
                    job_id,
                    {
                        "status": EvaluationStatus.TIMEOUT,
                        "error_message": error_message,
                        "completed_at": completed_at,
                    },
                )
                try:
                    status_msg = JobStatusUpdateMessage(
                        job_id=eval_job_msg.job_id,
                        validator_hotkey=self.keypair.ss58_address,
                        status=EvaluationStatus.TIMEOUT,
                        detail=error_message,
                    )
                    await self.db.queue_job_status_update_msg(status_msg)
                except Exception as status_err:
                    logger.warning(
                        "Failed to queue timeout status update for job %s: %s",
                        eval_job_msg.job_id,
                        status_err,
                    )

                # Clean up Ray worker and container resources on timeout
                try:
                    # Clean up queues first
                    self._cleanup_queues(job_context)

                    # Clean up Ray worker
                    cluster = job_context.get("cluster")
                    worker = job_context.get("worker")
                    if cluster and worker:
                        # Try to call cleanup on the worker before killing it
                        try:
                            ray.get(worker.cleanup.remote(), timeout=2)
                        except Exception as e:
                            logger.warning(f"Worker cleanup failed on timeout: {e}")
                        cluster.delete_worker(worker)
                        logger.info(f"Cleaned up Ray worker for timed out job {job_id}")

                    # Then clean up container
                    containers.cleanup_container(submission_id, job_id)
                    logger.info(
                        "Cleaned up container resources for timed out submission %s (job %s)",
                        submission_id,
                        job_id,
                    )

                    gc.collect()
                except Exception as e:
                    logger.error(
                        f"Failed to clean up resources for timed out submission {submission_id}: {e}"
                    )

                return True  # Job timed out

            return False  # Job still running

        except Exception as e:
            logger.error(f"Error monitoring job {job_id}: {e}")
            completed_at = datetime.now(timezone.utc)
            error_detail = str(e)
            summary_data: Dict[str, Any] = {
                "exception_type": type(e).__name__,
                "message": error_detail,
                "completed_at": completed_at.isoformat(),
            }
            start_time = job_context.get("start_time")
            if isinstance(start_time, datetime):
                summary_data["started_at"] = start_time.astimezone(
                    timezone.utc
                ).isoformat()
                summary_data["duration_seconds"] = (
                    completed_at - start_time
                ).total_seconds()

            containers = Containers()
            pod_logs: Optional[Dict[str, Any]] = None
            try:
                pod_logs = containers.collect_container_logs(
                    submission_id, job_id, tail_lines=POD_LOG_TAIL_LINES
                )
            except Exception as log_exc:
                logger.warning(
                    "Failed to collect pod logs for failed submission %s (job %s): %s",
                    submission_id,
                    job_id,
                    log_exc,
                )

            log_artifact = await self._upload_job_log_bundle(
                job_context=job_context,
                eval_job_msg=eval_job_msg,
                status=EvaluationStatus.FAILED,
                summary=summary_data,
                completed_at=completed_at,
                error=error_detail,
                pod_logs=pod_logs,
            )

            error_message = error_detail
            if log_artifact:
                artifact_ref = log_artifact.get("public_url") or log_artifact.get(
                    "object_key"
                )
                if artifact_ref:
                    error_message = f"{error_message}. Log bundle: {artifact_ref}"
            elif pod_logs:
                error_message = (
                    f"{error_message}. Container logs attached to result payload."
                )

            await self._publish_failure_result(
                eval_job_msg=eval_job_msg,
                status=EvaluationStatus.FAILED,
                completed_at=completed_at,
                summary=summary_data,
                error_message=error_detail,
                log_artifact=log_artifact,
                pod_logs=pod_logs,
            )

            self.db.update_evaluation_job(
                job_id,
                {
                    "status": EvaluationStatus.FAILED,
                    "error_message": error_message,
                    "completed_at": completed_at,
                },
            )

            # Clean up Ray worker and container resources on error
            try:
                # Clean up queues first
                self._cleanup_queues(job_context)

                # Clean up Ray worker
                cluster = job_context.get("cluster")
                worker = job_context.get("worker")
                if cluster and worker:
                    # Try to call cleanup on the worker before killing it
                    try:
                        ray.get(worker.cleanup.remote(), timeout=2)
                    except Exception as e:
                        logger.warning(f"Worker cleanup failed on error: {e}")
                    cluster.delete_worker(worker)
                    logger.info(f"Cleaned up Ray worker for failed job {job_id}")

                # Then clean up container
                containers.cleanup_container(submission_id, job_id)
                logger.info(
                    "Cleaned up container resources for failed submission %s (job %s)",
                    submission_id,
                    job_id,
                )

                gc.collect()
            except Exception as ex:
                logger.error(
                    f"Failed to clean up resources for failed submission {submission_id}: {ex}"
                )

            return True  # Job failed

    async def process_job(self, job: Job):
        """Process a job asynchronously without blocking."""
        if self.concurrent_slots.locked():
            logger.warning(
                f"Max concurrent jobs ({self.max_concurrent_jobs}) reached. Job {getattr(job, 'id', 'unknown')} waiting for a free slot."
            )

        await self.concurrent_slots.acquire()

        try:
            job_context = await self.setup_job(job)
        except PodSchedulingError as e:
            job_id = getattr(job, "id", "unknown")
            logger.warning("Insufficient cluster resources for job %s: %s", job_id, e)

            if job.payload:
                try:
                    eval_job_msg = EvalJobMessage.from_bytes(job.payload)
                    backoff_detail = (
                        f"Waiting for cluster capacity: {e}" if str(e) else None
                    )
                    self.db.update_evaluation_job(
                        eval_job_msg.job_id,
                        {
                            "status": EvaluationStatus.QUEUED,
                            "error_message": backoff_detail,
                            "started_at": None,
                            "completed_at": None,
                        },
                    )

                    status_msg = JobStatusUpdateMessage(
                        job_id=eval_job_msg.job_id,
                        validator_hotkey=self.keypair.ss58_address,
                        status=EvaluationStatus.QUEUED,
                        detail=backoff_detail,
                    )
                    await self.db.queue_job_status_update_msg(status_msg)

                    logger.info(
                        "Requeueing job %s after resource backoff", eval_job_msg.job_id
                    )
                    await asyncio.sleep(RESOURCE_BACKOFF_SECONDS)
                    await self._enqueue_job_for_processing(eval_job_msg)
                except Exception as requeue_err:
                    logger.error(
                        "Failed to requeue job %s after resource shortfall: %s",
                        job_id,
                        requeue_err,
                    )

            self.concurrent_slots.release()
            return
        except Exception as e:
            job_id = getattr(job, "id", "unknown")
            logger.error(f"Failed to process job {job_id}: {e}")

            completed_at = datetime.now(timezone.utc)
            eval_job_msg: Optional[EvalJobMessage] = None
            pod_logs: Optional[Dict[str, Any]] = None
            log_artifact: Optional[Dict[str, Any]] = None
            error_detail = str(e)

            if job.payload:
                try:
                    eval_job_msg = EvalJobMessage.from_bytes(job.payload)
                except Exception as decode_err:
                    logger.error(
                        "Failed to decode job payload for job %s: %s",
                        job_id,
                        decode_err,
                    )

            if eval_job_msg is not None:
                containers = Containers()
                try:
                    pod_logs = containers.collect_container_logs(
                        eval_job_msg.submission_id,
                        eval_job_msg.job_id,
                        tail_lines=POD_LOG_TAIL_LINES,
                    )
                except Exception as log_exc:
                    logger.warning(
                        "Failed to collect pod logs for setup failure %s: %s",
                        eval_job_msg.job_id,
                        log_exc,
                    )
                else:
                    self._log_cached_runner_output(eval_job_msg.job_id, pod_logs)

                summary_data: Dict[str, Any] = {
                    "exception_type": type(e).__name__,
                    "message": error_detail,
                    "stage": "setup_job",
                    "completed_at": completed_at.isoformat(),
                }
                stub_context = {"job_id": eval_job_msg.job_id, "start_time": None}
                log_artifact = await self._upload_job_log_bundle(
                    job_context=stub_context,
                    eval_job_msg=eval_job_msg,
                    status=EvaluationStatus.FAILED,
                    summary=summary_data,
                    completed_at=completed_at,
                    error=error_detail,
                    pod_logs=pod_logs,
                )

                await self._publish_failure_result(
                    eval_job_msg=eval_job_msg,
                    status=EvaluationStatus.FAILED,
                    completed_at=completed_at,
                    summary=summary_data,
                    error_message=error_detail,
                    log_artifact=log_artifact,
                    pod_logs=pod_logs,
                )

            # Attempt to mark the evaluation job as failed and notify backend
            try:
                if eval_job_msg is not None:
                    failure_detail = f"Container setup failed: {e}"
                    if log_artifact:
                        artifact_ref = log_artifact.get(
                            "public_url"
                        ) or log_artifact.get("object_key")
                        if artifact_ref:
                            failure_detail = (
                                f"{failure_detail}. Log bundle: {artifact_ref}"
                            )
                    elif pod_logs:
                        failure_detail = f"{failure_detail}. Container logs attached to result payload."

                    self.db.update_evaluation_job(
                        eval_job_msg.job_id,
                        {
                            "status": EvaluationStatus.FAILED,
                            "error_message": failure_detail,
                            "completed_at": completed_at,
                        },
                    )

                    status_msg = JobStatusUpdateMessage(
                        job_id=eval_job_msg.job_id,
                        validator_hotkey=self.keypair.ss58_address,
                        status=EvaluationStatus.FAILED,
                        detail=failure_detail,
                    )
                    await self.db.queue_job_status_update_msg(status_msg)
            except Exception as status_err:
                logger.error(
                    "Failed to publish failure status for job %s: %s",
                    job_id,
                    status_err,
                )

            self.concurrent_slots.release()
            return

        if not job_context:
            logger.warning(
                f"Setup for job {getattr(job, 'id', 'unknown')} returned no context; releasing slot."
            )
            self.concurrent_slots.release()
            return

        job_id = job_context["job_id"]
        self.running_jobs[job_id] = job_context
        logger.info(
            f"Job {job_id} added to running jobs. Total running: {len(self.running_jobs)}"
        )

    async def monitor_running_jobs(self):
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
                        # Clear references to heavy objects
                        job_context.clear()
                    logger.info(
                        f"Job {job_id} removed from running jobs. Remaining: {len(self.running_jobs)}"
                    )

            await asyncio.sleep(1)  # Check every second

    async def recover_stale_jobs(self):
        """Recover jobs that were running when orchestrator crashed."""
        logger.info("Checking for stale jobs from previous orchestrator run...")

        # Find jobs that are stuck in STARTING or RUNNING state
        stale_statuses = [EvaluationStatus.STARTING, EvaluationStatus.RUNNING]
        for status in stale_statuses:
            stale_jobs = self.db.get_evaluation_jobs_by_status(status)
            logger.info(f"Found {len(stale_jobs)} jobs in {status.value} state")
            for job in stale_jobs:
                logger.info(
                    "Resetting stale job %s (previous status %s) for restart",
                    job.id,
                    status.value,
                )
                try:
                    if job.submission_id:
                        try:
                            containers = Containers()
                            containers.cleanup_container(
                                job.submission_id, job.id, wait=True
                            )
                            logger.info(
                                "Cleaned up orphaned container for submission %s (job %s)",
                                job.submission_id,
                                job.id,
                            )
                        except Exception as cleanup_err:
                            logger.warning(
                                "Failed to cleanup container for submission %s (job %s): %s",
                                job.submission_id,
                                job.id,
                                cleanup_err,
                            )

                    self.db.update_evaluation_job(
                        job.id,
                        {
                            "status": EvaluationStatus.QUEUED,
                            "error_message": None,
                            "started_at": None,
                            "completed_at": None,
                        },
                    )

                    status_msg = JobStatusUpdateMessage(
                        job_id=job.id,
                        validator_hotkey=self.keypair.ss58_address,
                        status=EvaluationStatus.QUEUED,
                        detail="Evaluator reset job after orchestrator restart",
                    )
                    try:
                        await self.db.queue_job_status_update_msg(status_msg)
                    except Exception as status_err:
                        logger.warning(
                            "Failed to queue status update for job %s: %s",
                            job.id,
                            status_err,
                        )

                    job_config_payload = (
                        copy.deepcopy(job.config)
                        if isinstance(job.config, dict)
                        else {}
                    )
                    timeout_seconds = self._resolve_job_timeout(
                        getattr(job, "timeout_seconds", None),
                        self.default_job_timeout,
                    )

                    spec_payload, base_config = self._split_config_data(
                        job_config_payload
                    )

                    eval_job_msg = EvalJobMessage(
                        job_id=job.id,
                        competition_id=job.competition_id,
                        submission_id=job.submission_id,
                        miner_hotkey=job.miner_hotkey,
                        hf_repo_id=job.hf_repo_id,
                        env_provider=job.env_provider,
                        benchmark_name=job.benchmark_name,
                        config=base_config,
                        benchmark_spec=spec_payload,
                        artifact_url=job.artifact_url,
                        artifact_expires_at=job.artifact_expires_at,
                        artifact_sha256=job.artifact_sha256,
                        artifact_size_bytes=job.artifact_size_bytes,
                        timeout_seconds=timeout_seconds,
                    )

                    await self._enqueue_job_for_processing(eval_job_msg)
                    logger.info("Stale job %s successfully reset and requeued", job.id)
                except Exception as exc:
                    logger.error("Failed to reset stale job %s: %s", job.id, exc)
                    self.db.update_evaluation_job(
                        job.id,
                        {
                            "status": EvaluationStatus.FAILED,
                            "error_message": f"Failed to restart after orchestrator crash: {exc}",
                            "completed_at": datetime.now(timezone.utc),
                        },
                    )

        logger.info("Stale job recovery completed")

    async def periodic_cleanup(self):
        """Periodic cleanup of orphaned containers and resources."""
        while True:
            try:
                await asyncio.sleep(600)  # Run every 10 minutes
                logger.info("Starting periodic cleanup...")

                containers = Containers()

                # Get all completed/failed jobs from the last 24 hours
                # that might have orphaned containers
                failed_jobs = self.db.get_evaluation_jobs_by_status(
                    EvaluationStatus.FAILED
                )
                timeout_jobs = self.db.get_evaluation_jobs_by_status(
                    EvaluationStatus.TIMEOUT
                )
                completed_jobs = self.db.get_evaluation_jobs_by_status(
                    EvaluationStatus.COMPLETED
                )

                # Clean up containers for old completed/failed jobs
                for job_list in [failed_jobs, timeout_jobs, completed_jobs]:
                    for job in job_list:
                        if job.completed_at:
                            job_timeout_seconds = self._resolve_job_timeout(
                                getattr(job, "timeout_seconds", None),
                                self.default_job_timeout,
                            )

                            # Ensure both datetimes have timezone info for comparison
                            current_time = datetime.now(timezone.utc)
                            completed_time = job.completed_at
                            if completed_time.tzinfo is None:
                                completed_time = completed_time.replace(
                                    tzinfo=timezone.utc
                                )

                            if (
                                current_time - completed_time
                            ).total_seconds() > job_timeout_seconds:
                                try:
                                    if job.submission_id:
                                        containers.cleanup_container(
                                            job.submission_id, job.id
                                        )
                                        logger.debug(
                                            "Cleaned up container for old job %s (submission %s)",
                                            job.id,
                                            job.submission_id,
                                        )
                                except Exception as e:
                                    logger.debug(
                                        f"Container cleanup failed for job {job.id}: {e}"
                                    )

                gc.collect()

                logger.info("Periodic cleanup completed")
            except Exception as e:
                logger.error(f"Error during periodic cleanup: {e}")

    async def start(self):
        logger.info("Starting orchestrator...")

        # Recover any stale jobs from previous run
        await self.recover_stale_jobs()

        conn = await asyncpg.connect(dsn=self.config.pg_database)

        driver = AsyncpgDriver(conn)
        pgq = PgQueuer(driver)

        # Start the job monitoring task
        monitor_task = asyncio.create_task(self.monitor_running_jobs())
        logger.info("Started job monitoring task")

        # Start periodic cleanup task
        cleanup_task = asyncio.create_task(self.periodic_cleanup())
        logger.info("Started periodic cleanup task")

        @pgq.entrypoint("add_job")
        async def process(job: Job) -> None:
            asyncio.create_task(self.process_job(job))
            logger.info(f"Job {job.id} added to processing queue.")

        logger.info(
            f"Orchestrator is now listening for jobs (max concurrent: {self.max_concurrent_jobs})..."
        )

        try:
            await pgq.run()
        finally:
            monitor_task.cancel()
            cleanup_task.cancel()

        await asyncio.Future()

    def stop(self):
        logger.info("Stopping orchestrator...")
        # Clean up all running jobs
        for job_id in list(self.running_jobs.keys()):
            job_context = self.running_jobs.pop(job_id, None)
            if job_context is None:
                continue
            try:
                # Clean up queues first
                self._cleanup_queues(job_context)

                cluster = job_context.get("cluster")
                worker = job_context.get("worker")
                if cluster and worker:
                    # Try to call cleanup on the worker before killing it
                    try:
                        ray.get(worker.cleanup.remote(), timeout=2)
                    except Exception as e:
                        logger.warning(f"Worker cleanup failed during shutdown: {e}")
                    cluster.delete_worker(worker)
                submission_id = job_context.get("submission_id")
                if submission_id:
                    containers = Containers()
                    containers.cleanup_container(submission_id, job_id)
            except Exception as e:
                logger.error(f"Error cleaning up job {job_id}: {e}")
            finally:
                self.concurrent_slots.release()

        self.running_jobs.clear()

        # Shutdown Ray if initialized
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown complete")


if __name__ == "__main__":
    evaluator_config = EvaluatorConfig()
    configure_logging(evaluator_config.log_file)
    orc = Orchestrator(evaluator_config)
    asyncio.run(orc.start())
