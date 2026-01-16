"""
RL Rollout task executor.

This executor wraps the existing RolloutWorker and RolloutCluster
infrastructure to execute RL rollout tasks through the TaskExecutor interface.
"""

from __future__ import annotations

import asyncio
import gc
import threading
import time
from datetime import datetime, timezone
from typing import Any, List, Optional

import ray
from kubernetes import client, config
from ray.util.queue import Queue

from core.db.models import SnowflakeId
from core.log import get_logger
from core.tasks import (
    TaskContext,
    TaskResult,
    TaskSpec,
    TaskType,
)
from evaluator.config import EvaluatorConfig
from evaluator.constants import (
    PROCESS_JOB_WAIT_TIME,
    QUEUE_MAXSIZE,
    RAY_WAIT_TIMEOUT,
    WAIT_TIME,
)
from evaluator.containers import Containers
from evaluator.providers.registry import ProviderRegistry
from evaluator.rollout import BenchmarkSpec, EnvManager, RolloutCluster
from evaluator.rollout.envs import EnvResult
from evaluator.rpc.rpc_process import RPCProcess

logger = get_logger(__name__)


class RLRolloutExecutor:
    """Executor for RL rollout tasks.

    This executor wraps the existing RolloutWorker/RolloutCluster infrastructure
    to provide RL evaluation through the TaskExecutor interface.

    The executor handles:
    - Creating Kubernetes pods for miner submissions
    - Setting up Ray workers for parallel evaluation
    - Running rollout episodes across multiple environments
    - Collecting and aggregating results
    """

    task_type = TaskType.RL_ROLLOUT

    def __init__(
        self,
        evaluator_config: EvaluatorConfig,
        provider_registry: Optional[ProviderRegistry] = None,
    ):
        """Initialize the RL rollout executor.

        Args:
            evaluator_config: Configuration for the evaluator
            provider_registry: Optional provider registry (uses global if not provided)
        """
        self.config = evaluator_config
        self.provider_registry = provider_registry or ProviderRegistry
        self.env_manager = EnvManager()

        # Default timeouts and settings
        self.rpc_handshake_max_attempts = getattr(
            evaluator_config, "rpc_handshake_max_attempts", 5
        )
        self.rpc_handshake_retry_seconds = getattr(
            evaluator_config, "rpc_handshake_retry_seconds", 2.0
        )

    async def validate_spec(self, spec: TaskSpec) -> list[str]:
        """Validate an RL rollout task specification.

        Checks:
        - Required fields are present (artifact_url, env_provider, benchmark_name)
        - Provider is registered
        - Benchmark configuration is valid

        Args:
            spec: The task specification to validate

        Returns:
            List of validation error messages. Empty list means valid.
        """
        errors: list[str] = []

        # Check required fields
        if not spec.artifact_url:
            errors.append("artifact_url is required for RL rollout tasks")

        env_provider = spec.env_provider or spec.config.get("env_provider")
        if not env_provider:
            errors.append("env_provider is required for RL rollout tasks")

        benchmark_name = spec.benchmark_name or spec.config.get("benchmark_name")
        if not benchmark_name:
            errors.append("benchmark_name is required for RL rollout tasks")

        # Check provider is registered
        if env_provider:
            if not self.provider_registry.has_provider(env_provider):
                # Not a hard error - EnvManager handles provider dispatch internally
                logger.debug(
                    "Provider %s not found in registry, will use EnvManager dispatch",
                    env_provider,
                )

        # Validate benchmark config structure
        config_payload = spec.config.get("config", spec.config)
        if not isinstance(config_payload, dict):
            errors.append("config must be a dictionary")

        return errors

    async def setup(self, spec: TaskSpec) -> TaskContext:
        """Set up the execution environment for an RL rollout.

        This method:
        1. Creates a Kubernetes pod for the miner's submission
        2. Waits for the pod to be ready
        3. Creates a RolloutCluster and RolloutWorker
        4. Establishes RPC communication with the submission container
        5. Returns a TaskContext with all required state

        Args:
            spec: The task specification

        Returns:
            TaskContext with execution state

        Raises:
            RuntimeError: If setup fails
        """
        context = TaskContext(
            spec=spec,
            work_dir="/tmp",  # TODO: configure per-job work directory
            start_time=datetime.now(timezone.utc),
        )

        submission_id = spec.submission_id
        job_id = spec.job_id or spec.task_id

        # Create container
        containers = Containers()
        try:
            pod_name = containers.create_container(
                submission_id,
                job_id,
                archive_url=spec.artifact_url,
                archive_sha256=spec.artifact_sha256,
            )
            context.container_name = pod_name
            context.state["containers"] = containers
            context.state["container_ready"] = True
            logger.info("Created pod: %s", pod_name)
        except Exception as e:
            logger.error("Failed to create container: %s", e)
            raise RuntimeError(f"Failed to create container: {e}") from e

        # Get NodePort and Node IP for direct TCP connection
        try:
            config.load_kube_config()
            k8v1api = client.CoreV1Api()
            v1 = client.CoreV1Api()
            service_name = pod_name
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

            context.container_host = node_ip
            context.container_port = node_port
        except Exception as e:
            logger.error("Failed to get container network info: %s", e)
            await self._cleanup_on_setup_failure(context)
            raise RuntimeError(f"Failed to get container network info: {e}") from e

        # Wait for container to be ready
        await asyncio.sleep(WAIT_TIME.total_seconds())

        # Build benchmark spec from task config
        benchmark_spec = self._build_benchmark_spec(spec)

        # Create Ray queues
        worker_to_rpc_queue = Queue(maxsize=QUEUE_MAXSIZE)
        rpc_to_worker_queue = Queue(maxsize=QUEUE_MAXSIZE)
        context.state["worker_to_rpc_queue"] = worker_to_rpc_queue
        context.state["rpc_to_worker_queue"] = rpc_to_worker_queue

        # Create rollout cluster and worker
        try:
            cluster = RolloutCluster(
                "eval-cluster",
                worker_remote_options=self.config.worker_remote_options,
            )
            worker = cluster.create_worker(
                SnowflakeId(job_id) if isinstance(job_id, int) else job_id,
                [benchmark_spec],
                node_ip,
                node_port,
                SnowflakeId(submission_id)
                if isinstance(submission_id, int)
                else submission_id,
                s3_config=self.config.s3_config,
                episode_log_interval=self.config.episode_log_interval,
                step_log_interval=self.config.step_log_interval,
                database_url=self.config.pg_database,
            )
            context.state["cluster"] = cluster
            context.state["worker"] = worker
        except Exception as e:
            logger.error("Failed to create rollout worker: %s", e)
            await self._cleanup_on_setup_failure(context)
            raise RuntimeError(f"Failed to create rollout worker: {e}") from e

        # Start RPC thread
        rpc_thread = threading.Thread(
            target=RPCProcess,
            args=(node_ip, node_port, rpc_to_worker_queue, worker_to_rpc_queue),
            daemon=True,
        )
        rpc_thread.start()
        context.state["rpc_thread"] = rpc_thread

        await asyncio.sleep(PROCESS_JOB_WAIT_TIME.total_seconds())

        # Wait for RPC handshake
        try:
            await self._wait_for_rpc_handshake(
                job_id=job_id,
                worker=worker,
                worker_to_rpc_queue=worker_to_rpc_queue,
                rpc_to_worker_queue=rpc_to_worker_queue,
            )
        except Exception as e:
            logger.error("RPC handshake failed: %s", e)
            await self._cleanup_on_setup_failure(context)
            raise RuntimeError(f"RPC handshake failed: {e}") from e

        logger.info("Setup complete for task %s", spec.task_id)
        return context

    async def execute(self, context: TaskContext) -> TaskResult:
        """Execute the RL rollout task.

        This runs all benchmark tasks and collects results.

        Args:
            context: The execution context from setup()

        Returns:
            TaskResult with evaluation metrics
        """
        spec = context.spec
        worker = context.state.get("worker")
        worker_to_rpc_queue = context.state.get("worker_to_rpc_queue")
        rpc_to_worker_queue = context.state.get("rpc_to_worker_queue")

        if not worker or not worker_to_rpc_queue or not rpc_to_worker_queue:
            return TaskResult(
                task_id=spec.task_id,
                success=False,
                error="Worker or queues not initialized",
            )

        start_time = time.time()

        try:
            # Start the evaluation
            evaluation_future = worker.run_all_benchmark_tasks.remote(
                worker_to_rpc_queue, rpc_to_worker_queue
            )

            # Wait for completion with timeout
            timeout_seconds = spec.timeout.total_seconds()
            elapsed = 0.0

            while elapsed < timeout_seconds:
                ready, _ = ray.wait(
                    [evaluation_future], timeout=RAY_WAIT_TIMEOUT.total_seconds()
                )

                if ready:
                    results: List[EnvResult] = ray.get(evaluation_future)
                    duration = time.time() - start_time

                    return self._build_success_result(
                        task_id=spec.task_id,
                        results=results,
                        duration=duration,
                    )

                elapsed = time.time() - start_time

            # Timeout
            logger.error("Task %s timed out after %.1f seconds", spec.task_id, elapsed)
            ray.cancel(evaluation_future)

            return TaskResult(
                task_id=spec.task_id,
                success=False,
                error=f"Task timed out after {elapsed:.1f} seconds",
                duration_seconds=elapsed,
            )

        except Exception as e:
            duration = time.time() - start_time
            logger.exception("Task %s failed: %s", spec.task_id, e)

            return TaskResult(
                task_id=spec.task_id,
                success=False,
                error=str(e),
                duration_seconds=duration,
            )

    async def teardown(self, context: TaskContext) -> None:
        """Clean up resources after task execution.

        Args:
            context: The execution context
        """
        spec = context.spec
        job_id = spec.job_id or spec.task_id
        submission_id = spec.submission_id

        logger.info("Starting teardown for task %s", spec.task_id)

        # Clean up queues
        self._cleanup_queues(context)

        # Clean up Ray worker
        cluster = context.state.get("cluster")
        worker = context.state.get("worker")
        if cluster and worker:
            try:
                ray.get(worker.cleanup.remote(), timeout=5)
            except Exception as e:
                logger.warning("Worker cleanup failed: %s", e)
            try:
                cluster.delete_worker(worker)
                logger.info("Cleaned up Ray worker for task %s", spec.task_id)
            except Exception as e:
                logger.warning("Failed to delete worker: %s", e)

        # Clean up container
        containers = context.state.get("containers")
        if containers and context.state.get("container_ready"):
            try:
                containers.cleanup_container(submission_id, job_id)
                logger.info(
                    "Cleaned up container for submission %s (task %s)",
                    submission_id,
                    spec.task_id,
                )
            except Exception as e:
                logger.warning("Failed to cleanup container: %s", e)

        # Force garbage collection
        gc.collect()

        logger.info("Teardown complete for task %s", spec.task_id)

    # Helper methods

    def _build_benchmark_spec(self, spec: TaskSpec) -> BenchmarkSpec:
        """Build a BenchmarkSpec from a TaskSpec."""
        config_payload = spec.config.get("config", spec.config)
        if not isinstance(config_payload, dict):
            config_payload = {}

        # Normalize camera names
        camera_names = spec.config.get("camera_names")
        if camera_names is None:
            camera_names = ("corner",)
        elif isinstance(camera_names, str):
            camera_names = (camera_names,)
        elif isinstance(camera_names, (list, tuple)):
            camera_names = tuple(camera_names)

        return BenchmarkSpec(
            provider=spec.env_provider or spec.config.get("env_provider", ""),
            benchmark_name=spec.benchmark_name or spec.config.get("benchmark_name", ""),
            config=config_payload,
            render_mode=spec.config.get("render_mode", "rgb_array"),
            camera_names=camera_names,
            camera_attribute=spec.config.get("camera_attribute", "camera_name"),
        )

    def _build_success_result(
        self,
        task_id: str,
        results: List[EnvResult],
        duration: float,
    ) -> TaskResult:
        """Build a TaskResult from successful evaluation results."""
        if not results:
            return TaskResult(
                task_id=task_id,
                success=True,
                metrics={
                    "success_rate": 0.0,
                    "avg_reward": 0.0,
                    "total_episodes": 0,
                },
                duration_seconds=duration,
                total_episodes=0,
                env_results=results,
            )

        total_episodes = sum(len(result.episodes) for result in results)
        avg_success_rate = sum(result.success_rate for result in results) / len(results)
        avg_reward = sum(result.mean_reward for result in results) / len(results)

        return TaskResult(
            task_id=task_id,
            success=True,
            metrics={
                "success_rate": avg_success_rate,
                "avg_reward": avg_reward,
                "total_episodes": float(total_episodes) if total_episodes else 0.0,
                "num_environments": float(len(results)),
            },
            duration_seconds=duration,
            total_episodes=total_episodes or None,
            env_results=results,
        )

    def _cleanup_queues(self, context: TaskContext) -> None:
        """Clean up Ray Queue actors."""
        worker_to_rpc = context.state.get("worker_to_rpc_queue")
        rpc_to_worker = context.state.get("rpc_to_worker_queue")

        for queue, name in [
            (worker_to_rpc, "worker_to_rpc"),
            (rpc_to_worker, "rpc_to_worker"),
        ]:
            if queue is not None:
                try:
                    if hasattr(queue, "actor") and queue.actor is not None:
                        queue.shutdown(force=True)
                        logger.debug("Shutdown %s queue", name)
                except Exception as e:
                    logger.warning("Failed to shutdown %s queue: %s", name, e)

    async def _cleanup_on_setup_failure(self, context: TaskContext) -> None:
        """Clean up resources after a setup failure."""
        self._cleanup_queues(context)

        cluster = context.state.get("cluster")
        worker = context.state.get("worker")
        if cluster and worker:
            try:
                cluster.delete_worker(worker)
            except Exception as e:
                logger.warning("Failed to delete worker during setup cleanup: %s", e)

        containers = context.state.get("containers")
        if containers and context.state.get("container_ready"):
            try:
                job_id = context.spec.job_id or context.spec.task_id
                containers.cleanup_container(context.spec.submission_id, job_id)
            except Exception as e:
                logger.warning(
                    "Failed to cleanup container during setup cleanup: %s", e
                )

        gc.collect()

    async def _wait_for_rpc_handshake(
        self,
        *,
        job_id: Any,
        worker,
        worker_to_rpc_queue: Queue,
        rpc_to_worker_queue: Queue,
    ) -> None:
        """Wait for RPC connection to be established."""
        max_attempts = max(1, self.rpc_handshake_max_attempts)
        retry_seconds = max(0.0, self.rpc_handshake_retry_seconds)

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
            f"Unable to establish RPC connection for job {job_id} "
            f"after {max_attempts} attempts: {last_error}"
        )
