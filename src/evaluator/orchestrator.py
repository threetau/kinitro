import asyncio
import gc
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional

import asyncpg
import ray
from fiber.chain.chain_utils import load_hotkey_keypair
from kubernetes import client, config
from pgqueuer import PgQueuer
from pgqueuer.db import AsyncpgDriver
from pgqueuer.models import Job
from ray.util.queue import Queue
from snowflake import SnowflakeGenerator

from core.db.models import EvaluationStatus
from core.log import get_logger
from core.messages import EvalJobMessage, EvalResultMessage, JobStatusUpdateMessage
from evaluator.config import EvaluatorConfig
from evaluator.containers import Containers
from evaluator.rollout import BenchmarkSpec, RolloutCluster
from evaluator.rollout.envs import EnvResult
from evaluator.rpc.rpc_process import RPCProcess
from validator.db.db_manager import DatabaseManager
from validator.db.models import EvaluationJob

logger = get_logger(__name__)

WAIT_TIME = 5
PROCESS_JOB_WAIT_TIME = 1
QUEUE_MAXSIZE = 100
# TODO: this might be way too long
EVAL_TIMEOUT = 900
RAY_WAIT_TIMEOUT = 0.1
MIN_CONCURRENT_JOBS = 4


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

        # Track running jobs for concurrent execution
        self.running_jobs: Dict[str, Dict] = {}  # job_id -> job_info
        if config.max_concurrent_jobs < MIN_CONCURRENT_JOBS:
            logger.warning(
                "Configured max_concurrent_jobs (%s) below minimum (%s); clamping.",
                config.max_concurrent_jobs,
                MIN_CONCURRENT_JOBS,
            )
        self.max_concurrent_jobs = max(MIN_CONCURRENT_JOBS, config.max_concurrent_jobs)
        self.job_timeout = config.settings.get("job_timeout", EVAL_TIMEOUT)
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

        evaluation_job = EvaluationJob(
            id=eval_job_msg.job_id,
            competition_id=eval_job_msg.competition_id,
            submission_id=eval_job_msg.submission_id,
            miner_hotkey=eval_job_msg.miner_hotkey,
            hf_repo_id=eval_job_msg.hf_repo_id,
            env_provider=eval_job_msg.env_provider,
            benchmark_name=eval_job_msg.benchmark_name,
            config=eval_job_msg.config,
            artifact_url=eval_job_msg.artifact_url,
            artifact_expires_at=eval_job_msg.artifact_expires_at,
            artifact_sha256=eval_job_msg.artifact_sha256,
            artifact_size_bytes=eval_job_msg.artifact_size_bytes,
            created_at=datetime.now(timezone.utc),
        )

        # Create job entry in the database with QUEUED status
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
        pod = containers.create_container(
            eval_job_msg.submission_id,
            archive_url=eval_job_msg.artifact_url,
            archive_sha256=eval_job_msg.artifact_sha256,
        )
        logger.info(f"Created pod: {pod}")

        # Get NodePort and Node IP for direct TCP connection
        config.load_kube_config()
        k8v1api = client.CoreV1Api()
        v1 = client.CoreV1Api()
        service_name = f"submission-{eval_job_msg.submission_id}"
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

        # Create a benchmark spec for the job
        benchmark_spec = BenchmarkSpec(
            provider=eval_job_msg.env_provider,
            benchmark_name=eval_job_msg.benchmark_name,
            config=eval_job_msg.config,
            render_mode="rgb_array",
        )

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

        # Test RPC connection
        res = await worker.test_rpc.remote(worker_to_rpc_queue, rpc_to_worker_queue)
        logger.info(f"RPC test result for job {eval_job_msg.job_id}: {res}")

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
        }

        logger.info(
            f"Created job context for {eval_job_msg.job_id} with queues: worker_to_rpc={worker_to_rpc_queue is not None}, rpc_to_worker={rpc_to_worker_queue is not None}"
        )

        return job_context

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

                # Update database
                try:
                    self.db.update_evaluation_job(
                        job_id,
                        {
                            "status": EvaluationStatus.COMPLETED,
                            "completed_at": datetime.now(timezone.utc),
                        },
                    )
                except Exception as e:
                    logger.error(f"Failed to update job status for job {job_id}: {e}")

                # Queue result message
                eval_result_msg = EvalResultMessage(
                    job_id=job_id,
                    validator_hotkey=self.keypair.ss58_address,
                    miner_hotkey=eval_job_msg.miner_hotkey,
                    competition_id=eval_job_msg.competition_id,
                    env_provider=eval_job_msg.env_provider,
                    benchmark_name=eval_job_msg.benchmark_name,
                    config=eval_job_msg.config,
                    score=avg_reward,
                    success_rate=avg_success_rate,
                    avg_reward=avg_reward,
                    total_episodes=total_episodes,
                    logs="Evaluation completed successfully",
                    error=None,
                    extra_data=None,
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
                    containers = Containers()
                    containers.cleanup_container(submission_id)
                    logger.info(
                        f"Cleaned up container resources for submission {submission_id}"
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
            if elapsed > self.job_timeout:
                logger.error(f"Job {job_id} timed out after {elapsed} seconds")
                ray.cancel(evaluation_future)
                self.db.update_evaluation_job(
                    job_id,
                    {
                        "status": EvaluationStatus.TIMEOUT,
                        "error_message": f"Job timed out after {elapsed:.1f} seconds",
                        "completed_at": datetime.now(timezone.utc),
                    },
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
                    containers = Containers()
                    containers.cleanup_container(submission_id)
                    logger.info(
                        f"Cleaned up container resources for timed out submission {submission_id}"
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
            self.db.update_evaluation_job(
                job_id,
                {
                    "status": EvaluationStatus.FAILED,
                    "error_message": str(e),
                    "completed_at": datetime.now(timezone.utc),
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
                containers = Containers()
                containers.cleanup_container(submission_id)
                logger.info(
                    f"Cleaned up container resources for failed submission {submission_id}"
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
        except Exception as e:
            logger.error(f"Failed to process job {getattr(job, 'id', 'unknown')}: {e}")
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
            if stale_jobs:
                for job in stale_jobs:
                    logger.info(f"Marking stale job {job.id} as FAILED")
                    # Update job status to FAILED with error message
                    self.db.update_evaluation_job(
                        job.id,
                        {
                            "status": EvaluationStatus.FAILED,
                            "error_message": "Job was interrupted due to orchestrator restart",
                            "completed_at": datetime.now(timezone.utc),
                        },
                    )

                    # Queue failure result message to backend
                    eval_result_msg = EvalResultMessage(
                        job_id=job.id,
                        validator_hotkey=self.keypair.ss58_address,
                        miner_hotkey=job.miner_hotkey,
                        competition_id=job.competition_id,
                        env_provider=job.env_provider,
                        benchmark_name=job.benchmark_name,
                        config=job.config,
                        score=0.0,
                        success_rate=0.0,
                        avg_reward=0.0,
                        total_episodes=None,
                        logs="Job failed due to orchestrator restart",
                        error="Orchestrator restart - job was not completed",
                        extra_data=None,
                    )
                    await self.db.queue_evaluation_result_msg(eval_result_msg)

                    # Clean up any orphaned containers for this job
                    if job.submission_id:
                        try:
                            containers = Containers()
                            containers.cleanup_container(job.submission_id)
                            logger.info(
                                f"Cleaned up orphaned container for submission {job.submission_id}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to cleanup container for submission {job.submission_id}: {e}"
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
                            # Ensure both datetimes have timezone info for comparison
                            current_time = datetime.now(timezone.utc)
                            completed_time = job.completed_at
                            if completed_time.tzinfo is None:
                                completed_time = completed_time.replace(
                                    tzinfo=timezone.utc
                                )

                            if (
                                current_time - completed_time
                            ).total_seconds() > self.job_timeout:
                                try:
                                    if job.submission_id:
                                        containers.cleanup_container(job.submission_id)
                                        logger.debug(
                                            f"Cleaned up container for old job {job.id}"
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
                    containers.cleanup_container(submission_id)
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
    orc = Orchestrator(EvaluatorConfig())
    asyncio.run(orc.start())
