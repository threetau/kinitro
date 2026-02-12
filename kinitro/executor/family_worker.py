"""Family worker - runs within a subprocess, handles one environment family."""

import asyncio
import logging
import multiprocessing as mp
import os

import structlog

from kinitro.backend.models import Task, TaskResult
from kinitro.environments import get_environments_by_family
from kinitro.executor.api_client import APIClient
from kinitro.executor.env_loader import (
    build_load_kwargs,
    force_remove_container,
    load_and_warmup_env,
    run_evaluation,
)
from kinitro.types import AffinetesEnv

# Configure structlog for this subprocess
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.add_log_level,
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.BoundLogger,
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class FamilyWorker:
    """
    Runs within a subprocess, handles one environment family.

    Architecture:
    - Single affinetes environment container (shared by all workers)
    - Fetch loop (producer): pulls tasks from API, pushes to internal queue
    - N execution workers (consumers): pull from queue, execute, submit results
    - Semaphore: limits concurrent task executions
    """

    def __init__(
        self,
        family: str,
        max_concurrent: int,
        api_url: str,
        executor_id: str,
        batch_size: int,
        image: str,
        eval_mode: str,
        mem_limit: str,
        hosts: list[str],
        max_timesteps: int,
        action_timeout: float,
        eval_timeout: int,
        use_images: bool,
        poll_interval: int,
        stats_queue: mp.Queue,
        api_key: str | None = None,
        gpu_enabled: bool = False,
    ):
        self.family = family
        self.max_concurrent = max_concurrent
        self.api_url = api_url.rstrip("/")
        self.executor_id = executor_id
        self.batch_size = batch_size
        self.image = image
        self.eval_mode = eval_mode
        self.mem_limit = mem_limit
        self.hosts = hosts
        self.max_timesteps = max_timesteps
        self.action_timeout = action_timeout
        self.eval_timeout = eval_timeout
        self.use_images = use_images
        self.poll_interval = poll_interval
        self.stats_queue = stats_queue
        self.api_key = api_key
        self.gpu_enabled = gpu_enabled

        # API client for HTTP communication
        self.api_client = APIClient(api_url, executor_id, api_key)

        # Async primitives (initialized in run())
        self.task_queue: asyncio.Queue | None = None
        self.semaphore: asyncio.Semaphore | None = None
        self.env: AffinetesEnv | None = None
        self.running = False

        # Metrics
        self.tasks_succeeded = 0
        self.tasks_failed = 0

    async def initialize(self) -> None:
        """Load the environment container once."""
        self.task_queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(self.max_concurrent)

        logger.info(
            "loading_eval_environment",
            family=self.family,
            image=self.image,
            mode=self.eval_mode,
            max_concurrent=self.max_concurrent,
        )

        # Load environment via affinetes
        load_kwargs = build_load_kwargs(
            image=self.image,
            eval_mode=self.eval_mode,
            mem_limit=self.mem_limit,
            executor_id=self.executor_id,
            family=self.family,
            hosts=self.hosts,
            eval_timeout=self.eval_timeout,
            gpu_enabled=self.gpu_enabled,
        )

        self.env = await load_and_warmup_env(self.family, self.image, load_kwargs)

    async def _fetch_loop(self) -> None:
        """Producer: fetch tasks from API, push to queue."""
        assert self.task_queue is not None, "task_queue not initialized"

        while self.running:
            try:
                # Backpressure: only fetch when queue has capacity
                current_size = self.task_queue.qsize()
                if current_size >= self.max_concurrent:
                    await asyncio.sleep(1)
                    continue

                # Calculate how many tasks to fetch
                fetch_size = min(self.batch_size, self.max_concurrent - current_size)

                tasks = await self._fetch_tasks_batch(fetch_size)
                if tasks:
                    for task in tasks:
                        await self.task_queue.put(task)
                    logger.debug(
                        "tasks_queued",
                        family=self.family,
                        count=len(tasks),
                        queue_size=self.task_queue.qsize(),
                    )
                else:
                    # No tasks available, wait before retrying
                    await asyncio.sleep(self.poll_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("fetch_error", family=self.family, error=str(e))
                await asyncio.sleep(5)

    async def _fetch_tasks_batch(self, batch_size: int) -> list[Task]:
        """Fetch a batch of tasks from the API, filtered to this family."""
        env_ids = get_environments_by_family(self.family)
        return await self.api_client.fetch_tasks(batch_size=batch_size, env_ids=env_ids)

    async def _execution_worker(self, worker_id: int) -> None:
        """Consumer: pull tasks from queue, execute, submit results."""
        assert self.task_queue is not None, "task_queue not initialized"
        assert self.semaphore is not None, "semaphore not initialized"

        while self.running:
            try:
                # Get task from queue with timeout
                try:
                    task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                except TimeoutError:
                    continue

                # Acquire semaphore slot
                async with self.semaphore:
                    try:
                        result = await self._execute_task(task)
                        await self._submit_result(result)
                        self.tasks_succeeded += 1
                    except Exception as e:
                        logger.error(
                            "task_failed",
                            family=self.family,
                            worker_id=worker_id,
                            task_uuid=task.task_uuid,
                            error=str(e),
                        )
                        # Submit failure result
                        failure_result = TaskResult(
                            task_uuid=task.task_uuid,
                            success=False,
                            score=0.0,
                            error=str(e),
                        )
                        await self._submit_result(failure_result)
                        self.tasks_failed += 1
                    finally:
                        self.task_queue.task_done()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    "execution_worker_error",
                    family=self.family,
                    worker_id=worker_id,
                    error=str(e),
                )

    async def _execute_task(self, task: Task) -> TaskResult:
        """Execute a single task."""
        logger.debug(
            "executing_task",
            family=self.family,
            task_uuid=task.task_uuid,
            miner_uid=task.miner_uid,
            env_id=task.env_id,
        )

        assert self.env is not None, "env not initialized"
        task_result = await run_evaluation(
            env=self.env,
            task=task,
            max_timesteps=self.max_timesteps,
            action_timeout=self.action_timeout,
            use_images=self.use_images,
            eval_timeout=self.eval_timeout,
        )

        logger.info(
            "task_executed",
            family=self.family,
            task_uuid=task.task_uuid,
            success=task_result.success,
            score=task_result.score,
        )

        return task_result

    async def _submit_result(self, result: TaskResult) -> None:
        """Submit a task result to the API."""
        await self.api_client.submit_results([result])

    async def _report_metrics(self) -> None:
        """Periodically report metrics to main process."""
        while self.running:
            try:
                metrics = {
                    "family": self.family,
                    "queue_size": self.task_queue.qsize() if self.task_queue else 0,
                    "tasks_succeeded": self.tasks_succeeded,
                    "tasks_failed": self.tasks_failed,
                }
                self.stats_queue.put(metrics)
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(5)

    async def run(self) -> None:
        """Main entry point - spawn workers and run."""
        await self.initialize()
        self.running = True

        logger.info(
            "family_worker_started",
            family=self.family,
            max_concurrent=self.max_concurrent,
            pid=os.getpid(),
        )

        # Start fetch loop (producer)
        fetch_task = asyncio.create_task(self._fetch_loop())

        # Start metrics reporter
        metrics_task = asyncio.create_task(self._report_metrics())

        # Create N execution workers (consumers)
        worker_tasks = [
            asyncio.create_task(self._execution_worker(i)) for i in range(self.max_concurrent)
        ]

        try:
            # Wait for all tasks (they run forever until cancelled)
            await asyncio.gather(fetch_task, metrics_task, *worker_tasks)
        except asyncio.CancelledError:
            pass
        finally:
            self.running = False
            await self._cleanup()

    async def _cleanup(self) -> None:
        """Cleanup resources."""
        logger.info("family_worker_cleaning_up", family=self.family)

        # Close HTTP session
        await self.api_client.close()

        # Cleanup environment
        if self.env:
            try:
                await self.env.cleanup()
            except Exception as e:
                logger.warning("env_cleanup_error", family=self.family, error=str(e))

        # Force cleanup docker container
        container_name = f"kinitro-eval-{self.executor_id}-{self.family}"
        force_remove_container(container_name)

        logger.info("family_worker_stopped", family=self.family)


def run_family_worker(
    family: str,
    max_concurrent: int,
    api_url: str,
    executor_id: str,
    batch_size: int,
    image: str,
    eval_mode: str,
    mem_limit: str,
    hosts: list[str],
    max_timesteps: int,
    action_timeout: float,
    eval_timeout: int,
    use_images: bool,
    poll_interval: int,
    stats_queue: mp.Queue,
    log_level: str,
    api_key: str | None = None,
    gpu_enabled: bool = False,
) -> None:
    """Entry point for subprocess (called by multiprocessing)."""
    # Configure logging for subprocess
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    worker = FamilyWorker(
        family=family,
        max_concurrent=max_concurrent,
        api_url=api_url,
        executor_id=executor_id,
        batch_size=batch_size,
        image=image,
        eval_mode=eval_mode,
        mem_limit=mem_limit,
        hosts=hosts,
        max_timesteps=max_timesteps,
        action_timeout=action_timeout,
        eval_timeout=eval_timeout,
        use_images=use_images,
        poll_interval=poll_interval,
        stats_queue=stats_queue,
        api_key=api_key,
        gpu_enabled=gpu_enabled,
    )

    try:
        asyncio.run(worker.run())
    except KeyboardInterrupt:
        pass
