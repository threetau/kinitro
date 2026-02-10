"""Concurrent executor manager - spawns worker processes per environment family."""

import asyncio
import multiprocessing as mp
import signal
from dataclasses import dataclass, field
from datetime import datetime
from queue import Empty

import structlog

from kinitro.executor.config import ExecutorConfig
from kinitro.executor.worker_process import WorkerProcess

logger = structlog.get_logger()

# Timeouts for worker process management (in seconds)
GRACEFUL_SHUTDOWN_TIMEOUT = 10  # Time to wait for worker to exit after SIGINT
FORCE_KILL_JOIN_TIMEOUT = 5  # Time to wait after SIGTERM before giving up
HEALTH_CHECK_INTERVAL = 10  # Interval between health checks
STATS_COLLECTION_INTERVAL = 30  # Interval between metrics aggregation


@dataclass
class WorkerMetrics:
    """Metrics collected from a worker process."""

    family: str
    queue_size: int = 0
    tasks_succeeded: int = 0
    tasks_failed: int = 0
    last_update: datetime = field(default_factory=datetime.now)


class ExecutorManager:
    """
    Main process that spawns and manages worker subprocesses.

    Architecture:
    - One subprocess per environment family (metaworld, genesis, etc.)
    - Each subprocess runs a FamilyWorker with N async execution workers
    - Health checker monitors and restarts dead workers
    - Stats collector aggregates metrics from workers via IPC queue
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.worker_processes: dict[str, WorkerProcess] = {}
        self.stats_queue: mp.Queue = mp.Queue()
        self.running = False
        self._shutdown_event = asyncio.Event()
        self._metrics: dict[str, WorkerMetrics] = {}

    async def start(self) -> None:
        """Start worker processes for each environment family."""
        if self.running:
            logger.warning("executor_manager_already_running")
            return

        self.running = True
        families = self.config.get_env_families()

        logger.info(
            "executor_manager_starting",
            executor_id=self.config.executor_id,
            families=families,
            api_url=self.config.api_url,
        )

        # Spawn one worker process per environment family
        for family in families:
            max_concurrent = self.config.get_max_concurrent(family)

            worker = WorkerProcess(
                family=family,
                max_concurrent=max_concurrent,
                config=self.config,
                stats_queue=self.stats_queue,
            )
            worker.start()
            self.worker_processes[family] = worker
            self._metrics[family] = WorkerMetrics(family=family)

            logger.info(
                "worker_started",
                family=family,
                max_concurrent=max_concurrent,
                pid=worker.pid,
            )

        # Start health checker and stats collector
        health_task = asyncio.create_task(self._health_checker())
        stats_task = asyncio.create_task(self._stats_collector())

        # Wait for shutdown
        await self._shutdown_event.wait()

        # Cancel background tasks
        health_task.cancel()
        stats_task.cancel()

        try:
            await asyncio.gather(health_task, stats_task, return_exceptions=True)
        except asyncio.CancelledError:
            pass

    async def stop(self) -> None:
        """Stop all worker processes."""
        self.running = False
        self._shutdown_event.set()

        logger.info("executor_manager_stopping")

        # Stop all workers
        for family, worker in self.worker_processes.items():
            logger.info("stopping_worker", family=family, pid=worker.pid)
            worker.stop()

        # Wait for workers to terminate
        for family, worker in self.worker_processes.items():
            worker.join(timeout=GRACEFUL_SHUTDOWN_TIMEOUT)
            if worker.is_alive():
                logger.warning("worker_force_kill", family=family, pid=worker.pid)
                worker.terminate()
                worker.join(timeout=FORCE_KILL_JOIN_TIMEOUT)

        self.worker_processes.clear()
        logger.info("executor_manager_stopped")

    async def _health_checker(self) -> None:
        """Monitor and restart dead workers."""
        while self.running:
            try:
                for family, worker in list(self.worker_processes.items()):
                    if not worker.is_alive():
                        old_pid = worker.pid
                        # Join dead worker to reap zombie process
                        worker.join(timeout=0)
                        logger.warning(
                            "worker_died_restarting",
                            family=family,
                            pid=old_pid,
                        )

                        # Restart worker
                        max_concurrent = self.config.get_max_concurrent(family)
                        new_worker = WorkerProcess(
                            family=family,
                            max_concurrent=max_concurrent,
                            config=self.config,
                            stats_queue=self.stats_queue,
                        )
                        new_worker.start()
                        self.worker_processes[family] = new_worker

                        logger.info(
                            "worker_restarted",
                            family=family,
                            old_pid=old_pid,
                            new_pid=new_worker.pid,
                        )

                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_checker_error", error=str(e))
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)

    async def _stats_collector(self) -> None:
        """Collect metrics from workers via IPC queue.

        Uses a 30-second interval for logging aggregated metrics. This is intentionally
        longer than the worker's 5-second reporting interval to batch updates and reduce
        log noise. No exponential backoff is needed since this is non-critical observability
        - failures just skip a reporting cycle and retry next interval.
        """
        while self.running:
            try:
                # Drain all available metrics from queue
                while True:
                    try:
                        metrics = self.stats_queue.get_nowait()
                        family = metrics.get("family")
                        if family and family in self._metrics:
                            self._metrics[family] = WorkerMetrics(
                                family=family,
                                queue_size=metrics.get("queue_size", 0),
                                tasks_succeeded=metrics.get("tasks_succeeded", 0),
                                tasks_failed=metrics.get("tasks_failed", 0),
                                last_update=datetime.now(),
                            )
                    except Empty:
                        break

                # Log aggregated metrics
                total_succeeded = sum(m.tasks_succeeded for m in self._metrics.values())
                total_failed = sum(m.tasks_failed for m in self._metrics.values())
                total_queued = sum(m.queue_size for m in self._metrics.values())

                if total_succeeded > 0 or total_failed > 0:
                    logger.info(
                        "executor_metrics",
                        total_succeeded=total_succeeded,
                        total_failed=total_failed,
                        total_queued=total_queued,
                        workers=len(self.worker_processes),
                    )

                await asyncio.sleep(STATS_COLLECTION_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("stats_collector_error", error=str(e))
                await asyncio.sleep(STATS_COLLECTION_INTERVAL)


async def run_concurrent_executor(config: ExecutorConfig) -> None:
    """
    Run the concurrent executor service.

    Args:
        config: Executor configuration
    """
    manager = ExecutorManager(config)

    # Setup signal handlers
    loop = asyncio.get_running_loop()

    def signal_handler():
        logger.info("shutdown_signal_received")
        asyncio.create_task(manager.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await manager.start()
    finally:
        if manager.running:
            await manager.stop()
