"""Concurrent executor manager - spawns worker processes per environment family."""

import asyncio
import multiprocessing as mp
import signal
from dataclasses import dataclass, field
from datetime import datetime

import structlog

from kinitro.executor.config import ExecutorConfig
from kinitro.executor.worker_process import WorkerProcess

logger = structlog.get_logger()


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
    - One subprocess per environment family (metaworld, procthor, etc.)
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
            worker.join(timeout=10)
            if worker.is_alive():
                logger.warning("worker_force_kill", family=family, pid=worker.pid)
                worker.terminate()

        self.worker_processes.clear()
        logger.info("executor_manager_stopped")

    async def _health_checker(self) -> None:
        """Monitor and restart dead workers."""
        while self.running:
            try:
                for family, worker in list(self.worker_processes.items()):
                    if not worker.is_alive():
                        old_pid = worker.pid
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

                await asyncio.sleep(10)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("health_checker_error", error=str(e))
                await asyncio.sleep(10)

    async def _stats_collector(self) -> None:
        """Collect metrics from workers via IPC queue."""
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
                    except Exception:
                        break  # Queue empty

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

                await asyncio.sleep(30)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("stats_collector_error", error=str(e))
                await asyncio.sleep(30)


async def run_concurrent_executor(config: ExecutorConfig) -> None:
    """
    Run the concurrent executor service.

    Args:
        config: Executor configuration
    """
    manager = ExecutorManager(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

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
