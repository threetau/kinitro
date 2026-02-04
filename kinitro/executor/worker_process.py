"""Worker process wrapper for concurrent executor."""

import multiprocessing as mp
import os
import signal
from typing import TYPE_CHECKING

import structlog

from kinitro.executor.family_worker import run_family_worker

logger = structlog.get_logger()

if TYPE_CHECKING:
    from kinitro.executor.config import ExecutorConfig


class WorkerProcess:
    """
    Wrapper for a worker subprocess.

    Each WorkerProcess manages one environment family and runs
    a FamilyWorker internally with multiple async execution workers.
    """

    def __init__(
        self,
        family: str,
        max_concurrent: int,
        config: "ExecutorConfig",
        stats_queue: mp.Queue,
    ):
        self.family = family
        self.max_concurrent = max_concurrent
        self.config = config
        self.stats_queue = stats_queue
        self.process: mp.Process | None = None

    def start(self) -> None:
        """Start the worker subprocess."""
        self.process = mp.Process(
            target=run_family_worker,
            args=(
                self.family,
                self.max_concurrent,
                self.config.api_url,
                self.config.executor_id,
                self.config.batch_size,
                self.config.get_image_for_env(self.family),
                self.config.eval_mode,
                self.config.eval_mem_limit,
                self.config.eval_hosts,
                self.config.max_timesteps,
                self.config.action_timeout,
                self.config.eval_timeout,
                self.config.use_images,
                self.config.poll_interval_seconds,
                self.stats_queue,
                self.config.log_level,
                self.config.api_key,
            ),
            name=f"Worker-{self.family}",
        )
        self.process.start()

    @property
    def pid(self) -> int | None:
        """Get the process ID."""
        return self.process.pid if self.process else None

    def is_alive(self) -> bool:
        """Check if the process is running."""
        return self.process is not None and self.process.is_alive()

    def stop(self) -> None:
        """Signal the worker to stop gracefully via SIGINT.

        The worker catches KeyboardInterrupt and performs cleanup (closing HTTP
        sessions, removing Docker containers). If it doesn't exit within the
        join timeout, the manager will call terminate() to force-kill it.
        """
        if self.process and self.process.is_alive() and self.process.pid:
            logger.warning("sending_sigint_to_worker", family=self.family, pid=self.process.pid)
            try:
                os.kill(self.process.pid, signal.SIGINT)
            except ProcessLookupError:
                # Process exited between is_alive() check and os.kill() - that's fine
                logger.info("worker_already_exited", family=self.family, pid=self.process.pid)

    def join(self, timeout: float | None = None) -> None:
        """Wait for the process to terminate."""
        if self.process:
            self.process.join(timeout=timeout)

    def terminate(self) -> None:
        """Forcefully terminate the process."""
        if self.process and self.process.is_alive():
            self.process.terminate()
