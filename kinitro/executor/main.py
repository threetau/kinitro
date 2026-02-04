"""Main executor service."""

import asyncio
import signal

import structlog

from kinitro.executor.api_client import APIClient
from kinitro.executor.config import ExecutorConfig
from kinitro.executor.worker import Worker

logger = structlog.get_logger()


class Executor:
    """
    Executor service that fetches and processes evaluation tasks.

    The executor:
    1. Polls the API for pending tasks
    2. Executes tasks using affinetes (MuJoCo evaluation)
    3. Submits results back to the API
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self.api_client = APIClient(config.api_url, config.executor_id, config.api_key)
        self.worker = Worker(config)
        self._running = False
        self._shutdown_event = asyncio.Event()

    async def start(self) -> None:
        """Start the executor loop."""
        if self._running:
            logger.warning("executor_already_running")
            return

        self._running = True
        logger.info(
            "executor_started",
            executor_id=self.config.executor_id,
            api_url=self.config.api_url,
            batch_size=self.config.batch_size,
        )

        # Check API health
        if not await self.api_client.health_check():
            logger.error("api_not_healthy", url=self.config.api_url)
            # Continue anyway, it might come up

        while self._running:
            try:
                await self._process_batch()
            except asyncio.CancelledError:
                logger.info("executor_cancelled")
                break
            except Exception as e:
                logger.error("executor_error", error=str(e))

            # Wait before next poll
            if self._running:
                try:
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.poll_interval_seconds,
                    )
                except TimeoutError:
                    pass  # Normal timeout, continue loop

    async def stop(self) -> None:
        """Stop the executor."""
        self._running = False
        self._shutdown_event.set()
        await self.worker.cleanup()
        await self.api_client.close()
        logger.info("executor_stopped", executor_id=self.config.executor_id)

    async def _process_batch(self) -> None:
        """Fetch and process a batch of tasks."""
        # Fetch tasks
        tasks = await self.api_client.fetch_tasks(
            batch_size=self.config.batch_size,
            env_ids=self.config.env_ids,
        )

        if not tasks:
            logger.debug("no_tasks_available")
            return

        logger.info("processing_batch", count=len(tasks))

        # Execute tasks
        results = await self.worker.execute_batch(tasks)

        # Submit results
        accepted, rejected = await self.api_client.submit_results(results)

        logger.info(
            "batch_complete",
            executed=len(tasks),
            accepted=accepted,
            rejected=rejected,
        )


async def run_executor(config: ExecutorConfig) -> None:
    """
    Run the executor service.

    Args:
        config: Executor configuration
    """
    executor = Executor(config)

    # Setup signal handlers
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("shutdown_signal_received")
        asyncio.create_task(executor.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    try:
        await executor.start()
    finally:
        # Ensure cleanup on any exit
        if executor._running:
            await executor.stop()
        executor.worker.force_cleanup()
