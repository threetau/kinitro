"""Main scheduler service for orchestrating evaluation cycles."""

import asyncio
import time

import structlog

from kinitro.backend.storage import Storage
from kinitro.chain.commitments import read_miner_commitments
from kinitro.environments import get_all_environment_ids
from kinitro.scheduler.config import SchedulerConfig
from kinitro.scheduler.scoring import (
    aggregate_task_results,
    compute_weights,
    convert_to_scores_data,
)
from kinitro.scheduler.task_generator import generate_tasks

logger = structlog.get_logger()


class Scheduler:
    """
    Scheduler service for orchestrating evaluation cycles.

    The scheduler is responsible for:
    1. Reading miner commitments from the chain
    2. Creating evaluation tasks in the task pool
    3. Waiting for executors to complete tasks
    4. Computing Pareto scores and weights
    5. Storing results in the database
    """

    def __init__(self, config: SchedulerConfig, storage: Storage):
        self.config = config
        self.storage = storage
        self.env_ids = get_all_environment_ids()
        self._running = False
        self._subtensor = None

    @property
    def subtensor(self):
        """Lazy-load subtensor connection."""
        if self._subtensor is None:
            import bittensor as bt

            self._subtensor = bt.Subtensor(network=self.config.network)
        return self._subtensor

    async def start(self) -> None:
        """Start the scheduler loop."""
        if self._running:
            logger.warning("scheduler_already_running")
            return

        self._running = True
        logger.info(
            "scheduler_started",
            interval_seconds=self.config.eval_interval_seconds,
            n_environments=len(self.env_ids),
        )

        while self._running:
            try:
                await self._run_evaluation_cycle()
            except asyncio.CancelledError:
                logger.info("evaluation_cycle_cancelled")
                raise
            except Exception as e:
                logger.error("evaluation_cycle_error", error=str(e))

            # Wait for next cycle
            if self._running:
                logger.info(
                    "waiting_for_next_cycle",
                    seconds=self.config.eval_interval_seconds,
                )
                try:
                    await asyncio.sleep(self.config.eval_interval_seconds)
                except asyncio.CancelledError:
                    logger.info("sleep_cancelled")
                    break

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        logger.info("scheduler_stopped")

    async def _run_evaluation_cycle(self) -> None:
        """Run a single evaluation cycle."""
        start_time = time.time()

        # Get current block
        block_number = self.subtensor.block
        logger.info("starting_evaluation_cycle", block=block_number)

        # Create cycle in database
        async with self.storage.session() as session:
            cycle = await self.storage.create_cycle(
                session,
                block_number=block_number,
                n_environments=len(self.env_ids),
            )
            cycle_id = cycle.id

        try:
            # 1. Discover miners from chain
            neurons = self.subtensor.neurons(netuid=self.config.netuid)
            miners = read_miner_commitments(
                self.subtensor,
                self.config.netuid,
                neurons,
            )

            if not miners:
                logger.warning("no_miners_with_commitments")
                async with self.storage.session() as session:
                    await self.storage.fail_cycle(session, cycle_id, "No miners with commitments")
                return

            # Update cycle with miner count
            async with self.storage.session() as session:
                cycle = await self.storage.get_cycle(session, cycle_id)
                if cycle:
                    cycle.n_miners = len(miners)
                    await session.commit()

            logger.info("found_miners", count=len(miners))

            # 2. Generate and create tasks
            tasks_data = generate_tasks(
                miners=miners,
                env_ids=self.env_ids,
                episodes_per_env=self.config.episodes_per_env,
                block_number=block_number,
                cycle_id=cycle_id,
            )

            async with self.storage.session() as session:
                await self.storage.create_tasks_bulk(session, tasks_data)

            logger.info(
                "tasks_created",
                cycle_id=cycle_id,
                total_tasks=len(tasks_data),
            )

            # 3. Wait for tasks to complete
            cycle_completed = await self._wait_for_cycle_completion(cycle_id)

            if not cycle_completed:
                # Cycle timed out - fail it and don't publish partial weights
                async with self.storage.session() as session:
                    await self.storage.fail_cycle(
                        session, cycle_id, "Cycle timed out waiting for tasks"
                    )
                logger.error(
                    "evaluation_cycle_timeout",
                    cycle_id=cycle_id,
                    timeout_seconds=self.config.cycle_timeout_seconds,
                )
                return

            # 4. Aggregate results and compute scores
            async with self.storage.session() as session:
                completed_tasks = await self.storage.get_cycle_task_results(session, cycle_id)

            miner_scores = aggregate_task_results(completed_tasks)

            # 5. Compute weights
            weights, weights_u16 = compute_weights(
                miner_scores=miner_scores,
                env_ids=self.env_ids,
                episodes_per_env=self.config.episodes_per_env,
                pareto_temperature=self.config.pareto_temperature,
            )

            # 6. Store scores and weights
            miners_by_uid = {m.uid: m.hotkey for m in miners}
            scores_data = convert_to_scores_data(
                miner_scores,
                miners_by_uid,
                self.config.episodes_per_env,
            )

            async with self.storage.session() as session:
                await self.storage.add_miner_scores_bulk(session, cycle_id, scores_data)
                await self.storage.save_weights(
                    session,
                    cycle_id=cycle_id,
                    block_number=block_number,
                    weights={int(k): float(v) for k, v in weights.items()},
                    weights_u16=weights_u16,
                )

            # 7. Mark cycle as completed
            duration = time.time() - start_time
            async with self.storage.session() as session:
                await self.storage.complete_cycle(session, cycle_id, duration)

            logger.info(
                "evaluation_cycle_complete",
                cycle_id=cycle_id,
                block=block_number,
                duration_seconds=duration,
                n_miners=len(miners),
            )

        except Exception as e:
            logger.error("evaluation_cycle_failed", cycle_id=cycle_id, error=str(e))
            async with self.storage.session() as session:
                await self.storage.fail_cycle(session, cycle_id, str(e))
            raise

    async def _wait_for_cycle_completion(self, cycle_id: int) -> bool:
        """
        Wait for all tasks in a cycle to complete.

        Returns:
            True if cycle completed successfully, False if timed out.
        """
        start_time = time.time()
        timeout = self.config.cycle_timeout_seconds
        check_interval = 10  # seconds

        while True:
            # Check if cycle is complete
            async with self.storage.session() as session:
                if await self.storage.is_cycle_complete(session, cycle_id):
                    logger.info("cycle_tasks_complete", cycle_id=cycle_id)
                    return True

                # Reassign stale tasks
                stale_count = await self.storage.reassign_stale_tasks(
                    session,
                    stale_threshold_seconds=self.config.task_stale_threshold_seconds,
                )
                if stale_count > 0:
                    logger.info("stale_tasks_reassigned", count=stale_count)

                # Get stats for logging
                stats = await self.storage.get_task_pool_stats(session, cycle_id)

            elapsed = time.time() - start_time
            logger.info(
                "waiting_for_tasks",
                cycle_id=cycle_id,
                pending=stats["pending_tasks"],
                assigned=stats["assigned_tasks"],
                completed=stats["completed_tasks"],
                failed=stats["failed_tasks"],
                elapsed_seconds=int(elapsed),
            )

            # Check timeout
            if elapsed > timeout:
                logger.warning(
                    "cycle_timeout",
                    cycle_id=cycle_id,
                    timeout_seconds=timeout,
                )
                return False

            # Wait before next check
            await asyncio.sleep(check_interval)


async def run_scheduler(config: SchedulerConfig) -> None:
    """
    Run the scheduler service.

    Args:
        config: Scheduler configuration
    """
    storage = Storage(config.database_url)
    await storage.initialize()

    scheduler = Scheduler(config, storage)

    try:
        await scheduler.start()
    finally:
        await scheduler.stop()
        await storage.close()
