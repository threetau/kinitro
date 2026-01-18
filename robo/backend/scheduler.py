"""Background evaluation scheduler for the backend service."""

import asyncio
import time
from typing import Any

import structlog

from robo.backend.config import BackendConfig
from robo.backend.storage import Storage
from robo.chain.commitments import MinerCommitment, read_miner_commitments
from robo.chain.weights import weights_to_u16
from robo.environments import get_all_environment_ids
from robo.evaluation.parallel import evaluate_all_miners
from robo.evaluation.rollout import RolloutConfig
from robo.scoring.pareto import compute_pareto_frontier
from robo.scoring.winners_take_all import compute_subset_scores, scores_to_weights

logger = structlog.get_logger()


class EvaluationScheduler:
    """
    Background scheduler that runs evaluation cycles.

    Discovers miners from chain, evaluates them, computes scores,
    and stores results in the database for validators to retrieve.
    """

    def __init__(self, config: BackendConfig, storage: Storage):
        """
        Initialize scheduler.

        Args:
            config: Backend configuration
            storage: Database storage instance
        """
        self.config = config
        self.storage = storage
        self.env_ids = get_all_environment_ids()
        self._running = False
        self._current_task: asyncio.Task | None = None

        # Lazy-loaded bittensor connection
        self._subtensor = None

    @property
    def subtensor(self):
        """Lazy-load subtensor connection."""
        if self._subtensor is None:
            import bittensor as bt

            self._subtensor = bt.Subtensor(network=self.config.network)
        return self._subtensor

    async def start(self) -> None:
        """Start the background evaluation loop."""
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
            except Exception as e:
                logger.error("evaluation_cycle_error", error=str(e))

            # Wait for next cycle
            if self._running:
                logger.info(
                    "waiting_for_next_cycle",
                    seconds=self.config.eval_interval_seconds,
                )
                await asyncio.sleep(self.config.eval_interval_seconds)

    async def stop(self) -> None:
        """Stop the scheduler."""
        self._running = False
        if self._current_task:
            self._current_task.cancel()
        logger.info("scheduler_stopped")

    @property
    def is_running(self) -> bool:
        """Check if scheduler is running."""
        return self._running

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

            logger.info("found_miners", count=len(miners))

            # 2. Run evaluations
            rollout_config = RolloutConfig(
                max_timesteps=self.config.max_timesteps_per_episode,
                action_timeout_ms=self.config.action_timeout_ms,
            )

            # Note: Using a dummy validator hotkey since backend doesn't have one
            # This is fine since seeds are deterministic per block anyway
            evaluation_result = await evaluate_all_miners(
                miners=miners,
                environment_ids=self.env_ids,
                block_number=block_number,
                validator_hotkey="backend-evaluator",
                episodes_per_env=self.config.episodes_per_env,
                rollout_config=rollout_config,
                use_basilica=bool(self.config.basilica_api_token),
                basilica_api_token=self.config.basilica_api_token,
            )

            # 3. Store scores in database
            scores_data = []
            miner_scores_dict: dict[int, dict[str, float]] = {}

            for uid, env_results in evaluation_result.miner_results.items():
                miner_scores_dict[uid] = {}
                # Find hotkey for this miner
                hotkey = next((m.hotkey for m in miners if m.uid == uid), "unknown")

                for env_id, result in env_results.items():
                    scores_data.append(
                        {
                            "uid": uid,
                            "hotkey": hotkey,
                            "env_id": env_id,
                            "success_rate": result.success_rate,
                            "mean_reward": result.mean_reward,
                            "episodes_completed": result.episodes_completed,
                            "episodes_failed": result.episodes_failed,
                        }
                    )
                    miner_scores_dict[uid][env_id] = result.success_rate

            async with self.storage.session() as session:
                await self.storage.add_miner_scores_bulk(session, cycle_id, scores_data)

            # 4. Compute Pareto scores and weights
            pareto_result = compute_pareto_frontier(
                miner_scores=miner_scores_dict,
                env_ids=self.env_ids,
                n_samples_per_env=self.config.episodes_per_env,
            )

            logger.info(
                "pareto_frontier_computed",
                frontier_size=len(pareto_result.frontier_uids),
            )

            # Compute winners-take-all scores
            epsilons = {
                env_id: float(pareto_result.epsilons[i]) for i, env_id in enumerate(self.env_ids)
            }

            subset_scores = compute_subset_scores(
                miner_scores=miner_scores_dict,
                env_ids=self.env_ids,
                epsilons=epsilons,
            )

            # Convert to weights
            weights = scores_to_weights(
                subset_scores,
                temperature=self.config.pareto_temperature,
            )

            # Convert to u16 for chain submission
            uids, values = weights_to_u16(weights)
            weights_u16 = {"uids": uids, "values": values}

            # 5. Store weights
            async with self.storage.session() as session:
                await self.storage.save_weights(
                    session,
                    cycle_id=cycle_id,
                    block_number=block_number,
                    weights={int(k): float(v) for k, v in weights.items()},
                    weights_u16=weights_u16,
                )

            # 6. Mark cycle as completed
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

    async def trigger_evaluation(self) -> int:
        """
        Manually trigger an evaluation cycle.

        Returns:
            cycle_id of the new evaluation
        """
        logger.info("manual_evaluation_triggered")

        # Run in background task
        self._current_task = asyncio.create_task(self._run_evaluation_cycle())

        # Get the cycle ID from database
        async with self.storage.session() as session:
            cycle = await self.storage.get_running_cycle(session)
            return cycle.id if cycle else -1
