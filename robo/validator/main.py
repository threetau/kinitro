"""Main validator loop for the robotics subnet."""

import asyncio
import time

import structlog

from robo.chain.commitments import read_miner_commitments
from robo.chain.weights import set_weights, verify_weight_setting_eligibility
from robo.config import ValidatorConfig
from robo.environments import get_all_environment_ids
from robo.evaluation.metrics import extract_score_matrix
from robo.evaluation.parallel import evaluate_all_miners
from robo.evaluation.rollout import RolloutConfig
from robo.scoring.pareto import compute_pareto_frontier
from robo.scoring.winners_take_all import compute_subset_scores, scores_to_weights

logger = structlog.get_logger()


class Validator:
    """
    Main validator class for the robotics subnet.

    Runs the evaluation cycle:
    1. Fetch miner commitments from chain
    2. Evaluate all miners on all environments
    3. Compute ε-Pareto dominance scores
    4. Set weights on chain
    """

    def __init__(self, config: ValidatorConfig):
        """
        Initialize validator.

        Args:
            config: Validator configuration
        """
        self.config = config
        self.env_ids = get_all_environment_ids()

        # Lazy-loaded Bittensor objects
        self._subtensor = None
        self._wallet = None
        self._metagraph = None

        # State
        self._last_evaluation_block = 0
        self._evaluation_results = {}

        logger.info(
            "validator_initialized",
            network=config.network,
            netuid=config.netuid,
            n_environments=len(self.env_ids),
        )

    @property
    def subtensor(self):
        """Lazy-load subtensor connection."""
        if self._subtensor is None:
            import bittensor as bt

            self._subtensor = bt.Subtensor(network=self.config.network)
        return self._subtensor

    @property
    def wallet(self):
        """Lazy-load wallet."""
        if self._wallet is None:
            import bittensor as bt

            self._wallet = bt.Wallet(
                name=self.config.wallet_name,
                hotkey=self.config.hotkey_name,
            )
        return self._wallet

    @property
    def metagraph(self):
        """Get current metagraph (refreshed each cycle)."""
        if self._metagraph is None:
            self._metagraph = self.subtensor.metagraph(self.config.netuid)
        return self._metagraph

    def refresh_metagraph(self):
        """Force refresh metagraph."""
        self._metagraph = self.subtensor.metagraph(self.config.netuid)

    async def run(self):
        """
        Main validator loop.

        Runs evaluation cycles at configured intervals.
        """
        logger.info("starting_validator_loop")

        # Check eligibility
        eligible, reason = verify_weight_setting_eligibility(
            self.subtensor, self.wallet, self.config.netuid
        )
        if not eligible:
            logger.error("validator_not_eligible", reason=reason)
            return

        while True:
            try:
                cycle_start = time.time()
                await self.evaluation_cycle()
                cycle_duration = time.time() - cycle_start

                logger.info(
                    "evaluation_cycle_complete",
                    duration_seconds=cycle_duration,
                )

                # Wait for next cycle
                wait_time = max(0, self.config.eval_interval_seconds - cycle_duration)
                if wait_time > 0:
                    logger.info("waiting_for_next_cycle", seconds=wait_time)
                    await asyncio.sleep(wait_time)

            except KeyboardInterrupt:
                logger.info("validator_interrupted")
                break
            except Exception as e:
                logger.error("evaluation_cycle_failed", error=str(e))
                # Wait before retrying
                await asyncio.sleep(60)

    async def evaluation_cycle(self):
        """
        Run a single evaluation cycle.

        Steps:
        1. Refresh metagraph
        2. Fetch miner commitments
        3. Evaluate all miners
        4. Compute scores
        5. Set weights
        """
        block = self.subtensor.block
        logger.info("starting_evaluation_cycle", block=block)

        # Refresh metagraph
        self.refresh_metagraph()

        # 1. Fetch miner commitments
        miners = read_miner_commitments(
            self.subtensor,
            self.config.netuid,
            self.metagraph,
        )

        if not miners:
            logger.warning("no_miners_with_commitments")
            return

        logger.info("found_miners", count=len(miners))

        # 2. Evaluate all miners
        rollout_config = RolloutConfig(
            max_timesteps=self.config.max_timesteps_per_episode,
            action_timeout_ms=self.config.action_timeout_ms,
        )

        evaluation_result = await evaluate_all_miners(
            miners=miners,
            environment_ids=self.env_ids,
            block_number=block,
            validator_hotkey=self.wallet.hotkey.ss58_address,
            episodes_per_env=self.config.episodes_per_env,
            rollout_config=rollout_config,
            use_basilica=bool(self.config.basilica_api_token),
            basilica_api_token=self.config.basilica_api_token,
        )

        self._evaluation_results = evaluation_result.miner_results
        self._last_evaluation_block = block

        # 3. Extract scores for Pareto computation
        uids, score_matrix = extract_score_matrix(evaluation_result, self.env_ids)

        # Convert to dict format for scoring functions
        miner_scores = {
            uid: {
                env_id: evaluation_result.miner_results[uid][env_id].success_rate
                for env_id in self.env_ids
                if env_id in evaluation_result.miner_results.get(uid, {})
            }
            for uid in uids
        }

        # 4. Compute Pareto frontier
        pareto_result = compute_pareto_frontier(
            miner_scores=miner_scores,
            env_ids=self.env_ids,
            n_samples_per_env=self.config.episodes_per_env,
        )

        logger.info(
            "pareto_frontier_computed",
            frontier_size=len(pareto_result.frontier_uids),
            frontier_uids=pareto_result.frontier_uids,
        )

        # 5. Compute winners-take-all scores
        epsilons = {
            env_id: float(pareto_result.epsilons[i]) for i, env_id in enumerate(self.env_ids)
        }

        subset_scores = compute_subset_scores(
            miner_scores=miner_scores,
            env_ids=self.env_ids,
            epsilons=epsilons,
        )

        # 6. Convert to weights
        weights = scores_to_weights(
            subset_scores,
            temperature=self.config.pareto_temperature,
        )

        # Log top weights
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        logger.info(
            "weights_computed",
            top_5=sorted_weights[:5],
            total_miners=len(weights),
        )

        # 7. Set weights on chain
        success = set_weights(
            subtensor=self.subtensor,
            wallet=self.wallet,
            netuid=self.config.netuid,
            weights=weights,
        )

        if success:
            logger.info("weights_set_successfully", block=block)
        else:
            logger.error("weights_set_failed", block=block)


async def run_validator(config: ValidatorConfig):
    """
    Entry point for running the validator.

    Args:
        config: Validator configuration
    """
    validator = Validator(config)
    await validator.run()
