import asyncio
import time
from datetime import datetime
from multiprocessing import Lock, Queue
from threading import Thread

import asyncpg
from pgqueuer.db import AsyncpgDriver
from pgqueuer.queries import Queries
from snowflake import SnowflakeGenerator

from core.chain import query_commitments_from_substrate
from core.db.models import EvaluationJob, EvaluationStatus
from core.log import get_logger
from core.neuron import Neuron
from core.schemas import ChainCommitmentResponse

from .config import ValidatorConfig

logger = get_logger(__name__)

# TODO: make this a proper config option
IS_CONTROL_VALIDATOR = True

FALLBACK_MAX_COMMITMENT_LOOKBACK = 360


class Validator(Neuron):
    def __init__(self, config: ValidatorConfig):
        super().__init__(config)
        self.config = config
        self.job_queue: Queue[ChainCommitmentResponse] = Queue()
        self.validators_to_query: list[str] = []
        self.validators_to_query_mutex = Lock()
        self.miners_to_query: list[str] = []
        self._last_commitment_fingerprint_by_hotkey: dict[str, str] = {}

        # Max lookback window (in blocks) to cap historical queries
        try:
            lookback_from_section = self.config.settings["neuron"].get(
                "max_commitment_lookback", None
            )
            lookback_from_top_level = self.config.settings.get(
                "max_commitment_lookback", None
            )
            effective_lookback = (
                lookback_from_top_level
                if lookback_from_top_level is not None
                else lookback_from_section
            )
            self.max_commitment_lookback: int = int(
                FALLBACK_MAX_COMMITMENT_LOOKBACK
                if effective_lookback is None
                else effective_lookback
            )
        except Exception:
            self.max_commitment_lookback = FALLBACK_MAX_COMMITMENT_LOOKBACK
        if self.max_commitment_lookback < 1:
            self.max_commitment_lookback = 1

        # Initialize last seen block bounded by the max lookback window
        latest_block_number = self.substrate.get_block_number()
        # Start at the edge of the lookback window so we don't scan unbounded history
        self.last_seen_block: int = max(
            latest_block_number - self.max_commitment_lookback, 0
        )

    async def run(self):
        """
        Run the validator
        """

        try:
            self.background_tasks()
        except Exception as e:
            logger.error(f"Error in validator run loop: {e}")

    def background_tasks(self):
        if IS_CONTROL_VALIDATOR:
            self.sync_metagraph_thread = Thread(target=self.sync_metagraph, daemon=True)
            self.sync_metagraph_thread.start()

            self.job_queuer_thread = Thread(target=self.queue_jobs, daemon=True)
            self.job_queuer_thread.start()
        else:
            # TODO: any tasks specific to non-control validators
            ...

    def sync_metagraph(self):
        """
        Sync nodes with the metagraph and query miner commitments from the chain.
        """
        while True:
            try:
                # Sync the metagraph to get the latest nodes
                self.metagraph.sync_nodes()
                logger.info("Metagraph synced successfully.")

                # Prepare fresh lists for validators and miners
                validators_to_query: list[str] = []
                miners_to_query: list[str] = []

                min_stake = self.config.settings["neuron"]["min_stake_threshold"]
                allowed_validators = self.config.settings["neuron"][
                    "allowed_validators"
                ]

                # Classify nodes as validators or miners
                for hotkey, node_info in self.metagraph.nodes.items():
                    if node_info.stake >= min_stake and hotkey in allowed_validators:
                        validators_to_query.append(hotkey)
                    else:
                        miners_to_query.append(hotkey)

                logger.info(
                    f"Filtered {len(validators_to_query)} validators and {len(miners_to_query)} miners from metagraph."
                )

                # Update the shared lists with mutex protection
                with self.validators_to_query_mutex:
                    self.validators_to_query = validators_to_query
                    self.miners_to_query = miners_to_query

                # Query commitments from chain for new blocks, capped by max lookback
                latest_block = self.substrate.get_block_number()
                start_block = max(
                    self.last_seen_block + 1,
                    max(latest_block - self.max_commitment_lookback + 1, 0),
                )
                logger.info(
                    f"Querying commitments from block {start_block} to {latest_block} (max lookback {self.max_commitment_lookback})."
                )

                total_commitments = 0
                total_skipped_unchanged = 0
                for block_num in range(start_block, latest_block + 1):
                    for miner_hotkey in self.miners_to_query:
                        commitments = query_commitments_from_substrate(
                            self.config, miner_hotkey, block=block_num
                        )
                        if commitments:
                            for commitment in commitments:
                                fingerprint = self._compute_commitment_fingerprint(
                                    commitment
                                )
                                previous_fingerprint = (
                                    self._last_commitment_fingerprint_by_hotkey.get(
                                        miner_hotkey
                                    )
                                )
                                if previous_fingerprint == fingerprint:
                                    total_skipped_unchanged += 1
                                    logger.debug(
                                        f"Skipping unchanged commitment for miner {miner_hotkey} at block {block_num}"
                                    )
                                    continue
                                self._last_commitment_fingerprint_by_hotkey[
                                    miner_hotkey
                                ] = fingerprint
                                logger.debug(
                                    f"Block {block_num} - Miner {miner_hotkey} new/updated commitment: {commitment}"
                                )
                                self.job_queue.put_nowait(commitment)
                                total_commitments += 1

                logger.info(
                    f"Processed {total_commitments} new commitments from chain (skipped {total_skipped_unchanged} unchanged). Previous last seen block {self.last_seen_block}. Latest block is {latest_block}."
                )
                self.last_seen_block = latest_block

                time.sleep(self.config.settings["neuron"]["sync_frequency"])
            except Exception as e:
                logger.error(f"Error syncing metagraph: {e}")
                time.sleep(self.config.settings["neuron"]["sync_frequency"] // 2)

    def queue_jobs(self):
        """
        Queue up evaluation jobs to process models submitted through chain commitments.

        For the control validator, it queues the job for itself, as well as sending the
        jobs to other validators.

        A validator that is not the control validator will not run this function.
        """

        if not IS_CONTROL_VALIDATOR:
            logger.warning("Not a control validator, skipping job queuing.")
            return

        while True:
            if self.job_queue.empty():
                continue

            commitment = self.job_queue.get_nowait()
            # TODO: process job for itself

            # Process job for other validators
            with self.validators_to_query_mutex:
                for validator in self.validators_to_query:
                    _ = validator
                    # TODO: use RPC to send to other validators

    async def send_job(self):
        """
        Hand off the job to the orchestrator
        """
        gen = SnowflakeGenerator(42)
        job_id = next(gen)
        sub_id = next(gen)

        # TODO: don't hardcode this
        job = EvaluationJob(
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=EvaluationStatus.QUEUED,
            submission_id=sub_id,  # type: ignore
            miner_hotkey="5CyY97KCfwRC5UZN58A1cLpZnMgSZAKWtqaaggUfzYiJ6B8d",
            hf_repo_id="rishiad/default_submission",
            hf_repo_commit="93a2aa6de1069bcc37c60e80954d3a2c6e202678",
            env_provider="metaworld",
            env_name="MT10",
            id=job_id,  # type: ignore
            container_id=None,
            ray_worker_id=None,
            retry_count=0,
            max_retries=3,
            logs_path="./data/logs",
            random_seed=None,
            eval_start=None,
            eval_end=None,
        )

        conn = await asyncpg.connect()
        driver = AsyncpgDriver(conn)
        q = Queries(driver)
        job_bytes = job.to_bytes()
        await q.enqueue(["add_job"], [job_bytes], [0])

        # TODO

    def _compute_commitment_fingerprint(
        self, commitment_response: ChainCommitmentResponse
    ):
        """Create a stable fingerprint to detect changes per miner hotkey.

        Uses key fields from the commitment payload; update this if the schema evolves.
        """
        data = commitment_response.data
        # Include version to differentiate schema updates
        return f"{data.version}|{data.provider}|{data.repo_id}"


async def main():
    config = ValidatorConfig()
    validator = Validator(config)
    await validator.run()

    logger.info(f"Validator running... timestamp: {time.time()}")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping validator...")
        # await validator.stop()


if __name__ == "__main__":
    asyncio.run(main())
