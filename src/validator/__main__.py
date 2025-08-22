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


class Validator(Neuron):
    def __init__(self, config: ValidatorConfig):
        super().__init__(config)
        self.config = config
        self.job_queue: Queue[ChainCommitmentResponse] = Queue()
        self.validators_to_query: list[str] = []
        self.validators_to_query_mutex = Lock()
        self.miners_to_query: list[str] = []

        # TODO: query the actual last seen block from the database
        self.last_seen_block: int = 6240964

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
                self.metagraph.sync_nodes()
                logger.debug("Metagraph synced successfully")

                # Update validators to query
                validators_to_query: list[str] = []
                for hotkey, node_info in self.metagraph.nodes.items():
                    if (
                        node_info.stake
                        >= self.config.settings["neuron"]["min_stake_threshold"]
                        and hotkey
                        in self.config.settings["neuron"]["allowed_validators"]
                    ):
                        validators_to_query.append(hotkey)
                    else:
                        # We can basically assume that the other nodes are miners
                        self.miners_to_query.append(hotkey)

                logger.debug("Filtered validators and miners")

                # Lock and replace
                with self.validators_to_query_mutex:
                    self.validators_to_query = validators_to_query

                # Get commitments from chain since the last block we've seen
                latest_block = self.substrate.get_block_number()
                for i in range(self.last_seen_block + 1, latest_block + 1):
                    for miner_hotkey in self.miners_to_query:
                        commitments = query_commitments_from_substrate(
                            self.config, miner_hotkey, block=i
                        )
                        self.job_queue.put_nowait(*commitments)

                logger.debug(
                    f"Got commitments from chain since last seen block of {self.last_seen_block}. Latest block is {latest_block}."
                )
                self.last_seen_block = latest_block

                time.sleep(self.config.settings["sync_frequency"])
            except Exception as e:
                logger.error(f"Error syncing metagraph: {e}")
                time.sleep(self.config.settings["sync_frequency"] // 2)

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
