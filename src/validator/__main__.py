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
from core.db.db_manager import DatabaseManager
from core.db.models import EvaluationJob, EvaluationStatus
from core.log import get_logger
from core.neuron import Neuron
from core.schemas import ChainCommitmentResponse

from .config import ValidatorConfig
from .rpc.parent_server import ParentValidatorServer, start_parent_server
from .rpc.child_server import ChildValidatorManager

logger = get_logger(__name__)


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
        self.last_seen_block: int = 0

        # Parent-child validator setup
        self.is_parent_validator = config.settings.get("is_parent_validator", False)
        self.parent_server = None
        self.child_manager = None

        if self.is_parent_validator:
            logger.info("Starting as parent validator")
            self.parent_server = ParentValidatorServer(self)
        else:
            logger.info("Starting as child validator")
            validator_id = config.settings.get("validator_id")
            if not validator_id:
                validator_id = f"validator_{self.keypair.ss58_address[:8]}"
                logger.info(f"Generated validator ID: {validator_id}")

            parent_host = config.settings.get("parent_host", "localhost")
            parent_port = config.settings.get("parent_port", 8001)
            child_port = config.settings.get("child_port", 8002)

            self.child_manager = ChildValidatorManager(
                self, validator_id, child_port, parent_host, parent_port
            )

        # Initialize database manager if database URL is provided
        self.db_manager = None
        if self.config.settings.get("pg_database", None):
            logger.info(f"Using Postgres DB at {self.config.settings['pg_database']}")
            self.db_manager = DatabaseManager(
                postgres_url=self.config.settings["pg_database"]
            )
            self.db_manager.initialize_databases()
        else:
            logger.warning("No Postgres DB configured, using in-memory database.")
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

        # Load state from database if available, otherwise initialize
        if self.db_manager:
            self._load_validator_state()
        else:
            # Initialize last seen block bounded by the max lookback window
            latest_block_number = self.substrate.get_block_number()
            # Start at the edge of the lookback window so we don't scan unbounded history
            self.last_seen_block: int = max(
                self.last_seen_block, latest_block_number - self.max_commitment_lookback
            )

    def _load_validator_state(self):
        """Load validator state from database on startup."""
        if not self.db_manager:
            return

        try:
            # Get validator hotkey (using a placeholder for now)
            validator_hotkey = self.keypair.ss58_address

            # Get or create validator state
            validator_state = self.db_manager.get_or_create_validator_state(
                validator_hotkey
            )
            self.last_seen_block = validator_state.last_seen_block

            # Initialize last seen block bounded by the max lookback window
            latest_block_number = self.substrate.get_block_number()
            # Start at the edge of the lookback window so we don't scan unbounded history
            self.last_seen_block = max(
                self.last_seen_block, latest_block_number - self.max_commitment_lookback
            )

            # Load commitment fingerprints
            fingerprints = self.db_manager.get_all_commitment_fingerprints()
            self._last_commitment_fingerprint_by_hotkey = {
                fp.miner_hotkey: fp.fingerprint for fp in fingerprints
            }

            logger.info(
                f"Loaded validator state: last_seen_block={self.last_seen_block}, "
                f"fingerprints_count={len(self._last_commitment_fingerprint_by_hotkey)}"
            )

        except Exception as e:
            logger.error(f"Failed to load validator state: {e}")
            # Fallback to default initialization
            latest_block_number = self.substrate.get_block_number()
            self.last_seen_block = max(
                0, latest_block_number - self.max_commitment_lookback
            )

    def _save_validator_state(self):
        """Save current validator state to database."""
        if not self.db_manager:
            return

        try:
            validator_hotkey = self.keypair.ss58_address

            # Update last seen block
            self.db_manager.update_validator_last_seen_block(
                validator_hotkey, self.last_seen_block
            )

            logger.debug(
                f"Saved validator state: last_seen_block={self.last_seen_block}"
            )

        except Exception as e:
            logger.error(f"Failed to save validator state: {e}")

    def _save_commitment_fingerprint(self, miner_hotkey: str, fingerprint: str):
        """Save commitment fingerprint to database."""
        if not self.db_manager:
            return

        try:
            self.db_manager.get_or_create_commitment_fingerprint(
                miner_hotkey, fingerprint
            )
            logger.debug(
                f"Saved commitment fingerprint for {miner_hotkey}: {fingerprint}"
            )

        except Exception as e:
            logger.error(
                f"Failed to save commitment fingerprint for {miner_hotkey}: {e}"
            )

    async def run(self):
        """
        Run the validator
        """

        try:
            self.background_tasks()
        except Exception as e:
            logger.error(f"Error in validator run loop: {e}")
        finally:
            # Clean up parent-child resources
            if self.is_parent_validator and hasattr(self, "parent_rpc_thread"):
                try:
                    # RPC thread will stop when main process exits (daemon=True)
                    logger.info("Parent RPC thread cleanup initiated")
                except Exception as e:
                    logger.error(f"Error stopping parent RPC thread: {e}")
            elif not self.is_parent_validator and self.child_manager:
                try:
                    # Simply disable job polling and let the thread terminate naturally
                    self.child_manager.job_polling_enabled = False
                    logger.info("Child manager cleanup initiated")
                except Exception as e:
                    logger.error(f"Error stopping child manager: {e}")

            # Save state on shutdown
            if self.db_manager:
                self._save_validator_state()
                self.db_manager.close_connections()

    def background_tasks(self):
        if self.is_parent_validator:
            # Parent validator tasks
            self.sync_metagraph_thread = Thread(target=self.sync_metagraph, daemon=True)
            self.sync_metagraph_thread.start()

            self.job_queuer_thread = Thread(target=self.queue_jobs, daemon=True)
            self.job_queuer_thread.start()

            # Start parent RPC server in separate thread (with logging visible)
            import threading
            import logging
            import sys

            # Configure logging for RPC thread to show up in main output
            rpc_logger = logging.getLogger("validator.rpc")
            rpc_handler = logging.StreamHandler(sys.stdout)
            rpc_handler.setFormatter(
                logging.Formatter("%(asctime)s | RPC | %(levelname)s | %(message)s")
            )
            rpc_logger.addHandler(rpc_handler)
            rpc_logger.setLevel(logging.DEBUG)

            self.parent_rpc_thread = threading.Thread(
                target=start_parent_server,
                args=(self, "127.0.0.1", self.config.settings.get("parent_port", 8001)),
                daemon=True,
            )
            self.parent_rpc_thread.start()
            logger.info(
                f"Started parent RPC server on port {self.config.settings.get('parent_port', 8001)}"
            )
        else:
            # Child validator tasks - start child manager
            import asyncio
            import threading
            import capnp
            import logging
            import sys

            # Configure logging for child RPC to show up in main output
            child_rpc_logger = logging.getLogger("validator.rpc")
            child_rpc_handler = logging.StreamHandler(sys.stdout)
            child_rpc_handler.setFormatter(
                logging.Formatter(
                    "%(asctime)s | CHILD-RPC | %(levelname)s | %(message)s"
                )
            )
            child_rpc_logger.addHandler(child_rpc_handler)
            child_rpc_logger.setLevel(logging.DEBUG)

            def run_child_manager():
                async def run_with_kj():
                    async with capnp.kj_loop():
                        await self.child_manager.start()
                        # Start job polling loop
                        await self.child_manager._job_polling_loop()

                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(run_with_kj())

            self.child_thread = threading.Thread(target=run_child_manager, daemon=True)
            self.child_thread.start()
            logger.info("Started child validator manager")

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
                                # Update in-memory cache
                                self._last_commitment_fingerprint_by_hotkey[
                                    miner_hotkey
                                ] = fingerprint

                                # Save to database if available
                                self._save_commitment_fingerprint(
                                    miner_hotkey, fingerprint
                                )

                                logger.debug(
                                    f"Block {block_num} - Miner {miner_hotkey} new/updated commitment: {commitment}"
                                )
                                self.job_queue.put_nowait(commitment)
                                total_commitments += 1

                                # If we have a parent server, add job to its queue
                                if self.parent_server:
                                    self.parent_server.add_job_to_queue(commitment)

                logger.info(
                    f"Processed {total_commitments} new commitments from chain (skipped {total_skipped_unchanged} unchanged). Previous last seen block {self.last_seen_block}. Latest block is {latest_block}."
                )

                # Update last seen block and save to database
                self.last_seen_block = latest_block
                self._save_validator_state()

                time.sleep(self.config.settings["neuron"]["sync_frequency"])
            except Exception as e:
                logger.error(f"Error syncing metagraph: {e}")
                time.sleep(self.config.settings["neuron"]["sync_frequency"] // 2)

    def queue_jobs(self):
        """
        Queue up evaluation jobs to process models submitted through chain commitments.

        For the parent validator, it queues the job for itself and distributes to child validators.
        """

        if not self.is_parent_validator:
            logger.warning("Not a parent validator, skipping job queuing.")
            return

        while True:
            if self.job_queue.empty():
                time.sleep(1)
                continue

            commitment = self.job_queue.get_nowait()

            # Process job for parent validator itself (if needed)
            # TODO: implement local job processing if desired

            # Job distribution to child validators is handled by ParentValidatorServer
            # when children request jobs via RPC
            logger.debug(f"Processed commitment from {commitment.hotkey}")

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
