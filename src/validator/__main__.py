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

from .child_receiver import ChildValidatorReceiver
from .config import ValidatorConfig
from .parent_broadcaster import ParentValidatorBroadcaster

logger = get_logger(__name__)


FALLBACK_MAX_COMMITMENT_LOOKBACK = 360
JOB_QUEUE_POLL_INTERVAL = 0.1  # Sleep duration when job queue is empty


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

        # Parent-child validator components
        self.is_parent = config.settings.get("is_parent", False)
        self.parent_broadcaster = None
        self.child_receiver = None

        # Shutdown control
        self._shutdown_requested = False
        self.sync_metagraph_thread = None
        self.job_queuer_thread = None

        # Initialize database manager if database URL is provided
        self.db_manager = None
        if self.config.settings.get("pg_database", None):
            pg_url = self.config.settings["pg_database"]
            duckdb_path = self.config.settings.get(
                "duckdb_path", "evaluation_data.duckdb"
            )

            logger.info(f"Using Postgres DB at {pg_url}")
            logger.info(f"Using DuckDB at {duckdb_path}")

            self.db_manager = DatabaseManager(
                postgres_url=pg_url, duckdb_path=duckdb_path
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

        # Initialize parent-child components
        self._init_parent_child_components()

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

    def _init_parent_child_components(self):
        """Initialize parent or child validator components based on configuration."""
        if self.is_parent:
            # Initialize parent broadcaster
            broadcast_host = self.config.settings.get("broadcast_host", "localhost")
            broadcast_port = self.config.settings.get("broadcast_port", 8765)
            self.parent_broadcaster = ParentValidatorBroadcaster(
                broadcast_host, broadcast_port
            )
            logger.info(
                f"Initialized as parent validator (broadcasting on {broadcast_host}:{broadcast_port})"
            )
        else:
            # Initialize child receiver
            parent_host = self.config.settings.get("parent_host", "localhost")
            parent_port = self.config.settings.get("parent_port", 8765)
            validator_hotkey = self.keypair.ss58_address

            self.child_receiver = ChildValidatorReceiver(
                parent_host, parent_port, validator_hotkey, self._handle_received_job
            )
            logger.info(
                f"Initialized as child validator (connecting to {parent_host}:{parent_port})"
            )

    async def graceful_shutdown(self):
        """Gracefully shutdown the validator by waiting for background threads to complete."""
        logger.info("Initiating graceful shutdown...")

        # Set shutdown flag to signal threads to stop
        self._shutdown_requested = True

        # Wait for background threads to complete
        if self.sync_metagraph_thread and self.sync_metagraph_thread.is_alive():
            logger.info("Waiting for metagraph sync to complete...")
            self.sync_metagraph_thread.join(timeout=10)  # Wait max 10 seconds
            if self.sync_metagraph_thread.is_alive():
                logger.warning("Metagraph sync thread did not stop within timeout")

        if self.job_queuer_thread and self.job_queuer_thread.is_alive():
            logger.info("Waiting for job queue processing to complete...")
            self.job_queuer_thread.join(timeout=10)  # Wait max 10 seconds
            if self.job_queuer_thread.is_alive():
                logger.warning("Job queuer thread did not stop within timeout")

        logger.info("Background threads stopped")

    async def _handle_received_job(self, commitment: ChainCommitmentResponse) -> bool:
        """Handle job received from parent validator.

        Returns:
            bool: True if job was successfully queued, False otherwise.
        """
        try:
            # Convert commitment to evaluation job
            job = self._create_evaluation_job_from_commitment(commitment)

            # Queue job in child validator's PostgreSQL database using pgqueuer
            await self._queue_job_with_pgqueuer(job)
            logger.info(
                f"Queued job from parent: {job.hf_repo_id} (job_id: {job.id}, "
                f"miner: {job.miner_hotkey}, env: {job.env_provider}/{job.env_name}, "
                f"status: {str(job.status)}, submission_id: {job.submission_id})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to queue job from parent: {e}")
            return False

    async def run(self):
        """
        Run the validator
        """

        try:
            # Start parent-child components
            if self.is_parent and self.parent_broadcaster:
                await self.parent_broadcaster.start_server()
            elif not self.is_parent and self.child_receiver:
                await self.child_receiver.start()

            # Start background tasks
            self.background_tasks()

            # Keep the validator running indefinitely
            logger.info("Validator is now running. Press Ctrl+C to stop.")
            while not self._shutdown_requested:
                await asyncio.sleep(1)
            logger.info("Shutdown requested, exiting run loop...")

        except KeyboardInterrupt:
            logger.info("Received shutdown signal, stopping validator...")
            self._shutdown_requested = True
        except Exception as e:
            logger.error(f"Error in validator run loop: {e}")
        finally:
            # Gracefully shutdown background threads
            await self.graceful_shutdown()

            # Stop parent-child components
            if self.parent_broadcaster:
                await self.parent_broadcaster.stop_server()
            if self.child_receiver:
                await self.child_receiver.stop()

            # Save state on shutdown
            if self.db_manager:
                self._save_validator_state()
                self.db_manager.close_connections()

    def background_tasks(self):
        if self.is_parent:
            self.sync_metagraph_thread = Thread(
                target=self.sync_metagraph, daemon=False
            )
            self.sync_metagraph_thread.start()

            self.job_queuer_thread = Thread(target=self.queue_jobs, daemon=False)
            self.job_queuer_thread.start()
        else:
            # Child validators receive jobs from parent, no need for chain querying
            logger.info("Child validator: waiting for jobs from parent")

    def sync_metagraph(self):
        """
        Sync nodes with the metagraph and query miner commitments from the chain.
        """
        while not self._shutdown_requested:
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
                    latest_block - self.max_commitment_lookback + 1,
                    0,
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

                logger.info(
                    f"Processed {total_commitments} new commitments from chain (skipped {total_skipped_unchanged} unchanged). Previous last seen block {self.last_seen_block}. Latest block is {latest_block}."
                )

                # Update last seen block and save to database
                self.last_seen_block = latest_block
                self._save_validator_state()

                # Interruptible sleep
                sync_freq = self.config.settings["neuron"]["sync_frequency"]
                for _ in range(sync_freq):
                    if self._shutdown_requested:
                        break
                    time.sleep(1)
            except Exception as e:
                logger.error(f"Error syncing metagraph: {e}")
                # Interruptible error sleep
                error_sleep = self.config.settings["neuron"]["sync_frequency"] // 2
                for _ in range(error_sleep):
                    if self._shutdown_requested:
                        break
                    time.sleep(1)

        logger.info("Metagraph sync thread exiting cleanly")

    def queue_jobs(self):
        """
        Queue up evaluation jobs to process models submitted through chain commitments.

        For the control validator, it queues the job for itself, as well as sending the
        jobs to other validators.

        A validator that is not the control validator will not run this function.
        """

        if not self.is_parent:
            logger.warning("Not a parent validator, skipping job queuing.")
            return

        while not self._shutdown_requested:
            if self.job_queue.empty():
                # Small sleep to avoid busy waiting
                time.sleep(JOB_QUEUE_POLL_INTERVAL)
                continue

            commitment = self.job_queue.get_nowait()

            # Convert commitment to evaluation job
            try:
                job = self._create_evaluation_job_from_commitment(commitment)

                # Queue job locally in parent validator's database
                asyncio.run(self._queue_job_with_pgqueuer(job))
                logger.info(f"Queued job locally: {job.hf_repo_id} (job_id: {job.id})")

            except Exception as e:
                logger.error(f"Failed to queue job locally: {e}")

            # Broadcast job to child validators
            if self.parent_broadcaster:
                try:
                    child_count = asyncio.run(
                        self.parent_broadcaster.broadcast_job(commitment)
                    )
                    logger.info(f"Broadcasted job to {child_count} child validators")
                except Exception as e:
                    logger.error(f"Failed to broadcast job to children: {e}")
            else:
                logger.debug("No parent broadcaster available")

        logger.info("Job queuer thread exiting cleanly")

    def _create_evaluation_job_from_commitment(
        self, commitment: ChainCommitmentResponse
    ) -> EvaluationJob:
        """Convert a ChainCommitmentResponse to an EvaluationJob."""
        gen = SnowflakeGenerator(42)
        job_id = next(gen)
        sub_id = next(gen)

        # Extract data from commitment
        data = commitment.data

        job = EvaluationJob(
            created_at=datetime.now(),
            updated_at=datetime.now(),
            status=EvaluationStatus.QUEUED,
            submission_id=sub_id,
            miner_hotkey=commitment.hotkey,
            hf_repo_id=data.repo_id,
            env_provider="metaworld",
            env_name="MT10",
            id=job_id,
            container_id=None,
            ray_worker_id=None,
            retry_count=0,
            max_retries=3,
            logs_path="./data/logs",
            random_seed=None,
            eval_start=None,
            eval_end=None,
        )

        return job

    async def _queue_job_with_pgqueuer(self, job: EvaluationJob):
        """Queue a job using pgqueuer into PostgreSQL."""
        if not self.db_manager:
            logger.error("No database manager available - cannot queue job")
            return

        try:
            # Get PostgreSQL connection URL
            pg_url = self.config.settings["pg_database"]

            # Create connection and queue the job
            conn = await asyncpg.connect(pg_url)
            driver = AsyncpgDriver(conn)
            q = Queries(driver)
            job_bytes = job.to_bytes()
            await q.enqueue(["add_job"], [job_bytes], [0])
            await conn.close()

            logger.info(
                f"Queued job {job.id} for miner {job.miner_hotkey} (repo: {job.hf_repo_id})"
            )

        except Exception as e:
            logger.error(f"Failed to queue job {job.id}: {e}")
            raise

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

    try:
        await validator.run()
    except KeyboardInterrupt:
        logger.info("Validator shutdown complete")
    except Exception as e:
        logger.error(f"Validator error: {e}")
        raise


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nValidator stopped by user")
        print("Hit Ctrl+C again to shut down")
        exit(0)
