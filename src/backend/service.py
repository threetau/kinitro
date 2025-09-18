"""

This provides REST API endpoints and WebSocket connections for:

- Competition management
- Validator connections
- Job distribution
- Result collection
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

from fastapi import (
    WebSocket,
)
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from snowflake import SnowflakeGenerator
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.constants import (
    CHAIN_SCAN_YIELD_INTERVAL,
    DEFAULT_CHAIN_SYNC_INTERVAL,
    DEFAULT_MAX_COMMITMENT_LOOKBACK,
    HEARTBEAT_INTERVAL,
    MAX_WORKERS,
    SCORE_EVALUATION_INTERVAL,
    SCORE_EVALUATION_STARTUP_DELAY,
    WEIGHT_BROADCAST_INTERVAL,
    WEIGHT_BROADCAST_STARTUP_DELAY,
)
from core.chain import query_commitments_from_substrate
from core.db.models import EvaluationStatus
from core.log import get_logger
from core.messages import (
    EvalJobMessage,
    SetWeightsMessage,
)
from core.schemas import ChainCommitmentResponse

from .config import BackendConfig
from .models import (
    BackendEvaluationJob,
    BackendEvaluationJobStatus,
    BackendEvaluationResult,
    BackendState,
    Competition,
    MinerSubmission,
    SS58Address,
    ValidatorConnection,
)

logger = get_logger(__name__)

ConnectionId = str  # Unique ID for each WebSocket connection


class BackendService:
    """Core backend service logic."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.db_url = config.settings.get("database_url")

        # Chain monitoring configuration
        self.max_commitment_lookback = config.settings.get(
            "max_commitment_lookback", DEFAULT_MAX_COMMITMENT_LOOKBACK
        )
        self.chain_sync_interval = config.settings.get(
            "chain_sync_interval", DEFAULT_CHAIN_SYNC_INTERVAL
        )

        # Scoring and weight broadcast intervals
        self.score_evaluation_interval = config.settings.get(
            "score_evaluation_interval", SCORE_EVALUATION_INTERVAL
        )
        self.weight_broadcast_interval = config.settings.get(
            "weight_broadcast_interval", WEIGHT_BROADCAST_INTERVAL
        )

        # Chain connection objects
        # Using Any since fiber's SubstrateInterface is from async_substrate_interface
        self.substrate: Optional[Any] = (
            None  # async_substrate_interface.sync_substrate.SubstrateInterface
        )
        self.metagraph: Optional[Metagraph] = None

        # WebSocket connections
        self.active_connections: Dict[ConnectionId, WebSocket] = {}
        self.validator_connections: Dict[ConnectionId, SS58Address] = {}

        # Background tasks
        self._running = False
        self._chain_monitor_task = None
        self._heartbeat_monitor_task = None
        self._score_evaluation_task = None
        self._weight_broadcast_task = None

        # Store latest scores for weight broadcasting
        self._latest_miner_scores: Dict[SS58Address, float] = {}

        # ID generator
        self.id_generator = SnowflakeGenerator(42)

        # Database
        self.engine: AsyncEngine = None
        self.async_session: async_sessionmaker[AsyncSession] = None

        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    async def startup(self) -> None:
        """Initialize the backend service without starting background tasks."""
        logger.info("Initializing Kinitro Backend Service")

        # Initialize database first
        await self._init_database()

        # Initialize chain connection
        await self._init_chain()

        # Load backend state
        await self._load_backend_state()

        self._running = True
        logger.info("Kinitro Backend Service initialized successfully")

    async def start_background_tasks(self) -> None:
        """Start background tasks after FastAPI is ready."""
        logger.info("Starting background tasks")

        # Start core monitoring tasks first
        self._chain_monitor_task = asyncio.create_task(self._monitor_chain())
        self._heartbeat_monitor_task = asyncio.create_task(
            self._monitor_validator_heartbeats()
        )

        # Delay before starting scoring
        logger.info(
            f"Starting score evaluation task in {SCORE_EVALUATION_STARTUP_DELAY} seconds..."
        )
        await asyncio.sleep(SCORE_EVALUATION_STARTUP_DELAY)
        self._score_evaluation_task = asyncio.create_task(
            self._periodic_score_evaluation()
        )

        # Delay before starting weight broadcast
        logger.info(
            f"Starting weight broadcast task in {WEIGHT_BROADCAST_STARTUP_DELAY} seconds..."
        )
        await asyncio.sleep(WEIGHT_BROADCAST_STARTUP_DELAY)
        self._weight_broadcast_task = asyncio.create_task(
            self._periodic_weight_broadcast()
        )

        logger.info(
            f"All background tasks started. "
            f"Score evaluation interval: {self.score_evaluation_interval}s, "
            f"Weight broadcast interval: {self.weight_broadcast_interval}s"
        )

    async def shutdown(self) -> None:
        """Shutdown the backend service."""
        logger.info("Shutting down Kinitro Backend Service")

        self._running = False

        # Cancel background tasks
        tasks_to_cancel = [
            (self._chain_monitor_task, "chain monitor"),
            (self._heartbeat_monitor_task, "heartbeat monitor"),
            (self._score_evaluation_task, "score evaluation"),
            (self._weight_broadcast_task, "weight broadcast"),
        ]

        for task, name in tasks_to_cancel:
            if task:
                logger.info(f"Cancelling {name} task")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError as e:
                    logger.error(f"{name} task cancelled: {e}")

        # Close WebSocket connections
        for ws in self.active_connections.values():
            await ws.close()

        # Close database
        if self.engine:
            await self.engine.dispose()

        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)

        logger.info("Backend Service shut down")

    async def _init_chain(self) -> None:
        """Initialize blockchain connection."""
        try:
            logger.info("Initializing blockchain connection...")

            self.substrate = get_substrate(
                subtensor_network=self.config.settings["subtensor"]["network"],
                subtensor_address=self.config.settings["subtensor"]["address"],
            )

            self.metagraph = Metagraph(
                netuid=self.config.settings["subtensor"]["netuid"],
                substrate=self.substrate,
            )

            logger.info("Blockchain connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connection: {e}")
            raise e

    async def _init_database(self) -> None:
        """Initialize database connection."""
        self.engine = create_async_engine(
            self.db_url, echo=False, pool_pre_ping=True, pool_size=20, max_overflow=0
        )

        self.async_session = async_sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        logger.info("Database connection initialized")

    async def _load_backend_state(self) -> None:
        """Load or initialize backend service state.

        This loads the singleton BackendState record which tracks:
        - Chain monitoring state (last seen block number, last chain scan time)
        - Service metadata (version, start time)
        - Persistence across service restarts
        """
        if not self.async_session:
            logger.error("Database not initialized")
            return
        async with self.async_session() as session:
            result = await session.execute(
                select(BackendState).where(BackendState.id == 1)
            )
            state = result.scalar_one_or_none()

            if not state:
                state = BackendState(id=1, last_seen_block=0, service_version="1.0.0")
                session.add(state)
                await session.commit()
                logger.info("Initialized new backend state")
            else:
                state.service_start_time = datetime.now(timezone.utc)
                await session.commit()
                logger.info(
                    f"Loaded backend state: last_seen_block={state.last_seen_block}"
                )

    async def _monitor_chain(self) -> None:
        """Background task to monitor blockchain for commitments."""
        while self._running:
            try:
                if self.substrate and self.metagraph and self.async_session:
                    await self._sync_metagraph()

                    async with self.async_session() as session:
                        # Get backend state
                        state_result = await session.execute(
                            select(BackendState).where(BackendState.id == 1)
                        )
                        state = state_result.scalar_one()

                        # Get latest block
                        latest_block = await self._get_latest_block()
                        start_block = max(
                            state.last_seen_block + 1,
                            latest_block - self.max_commitment_lookback + 1,
                        )

                        logger.info(f"Checking blocks {start_block} to {latest_block}")

                        # Get active competitions
                        comp_result = await session.execute(
                            select(Competition).where(Competition.active)
                        )

                        active_competitions = {c.id: c for c in comp_result.scalars()}
                        # preview active competitions
                        logger.debug(
                            f"Preview of active competitions: {list(active_competitions.keys())[:5]}"
                        )

                        # Query commitments (with yield points to prevent blocking)
                        for i, block_num in enumerate(
                            range(start_block, latest_block + 1)
                        ):
                            commitments = await self._query_block_commitments(block_num)
                            for commitment in commitments:
                                await self._process_commitment(
                                    commitment, block_num, active_competitions
                                )

                            # Yield control periodically to prevent blocking WebSocket connections
                            if i % CHAIN_SCAN_YIELD_INTERVAL == 0:
                                await asyncio.sleep(0)

                        # Update state
                        state.last_seen_block = latest_block
                        state.last_chain_scan = datetime.now(timezone.utc)
                        await session.commit()

                await asyncio.sleep(self.chain_sync_interval)

            except Exception as e:
                logger.error(f"Error monitoring chain: {e}")
                await asyncio.sleep(self.chain_sync_interval)

    async def _monitor_validator_heartbeats(self) -> None:
        """Monitor validator heartbeats and cleanup stale connections."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                timeout_threshold = current_time - timedelta(minutes=2)

                if self.async_session:
                    async with self.async_session() as session:
                        # Find stale validators
                        result = await session.execute(
                            select(ValidatorConnection).where(
                                and_(
                                    ValidatorConnection.is_connected,
                                    ValidatorConnection.last_heartbeat
                                    < timeout_threshold,
                                )
                            )
                        )

                        for validator in result.scalars():
                            logger.warning(
                                f"Marking validator as disconnected: {validator.validator_hotkey}"
                            )
                            validator.is_connected = False

                            # Close WebSocket if exists
                            for conn_id, hotkey in list(
                                self.validator_connections.items()
                            ):
                                if hotkey == validator.validator_hotkey:
                                    if conn_id in self.active_connections:
                                        await self.active_connections[conn_id].close()
                                        del self.active_connections[conn_id]
                                    del self.validator_connections[conn_id]
                                    break

                        await session.commit()

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(HEARTBEAT_INTERVAL)

    async def _score_evaluations(self) -> dict[SS58Address, float]:
        """
        Score completed evaluations with winner-takes-all per competition.

        Scoring logic:
        - Miners must meet minimum success rate threshold per competition to be considered
        - Miners must pass minimum avg reward threshold per competition
        - New miners must exceed current leader by win margin percentage to become leader
        - Current leader retains position if no challenger exceeds margin
        - Final scores are normalized based on competition points

        Returns:
            dict[SS58Address, float]: Mapping of miner hotkeys to their normalized scores (0-1).
        """
        # TODO: consider eval results from multiple (minimum 2) validators before applying scores?
        async with self.async_session() as session:
            # Fetch all active competitions
            competitions_result = await session.execute(
                select(Competition).where(Competition.active)
            )
            competitions = competitions_result.scalars().all()

            if not competitions:
                logger.info("No active competitions found for scoring")
                return {}

            # Calculate total points across all competitions
            total_points = sum(comp.points for comp in competitions)
            if total_points == 0:
                logger.warning("Total competition points is 0")
                return {}

            # Dictionary to store winner scores
            miner_scores: dict[SS58Address, float] = {}

            for competition in competitions:
                # Get all evaluation results for this competition
                results_query = select(BackendEvaluationResult).where(
                    BackendEvaluationResult.competition_id == competition.id
                )
                results = await session.execute(results_query)
                eval_results = results.scalars().all()

                if not eval_results:
                    logger.debug(
                        f"No evaluation results for competition {competition.id}"
                    )
                    continue

                # Find eligible challengers
                eligible_miners: List[tuple[str, float]] = []

                for result in eval_results:
                    if result.success_rate is None or result.avg_reward is None:
                        continue

                    # Check eligibility criteria
                    if result.success_rate < competition.min_success_rate:
                        logger.debug(
                            f"Miner {result.miner_hotkey} excluded from competition {competition.id}: "
                            f"success_rate={result.success_rate:.3f} < min_threshold={competition.min_success_rate:.3f}"
                        )
                        continue

                    if result.avg_reward < competition.min_avg_reward:
                        logger.debug(
                            f"Miner {result.miner_hotkey} excluded from competition {competition.id}: "
                            f"avg_reward={result.avg_reward:.3f} < min_threshold={competition.min_avg_reward}"
                        )
                        continue

                    eligible_miners.append((result.miner_hotkey, result.avg_reward))

                winner_hotkey = None

                if not eligible_miners:
                    # If no eligible miners and there's a current leader, they retain position
                    if competition.current_leader_hotkey:
                        logger.info(
                            f"Competition {competition.id}: Current leader {competition.current_leader_hotkey} "
                            f"retains position (no eligible challengers)"
                        )
                        winner_hotkey = competition.current_leader_hotkey
                    else:
                        logger.info(
                            f"Competition {competition.id}: No eligible miners found"
                        )
                        continue
                else:
                    # Sort by reward to find best performer
                    eligible_miners.sort(key=lambda x: x[1], reverse=True)
                    best_miner, best_reward = eligible_miners[0]

                    # Determine if we have a new leader or retain current
                    if competition.current_leader_hotkey:
                        # Check if best miner is already the leader
                        if best_miner == competition.current_leader_hotkey:
                            winner_hotkey = competition.current_leader_hotkey
                            logger.info(
                                f"Competition {competition.id}: Current leader {winner_hotkey} "
                                f"retains position with avg_reward={best_reward:.3f}"
                            )
                        else:
                            # Challenger must exceed current leader by win margin
                            required_reward = (
                                competition.current_leader_reward or 0
                            ) * (1 + competition.win_margin_pct)

                            if best_reward > required_reward:
                                # New leader!
                                winner_hotkey = best_miner
                                winner_reward = best_reward

                                logger.info(
                                    f"Competition {competition.id}: New leader {winner_hotkey} "
                                    f"(avg_reward={best_reward:.3f}) defeats previous leader "
                                    f"{competition.current_leader_hotkey} (required={required_reward:.3f})"
                                )

                                # Update competition with new leader
                                competition.current_leader_hotkey = winner_hotkey
                                competition.current_leader_reward = winner_reward
                                competition.leader_updated_at = datetime.now(
                                    timezone.utc
                                )
                            else:
                                # Current leader retains position
                                winner_hotkey = competition.current_leader_hotkey
                                logger.info(
                                    f"Competition {competition.id}: Current leader {winner_hotkey} "
                                    f"retains position. Challenger {best_miner} (avg_reward={best_reward:.3f}) "
                                    f"didn't exceed required margin {required_reward:.3f}"
                                )
                    else:
                        # No current leader, best miner becomes first leader
                        winner_hotkey = best_miner
                        winner_reward = best_reward

                        logger.info(
                            f"Competition {competition.id}: First leader {winner_hotkey} "
                            f"established with avg_reward={best_reward:.3f}"
                        )

                        # Update competition with first leader
                        competition.current_leader_hotkey = winner_hotkey
                        competition.current_leader_reward = winner_reward
                        competition.leader_updated_at = datetime.now(timezone.utc)

                # Award points to current leader (normalized)
                if winner_hotkey:
                    normalized_score = competition.points / total_points
                    miner_scores[winner_hotkey] = (
                        miner_scores.get(winner_hotkey, 0.0) + normalized_score
                    )
                    logger.info(
                        f"Competition {competition.id}: Awarded {normalized_score:.4f} normalized score to {winner_hotkey}"
                    )

            # Commit any leader updates to database
            await session.commit()

            # Log final scores
            if miner_scores:
                logger.info(f"Final miner scores: {len(miner_scores)} miners scored")
                for hotkey, score in sorted(
                    miner_scores.items(), key=lambda x: x[1], reverse=True
                )[:10]:
                    logger.info(f"  {hotkey}: {score:.4f}")
            else:
                logger.info("No miners received scores")

            return miner_scores

    async def _periodic_score_evaluation(self) -> None:
        """Periodically evaluate and update miner scores."""
        logger.info("Starting periodic score evaluation task")

        while self._running:
            try:
                logger.info("Running score evaluation cycle")
                miner_scores = await self._score_evaluations()

                # Store latest scores for weight broadcasting
                self._latest_miner_scores = miner_scores

                logger.info(
                    f"Score evaluation complete. {len(miner_scores)} miners scored."
                )

            except Exception as e:
                logger.error(f"Error in periodic score evaluation: {e}")

            # Wait for next evaluation cycle
            await asyncio.sleep(self.score_evaluation_interval)

    async def _periodic_weight_broadcast(self) -> None:
        """Periodically broadcast weights to validators and set on chain."""
        logger.info("Starting periodic weight broadcast task")

        while self._running:
            try:
                logger.info("Running weight broadcast cycle")
                await self._broadcast_and_set_weights()

            except Exception as e:
                logger.error(f"Error in periodic weight broadcast: {e}")

            # Wait for next broadcast cycle
            await asyncio.sleep(self.weight_broadcast_interval)

    async def _broadcast_and_set_weights(self) -> None:
        """Broadcast weights to connected validators and set on chain using latest scores."""
        try:
            if not self.substrate:
                logger.error("Substrate not initialized")
                return
            if not self.metagraph:
                logger.error("Metagraph not initialized")
                return

            # Use cached scores from periodic evaluation
            miner_scores = self._latest_miner_scores.copy()
            # get node uids from metagraph based on their hotkeys
            nodes = self.metagraph.nodes if self.metagraph else {}
            miner_hotkeys = miner_scores.keys()

            miner_uids: List[int] = []
            weights: List[float] = []

            for hotkey in miner_hotkeys:
                node = nodes.get(hotkey)
                if node:
                    miner_uids.append(node.node_id)
                    weights.append(miner_scores[hotkey])

            if not miner_scores:
                logger.info("No miner scores to broadcast")
                return

            # Broadcast to validators
            weight_msg = SetWeightsMessage(weights=weights, miner_uids=miner_uids)
            weights_msg_str = weight_msg.model_dump_json()
            broadcast_count = 0
            failed_connections = []

            for conn_id, ws in list(self.active_connections.items()):
                try:
                    await ws.send_text(weights_msg_str)
                    broadcast_count += 1
                except Exception as e:
                    logger.error(f"Failed to send to {conn_id}: {e}")
                    failed_connections.append(conn_id)

            # Clean up failed connections
            for conn_id in failed_connections:
                if conn_id in self.active_connections:
                    del self.active_connections[conn_id]
                if conn_id in self.validator_connections:
                    del self.validator_connections[conn_id]

            logger.info(
                f"Broadcasted weight update to {broadcast_count} validators:\n{weights_msg_str}"
            )
        except Exception as e:
            logger.error(f"Failed to broadcast weights: {e}")

    async def _get_latest_block(self) -> int:
        """Get latest block from chain.

        Returns:
            int: The latest block number, or -1 if an error occurred.
                 Block 0 is a valid genesis block, so -1 indicates failure.
        """
        try:
            if not self.substrate:
                return -1
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_pool, self.substrate.get_block_number
            )
        except Exception as e:
            logger.error(f"Failed to get latest block: {e}")
            return -1

    async def _sync_metagraph(self) -> None:
        """Sync metagraph nodes."""
        try:
            if self.metagraph:
                # Run in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.thread_pool, self.metagraph.sync_nodes)
                logger.debug("Metagraph synced")
        except Exception as e:
            logger.error(f"Failed to sync metagraph: {e}")

    def _query_commitments_sync(
        self, block_num: int, nodes: list
    ) -> List[ChainCommitmentResponse]:
        """Synchronous version of commitment querying for thread pool."""
        commitments = []

        for node in nodes:
            try:
                miner_commitments = query_commitments_from_substrate(
                    self.config, node.hotkey, block=block_num
                )
                if miner_commitments:
                    commitments.extend(miner_commitments)
            except Exception as e:
                logger.debug(f"Failed to query {node.hotkey}: {e}")
                continue

        return commitments

    async def _query_block_commitments(
        self, block_num: int
    ) -> List[ChainCommitmentResponse]:
        """Query commitments for a block."""
        try:
            if not self.metagraph:
                return []

            # Get nodes as a list for thread pool execution
            nodes = list(self.metagraph.nodes.values())

            # Run commitment querying in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            commitments = await loop.run_in_executor(
                self.thread_pool, self._query_commitments_sync, block_num, nodes
            )

            return commitments

        except Exception as e:
            logger.error(f"Failed to query block {block_num}: {e}")
            return []

    async def _process_commitment(
        self,
        commitment: ChainCommitmentResponse,
        block_num: int,
        active_competitions: dict[str, Competition],
    ):
        """Process a commitment from the chain.

        This function:
        1. Validates the commitment belongs to an active competition
        2. Checks for duplicate submissions from the same miner/competition/version
        3. Creates a new MinerSubmission record in the database
        4. Creates BackendEvaluationJob(s) for each benchmark in the competition
        5. Broadcasts the job(s) to connected validators for evaluation
        """
        try:
            logger.debug(f"Processing commitment for block {block_num}: {commitment}")
            competition_id = getattr(commitment.data, "comp_id", None)

            if not competition_id or competition_id not in active_competitions:
                logger.warning(
                    f"Miner {commitment.hotkey}'s commitment at block {block_num} provided unknown competition {competition_id}"
                )
                return

            competition = active_competitions[competition_id]

            if not self.async_session:
                logger.error("Database not initialized")
                return
            async with self.async_session() as session:
                # Check if submission exists
                existing = await session.execute(
                    select(MinerSubmission).where(
                        and_(
                            MinerSubmission.miner_hotkey == commitment.hotkey,
                            MinerSubmission.competition_id == competition_id,
                            MinerSubmission.version == commitment.data.v,
                        )
                    )
                )

                if existing.scalar_one_or_none():
                    logger.debug(f"Submission already exists for {commitment.hotkey}")
                    return

                # Create submission
                submission = MinerSubmission(
                    id=next(self.id_generator),
                    miner_hotkey=commitment.hotkey,
                    competition_id=competition_id,
                    hf_repo_id=commitment.data.repo_id,
                    version=commitment.data.v,
                    commitment_block=block_num,
                )

                session.add(submission)
                await session.flush()

                # Create job
                job_id = next(self.id_generator)
                for benchmark in competition.benchmarks:
                    # check if the benchmark has a provider or a benchmark name
                    if "provider" not in benchmark or "benchmark_name" not in benchmark:
                        logger.error(
                            f"Benchmark missing provider or benchmark_name: {benchmark}"
                        )
                        continue

                    eval_job = BackendEvaluationJob(
                        id=job_id,
                        submission_id=submission.id,
                        competition_id=competition_id,
                        miner_hotkey=submission.miner_hotkey,
                        hf_repo_id=submission.hf_repo_id,
                        env_provider=benchmark["provider"],
                        benchmark_name=benchmark["benchmark_name"],
                        config=benchmark.get("config", {}),  # Optional config
                    )

                session.add(eval_job)
                await session.commit()

                logger.debug(f"Created evaluation job: {eval_job}")

                # Broadcast to validators
                await self._broadcast_job(eval_job)
                logger.debug(f"Broadcasted job {job_id} to validators")

                logger.info(f"Processed commitment from {commitment.hotkey}")

        except Exception as e:
            logger.error(f"Failed to process commitment: {e}")

    async def _update_job_status(
        self,
        job_id: int,
        validator_hotkey: str,
        status: EvaluationStatus,
        detail: str = None,
    ):
        """Update job status for a specific validator."""
        if not self.async_session:
            logger.error("Database not initialized")
            return

        try:
            async with self.async_session() as session:
                # Create new status record
                status_record = BackendEvaluationJobStatus(
                    id=next(self.id_generator),
                    job_id=job_id,
                    validator_hotkey=validator_hotkey,
                    status=status,
                    detail=detail,
                )
                session.add(status_record)
                await session.commit()

                logger.debug(
                    f"Updated job {job_id} status to {status.value} for validator {validator_hotkey}"
                )

        except Exception as e:
            logger.error(f"Failed to update job status: {e}")

    async def _update_job_status_for_validators(
        self, job_id: int, status: EvaluationStatus, detail: str = None
    ):
        """Update job status for all connected validators."""
        for validator_hotkey in self.validator_connections.values():
            await self._update_job_status(job_id, validator_hotkey, status, detail)

    async def _broadcast_job(self, job: BackendEvaluationJob):
        """Broadcast job to connected validators."""
        if not self.active_connections:
            logger.warning("No validators connected")
            return

        env_config = job.config if job.config else {}

        job_msg = EvalJobMessage(
            job_id=job.id,
            competition_id=job.competition_id,
            submission_id=job.submission_id,
            miner_hotkey=job.miner_hotkey,
            hf_repo_id=job.hf_repo_id,
            env_provider=job.env_provider,
            benchmark_name=job.benchmark_name,
            config=env_config,
        )

        message = job_msg.model_dump_json()
        broadcast_count = 0
        failed_connections = []

        for conn_id, ws in list(self.active_connections.items()):
            try:
                await ws.send_text(message)
                broadcast_count += 1
            except Exception as e:
                logger.error(f"Failed to send to {conn_id}: {e}")
                failed_connections.append(conn_id)

        # Clean up failed connections
        for conn_id in failed_connections:
            if conn_id in self.active_connections:
                del self.active_connections[conn_id]
            if conn_id in self.validator_connections:
                del self.validator_connections[conn_id]

        # Update job status to QUEUED for all validators that received the job
        if broadcast_count > 0 and self.async_session:
            await self._update_job_status_for_validators(
                job.id, EvaluationStatus.QUEUED, "Job queued to validators"
            )

        logger.info(f"Broadcasted job {job.id} to {broadcast_count} validators")
