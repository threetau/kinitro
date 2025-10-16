"""

This provides REST API endpoints and WebSocket connections for:

- Competition management
- Validator connections
- Job distribution
- Result collection
"""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import dotenv
from fastapi import (
    WebSocket,
)
from fiber.chain.fetch_nodes import _get_nodes_for_uid
from fiber.chain.interface import get_substrate
from fiber.chain.models import Node
from snowflake import SnowflakeGenerator
from sqlalchemy import and_, func, select
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
    DEFAULT_SUBMISSION_HOLDOUT_SECONDS,
    EVAL_JOB_TIMEOUT,
    HEARTBEAT_INTERVAL,
    HOLDOUT_RELEASE_SCAN_INTERVAL,
    MAX_WORKERS,
    SCORE_EVALUATION_INTERVAL,
    SCORE_EVALUATION_STARTUP_DELAY,
    SUBMISSION_DOWNLOAD_URL_TTL_SECONDS,
    SUBMISSION_RELEASE_URL_TTL_SECONDS,
    SUBMISSION_UPLOAD_URL_TTL_SECONDS,
    WEIGHT_BROADCAST_INTERVAL,
    WEIGHT_BROADCAST_STARTUP_DELAY,
)
from backend.events import (
    JobCompletedEvent,
    JobCreatedEvent,
    JobStatusChangedEvent,
    StatsUpdatedEvent,
    SubmissionReceivedEvent,
    ValidatorDisconnectedEvent,
)
from backend.realtime import event_broadcaster
from backend.submission_storage import PresignedUpload, SubmissionStorage

try:  # pragma: no cover - optional dependency
    from fiber import Keypair as FiberKeypair  # type: ignore
except ImportError:  # pragma: no cover
    FiberKeypair = None

try:  # pragma: no cover - fallback dependency
    from substrateinterface import Keypair as SubstrateKeypair  # type: ignore
except ImportError:  # pragma: no cover
    SubstrateKeypair = None
from core.chain import query_commitments_from_substrate
from core.db.models import EvaluationStatus
from core.log import get_logger
from core.messages import (
    EvalJobMessage,
    EventType,
    SetWeightsMessage,
)
from core.schemas import ChainCommitmentResponse, ModelProvider
from core.storage import R2Config

from .config import BackendConfig
from .models import (
    BackendEvaluationJob,
    BackendEvaluationJobStatus,
    BackendEvaluationResult,
    BackendState,
    Competition,
    MinerSubmission,
    SS58Address,
    SubmissionUpload,
    SubmissionUploadStatus,
    ValidatorConnection,
)

dotenv.load_dotenv()

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

        # Submission hold-out configuration
        self.submission_holdout_seconds = int(
            config.settings.get(
                "submission_holdout_seconds", DEFAULT_SUBMISSION_HOLDOUT_SECONDS
            )
        )
        self.submission_upload_url_ttl = int(
            config.settings.get(
                "submission_upload_url_ttl_seconds",
                SUBMISSION_UPLOAD_URL_TTL_SECONDS,
            )
        )
        self.submission_download_url_ttl = int(
            config.settings.get(
                "submission_download_url_ttl_seconds",
                SUBMISSION_DOWNLOAD_URL_TTL_SECONDS,
            )
        )
        self.holdout_release_scan_interval = int(
            config.settings.get(
                "holdout_release_scan_interval", HOLDOUT_RELEASE_SCAN_INTERVAL
            )
        )

        self.r2_config = self._load_r2_config()
        logger.debug(f"R2 Config: {self.r2_config}")
        self.submission_storage: Optional[SubmissionStorage] = (
            SubmissionStorage(self.r2_config) if self.r2_config else None
        )

        # Chain connection objects
        # Using Any since fiber's SubstrateInterface is from async_substrate_interface
        self.substrate: Optional[Any] = (
            None  # async_substrate_interface.sync_substrate.SubstrateInterface
        )
        self.nodes: Optional[Dict[SS58Address, Node]] = None

        # WebSocket connections
        self.active_connections: Dict[ConnectionId, WebSocket] = {}
        self.validator_connections: Dict[ConnectionId, SS58Address] = {}

        # Background tasks
        self._running = False
        self._chain_monitor_task = None
        self._heartbeat_monitor_task = None
        self._stale_job_monitor_task = None
        self._score_evaluation_task = None
        self._weight_broadcast_task = None
        self._holdout_release_task = None

        # Store latest scores for weight broadcasting
        self._latest_miner_scores: Dict[SS58Address, float] = {}

        # ID generator
        self.id_generator = SnowflakeGenerator(42)

        # Database
        self.engine: AsyncEngine = None
        self.async_session: async_sessionmaker[AsyncSession] = None

        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

    # TODO: rename this to _load_s3_config
    def _load_r2_config(self) -> Optional[R2Config]:
        """Load R2 configuration for submission vault from environment variables."""
        endpoint_url = os.environ.get("S3_ENDPOINT_URL")
        access_key_id = os.environ.get("S3_ACCESS_KEY_ID")
        secret_access_key = os.environ.get("S3_SECRET_ACCESS_KEY")
        bucket_name = os.environ.get("S3_BUCKET_NAME")
        region = os.environ.get("S3_REGION", "auto")
        public_url_base = os.environ.get("S3_PUBLIC_URL_BASE")

        if not all([endpoint_url, access_key_id, secret_access_key, bucket_name]):
            logger.warning(
                "S3 credentials missing; direct submission uploads are disabled"
            )
            return None

        return R2Config(
            endpoint_url=endpoint_url,
            access_key_id=access_key_id,
            secret_access_key=secret_access_key,
            bucket_name=bucket_name,
            region=region,
            public_url_base=public_url_base,
        )

    @staticmethod
    def verify_hotkey_signature(
        hotkey: str, message: bytes, signature_hex: str
    ) -> bool:
        """Verify a hotkey-signed payload using available sr25519 implementations."""

        signature_body = (
            signature_hex[2:] if signature_hex.startswith("0x") else signature_hex
        )
        try:
            signature = bytes.fromhex(signature_body)
        except ValueError:
            return False

        keypair_classes = [
            cls for cls in (FiberKeypair, SubstrateKeypair) if cls is not None
        ]
        if not keypair_classes:
            raise RuntimeError(
                "No signature verification backend available (fiber/substrateinterface not installed)"
            )

        for keypair_cls in keypair_classes:
            try:
                if hasattr(keypair_cls, "create_from_ss58_address"):
                    keypair = keypair_cls.create_from_ss58_address(hotkey)
                else:
                    keypair = keypair_cls(ss58_address=hotkey)
                if keypair.verify(message, signature):
                    return True
            except Exception as exc:  # pragma: no cover - verification failure
                logger.debug(
                    "Signature verification attempt failed with %s: %s",
                    getattr(keypair_cls, "__name__", str(keypair_cls)),
                    exc,
                )
                continue

        return False

    async def create_submission_upload(
        self,
        *,
        miner_hotkey: SS58Address,
        competition_id: str,
        version: str,
        artifact_sha256: str,
        artifact_size_bytes: int,
        holdout_seconds: Optional[int] = None,
    ) -> tuple[SubmissionUpload, PresignedUpload]:
        """Create a submission upload record and mint an upload URL."""

        if not self.submission_storage:
            raise RuntimeError("Submission storage is not configured")
        if not self.async_session:
            raise RuntimeError("Database not initialized")

        if artifact_size_bytes <= 0:
            raise RuntimeError("artifact_size_bytes must be positive")

        competition_id = competition_id.strip()
        if not competition_id:
            raise RuntimeError("competition_id cannot be empty")

        version = version.strip()
        if not version:
            raise RuntimeError("version cannot be empty")

        normalized_sha = artifact_sha256.lower()

        submission_id = next(self.id_generator)
        object_key = self.submission_storage.build_object_key(
            submission_id, f"{version}.tar.gz"
        )

        presigned = self.submission_storage.create_presigned_upload(
            object_key,
            expires_in=self.submission_upload_url_ttl,
        )

        async with self.async_session() as session:
            competition = await session.get(Competition, competition_id)
            if not competition:
                raise RuntimeError(f"Competition {competition_id} does not exist")

            if not competition.active:
                raise RuntimeError(f"Competition {competition_id} is not active")

            if holdout_seconds is not None:
                holdout = max(0, int(holdout_seconds))
            else:
                holdout = max(0, competition.submission_holdout_seconds)

            existing_submission = await session.execute(
                select(MinerSubmission).where(
                    MinerSubmission.miner_hotkey == miner_hotkey,
                    MinerSubmission.competition_id == competition_id,
                    MinerSubmission.version == version,
                )
            )
            if existing_submission.scalar_one_or_none():
                raise RuntimeError(
                    "Submission with this version already exists for this competition"
                )

            existing_upload = await session.execute(
                select(SubmissionUpload).where(
                    SubmissionUpload.miner_hotkey == miner_hotkey,
                    SubmissionUpload.competition_id == competition_id,
                    SubmissionUpload.version == version,
                    SubmissionUpload.status != SubmissionUploadStatus.PROCESSED,
                )
            )
            if existing_upload.scalar_one_or_none():
                raise RuntimeError(
                    "A pending upload already exists for this version and competition"
                )

            upload_record = SubmissionUpload(
                submission_id=submission_id,
                miner_hotkey=miner_hotkey,
                competition_id=competition_id,
                version=version,
                artifact_object_key=object_key,
                artifact_sha256=normalized_sha,
                artifact_size_bytes=artifact_size_bytes,
                upload_url_expires_at=presigned.expires_at,
                holdout_seconds=holdout,
            )

            session.add(upload_record)
            await session.commit()
            await session.refresh(upload_record)

        return upload_record, presigned

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
        self._stale_job_monitor_task = asyncio.create_task(self._monitor_stale_jobs())

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

        if self.submission_storage:
            self._holdout_release_task = asyncio.create_task(
                self._holdout_release_loop()
            )
            logger.info(
                "Hold-out release task started (interval=%ss)",
                self.holdout_release_scan_interval,
            )
        else:
            logger.info(
                "Hold-out release task not started (submission storage unavailable)"
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
            (self._chain_monitor_task, "chain_monitor"),
            (self._heartbeat_monitor_task, "heartbeat_monitor"),
            (self._stale_job_monitor_task, "stale_job_monitor"),
            (self._score_evaluation_task, "score_evaluation"),
            (self._weight_broadcast_task, "weight_broadcast"),
            (self._holdout_release_task, "holdout_release"),
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

            node_list = _get_nodes_for_uid(
                self.substrate, self.config.settings["subtensor"]["netuid"]
            )

            self.nodes = {node.hotkey: node for node in node_list}

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
                if self.substrate and self.nodes and self.async_session:
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

    async def _monitor_stale_jobs(self) -> None:
        """Monitor for stale jobs and mark them as failed."""
        while self._running:
            try:
                if self.async_session:
                    async with self.async_session() as session:
                        # Define timeout threshold (jobs older than EVAL_JOB_TIMEOUT seconds without completion)
                        current_time = datetime.now(timezone.utc)
                        stale_threshold = current_time - timedelta(
                            seconds=EVAL_JOB_TIMEOUT
                        )
                        # Remove timezone info to match database column type
                        stale_threshold = stale_threshold.replace(tzinfo=None)

                        # Find jobs that have been running for too long
                        result = await session.execute(
                            select(BackendEvaluationJob).where(
                                BackendEvaluationJob.created_at < stale_threshold
                            )
                        )

                        stale_jobs = result.scalars().all()

                        for job in stale_jobs:
                            # Check if this job has any recent status updates
                            status_result = await session.execute(
                                select(BackendEvaluationJobStatus)
                                .where(BackendEvaluationJobStatus.job_id == job.id)
                                .order_by(BackendEvaluationJobStatus.created_at.desc())
                                .limit(1)
                            )
                            latest_status = status_result.scalar()

                            # If no status or last status is not terminal, mark as failed
                            if not latest_status or latest_status.status not in [
                                EvaluationStatus.COMPLETED,
                                EvaluationStatus.FAILED,
                                EvaluationStatus.CANCELLED,
                                EvaluationStatus.TIMEOUT,
                            ]:
                                logger.warning(f"Marking stale job {job.id} as TIMEOUT")

                                # Create timeout status for all connected validators
                                for (
                                    validator_hotkey
                                ) in self.validator_connections.values():
                                    timeout_status = BackendEvaluationJobStatus(
                                        id=next(self.id_generator),
                                        job_id=job.id,
                                        validator_hotkey=validator_hotkey,
                                        status=EvaluationStatus.TIMEOUT,
                                        detail="Job marked as timeout due to inactivity",
                                    )
                                    session.add(timeout_status)

                                await session.commit()

                                # Broadcast timeout event
                                await event_broadcaster.broadcast_event(
                                    EventType.JOB_STATUS_CHANGED,
                                    {
                                        "job_id": str(job.id),
                                        "status": "TIMEOUT",
                                        "detail": "Job marked as timeout due to inactivity",
                                    },
                                )

                await asyncio.sleep(300)  # Check every 5 minutes

            except Exception as e:
                logger.error(f"Error monitoring stale jobs: {e}")
                await asyncio.sleep(300)

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

                        stale_validators = result.scalars().all()

                        for validator in stale_validators:
                            logger.warning(
                                f"Marking validator as disconnected: {validator.validator_hotkey}"
                            )
                            validator.is_connected = False

                            # Close WebSocket if exists
                            conn_id_to_remove = None
                            for conn_id, hotkey in list(
                                self.validator_connections.items()
                            ):
                                if hotkey == validator.validator_hotkey:
                                    if conn_id in self.active_connections:
                                        await self.active_connections[conn_id].close()
                                        del self.active_connections[conn_id]
                                    del self.validator_connections[conn_id]
                                    conn_id_to_remove = conn_id
                                    break

                            # Broadcast validator disconnected event
                            if conn_id_to_remove:
                                disconnected_event = ValidatorDisconnectedEvent(
                                    validator_hotkey=validator.validator_hotkey,
                                    connection_id=conn_id_to_remove,
                                    disconnected_at=datetime.now(timezone.utc),
                                    reason="Heartbeat timeout",
                                )
                                await event_broadcaster.broadcast_event(
                                    EventType.VALIDATOR_DISCONNECTED, disconnected_event
                                )

                        await session.commit()

                        # Broadcast updated stats if any validators were disconnected
                        if stale_validators:
                            await self._broadcast_stats_update()

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(HEARTBEAT_INTERVAL)

    def _is_miner_eligible(
        self,
        result: BackendEvaluationResult,
        competition: Competition,
    ) -> bool:
        """Check if a miner meets eligibility criteria for a competition."""
        if result.success_rate is None or result.avg_reward is None:
            return False

        if result.success_rate < competition.min_success_rate:
            logger.debug(
                f"Miner {result.miner_hotkey} excluded from competition {competition.id}: "
                f"success_rate={result.success_rate:.3f} < min_threshold={competition.min_success_rate:.3f}"
            )
            return False

        if result.avg_reward < competition.min_avg_reward:
            logger.debug(
                f"Miner {result.miner_hotkey} excluded from competition {competition.id}: "
                f"avg_reward={result.avg_reward:.3f} < min_threshold={competition.min_avg_reward}"
            )
            return False

        return True

    def _determine_competition_winner(
        self,
        competition: Competition,
        eligible_miners: List[tuple[str, float]],
    ) -> tuple[Optional[str], Optional[float]]:
        """Determine the winner of a competition based on eligible miners."""
        if not eligible_miners:
            if competition.current_leader_hotkey:
                logger.info(
                    f"Competition {competition.id}: Current leader {competition.current_leader_hotkey} "
                    f"retains position (no eligible challengers)"
                )
                return (
                    competition.current_leader_hotkey,
                    competition.current_leader_reward,
                )
            logger.info(f"Competition {competition.id}: No eligible miners found")
            return None, None

        # Sort by reward to find best performer
        eligible_miners.sort(key=lambda x: x[1], reverse=True)
        best_miner, best_reward = eligible_miners[0]

        # No current leader - best miner becomes first leader
        if not competition.current_leader_hotkey:
            logger.info(
                f"Competition {competition.id}: First leader {best_miner} "
                f"established with avg_reward={best_reward:.3f}"
            )
            return best_miner, best_reward

        # Check if best miner is already the leader
        if best_miner == competition.current_leader_hotkey:
            logger.info(
                f"Competition {competition.id}: Current leader {best_miner} "
                f"retains position with avg_reward={best_reward:.3f}"
            )
            return competition.current_leader_hotkey, competition.current_leader_reward

        # Challenger must exceed current leader by win margin
        required_reward = (competition.current_leader_reward or 0) * (
            1 + competition.win_margin_pct
        )

        if best_reward > required_reward:
            logger.info(
                f"Competition {competition.id}: New leader {best_miner} "
                f"(avg_reward={best_reward:.3f}) defeats previous leader "
                f"{competition.current_leader_hotkey} (required={required_reward:.3f})"
            )
            return best_miner, best_reward

        # Current leader retains position
        logger.info(
            f"Competition {competition.id}: Current leader {competition.current_leader_hotkey} "
            f"retains position. Challenger {best_miner} (avg_reward={best_reward:.3f}) "
            f"didn't exceed required margin {required_reward:.3f}"
        )
        return competition.current_leader_hotkey, competition.current_leader_reward

    async def _score_evaluations(self) -> dict[SS58Address, float]:
        """
        Score completed evaluations with winner-takes-all per competition.

        Scoring logic:
        - Miners must meet minimum success rate threshold per competition to be considered
        - Miners must pass minimum avg reward threshold per competition
        - New miners must exceed current leader by win margin percentage to become leader
        - Current leader retains position if no challenger exceeds margin
        - Each miner can only win ONE competition (first-win policy if appearing in multiple)
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
                    if self._is_miner_eligible(result, competition):
                        eligible_miners.append((result.miner_hotkey, result.avg_reward))

                # Determine competition winner
                winner_hotkey, winner_reward = self._determine_competition_winner(
                    competition, eligible_miners
                )

                if not winner_hotkey:
                    continue

                # Update competition leader if changed
                if (
                    winner_hotkey != competition.current_leader_hotkey
                    or winner_reward != competition.current_leader_reward
                ):
                    competition.current_leader_hotkey = winner_hotkey
                    competition.current_leader_reward = winner_reward
                    competition.leader_updated_at = datetime.now(timezone.utc)

                # Award points to winner (check for duplicate wins first)
                normalized_score = competition.points / total_points

                if winner_hotkey in miner_scores:
                    logger.warning(
                        f"Miner {winner_hotkey} already won competition - skipping score from {competition.id}. "
                        f"Previous score: {miner_scores[winner_hotkey]:.4f}, would have added: {normalized_score:.4f}"
                    )
                    continue

                miner_scores[winner_hotkey] = normalized_score
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

    async def _holdout_release_loop(self) -> None:
        """Periodically release submission artifacts after the hold-out window."""
        if not self.submission_storage:
            return

        logger.info("Starting hold-out release task")
        while self._running:
            try:
                await self._release_due_submissions()
            except Exception as e:
                logger.error(f"Hold-out release task error: {e}")
            await asyncio.sleep(self.holdout_release_scan_interval)

    async def _release_due_submissions(self) -> None:
        """Mark submissions as released when their hold-out period expires."""
        if not self.submission_storage or not self.async_session:
            logger.warning("Submission storage or async session not available")
            return

        now = datetime.now(timezone.utc)
        async with self.async_session() as session:
            stmt = select(MinerSubmission).where(
                and_(
                    MinerSubmission.holdout_release_at.is_not(None),
                    MinerSubmission.holdout_release_at <= now,
                    MinerSubmission.released_at.is_(None),
                    MinerSubmission.artifact_object_key.is_not(None),
                )
            )
            result = await session.execute(stmt)
            submissions = result.scalars().all()

            if not submissions:
                logger.info("No submissions due for hold-out release")
                return

            for submission in submissions:
                try:
                    release_url, expires_at = (
                        self.submission_storage.generate_download_url(
                            submission.artifact_object_key,
                            SUBMISSION_RELEASE_URL_TTL_SECONDS,
                        )
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to generate release URL for submission %s: %s",
                        submission.id,
                        exc,
                    )
                    continue

                submission.released_at = now
                submission.public_artifact_url = release_url
                submission.public_artifact_url_expires_at = expires_at
                logger.info(
                    "Hold-out released submission %s (artifact=%s)",
                    submission.id,
                    submission.artifact_object_key,
                )

            await session.commit()

    async def _broadcast_and_set_weights(self) -> None:
        """Broadcast weights to connected validators and set on chain using latest scores."""
        try:
            if not self.substrate:
                logger.error("Substrate not initialized")
                return
            if not self.nodes:
                logger.error("Node list not initialized")
                return

            # Use cached scores from periodic evaluation
            miner_scores = self._latest_miner_scores.copy()

            # Build weights dict mapping UIDs to weights
            weights_dict: dict[int, float] = {}
            for hotkey, weight in miner_scores.items():
                node = self.nodes.get(hotkey)
                if node:
                    weights_dict[node.node_id] = weight

            if not weights_dict:
                logger.info("No miner scores to broadcast")
                return

            # Populate missing entries with 0.0 weight for all nodes
            for node in self.nodes.values():
                weights_dict.setdefault(node.node_id, 0.0)

            # Broadcast to validators
            weight_msg = SetWeightsMessage(weights=weights_dict)
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

    def _sync_nodes(self) -> None:
        node_list = _get_nodes_for_uid(
            self.substrate, self.config.settings["subtensor"]["netuid"]
        )
        self.nodes = {node.hotkey: node for node in node_list}

    async def _sync_metagraph(self) -> None:
        """Sync metagraph nodes with memory leak prevention."""
        try:
            if self.nodes:
                # Run sync in thread pool to avoid blocking
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.thread_pool, self._sync_nodes)

                logger.debug("Nodes synced")
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
                    self.config, self.substrate, node.hotkey, block=block_num
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
            if not self.nodes:
                return []

            node_list = list(self.nodes.values())

            # Run commitment querying in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            commitments = await loop.run_in_executor(
                self.thread_pool, self._query_commitments_sync, block_num, node_list
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
        """Process a commitment from the chain."""

        try:
            logger.debug(
                "Processing commitment for block %s: %s", block_num, commitment
            )
            competition_id = getattr(commitment.data, "comp_id", None)

            if not competition_id or competition_id not in active_competitions:
                logger.warning(
                    "Miner %s submitted commitment for unknown competition %s",
                    commitment.hotkey,
                    competition_id,
                )
                return

            provider = getattr(commitment.data, "provider", None)

            if provider == ModelProvider.R2:
                await self._process_r2_commitment(
                    commitment,
                    block_num,
                    active_competitions[competition_id],
                )
                return

            logger.warning(
                "Unsupported commitment provider %s from miner %s",
                provider,
                commitment.hotkey,
            )
        except Exception as exc:
            logger.error("Failed to process commitment: %s", exc)

    async def _process_r2_commitment(
        self,
        commitment: ChainCommitmentResponse,
        block_num: int,
        competition: Competition,
    ) -> None:
        """Handle commitments referencing direct-vault submissions."""

        if not self.submission_storage:
            logger.error(
                "Submission storage not configured; cannot process R2 commitment"
            )
            return

        if not self.async_session:
            logger.error("Database not initialized; cannot process commitment")
            return

        try:
            submission_id = int(commitment.data.repo_id)
        except (TypeError, ValueError):
            logger.error(
                "Invalid submission identifier '%s' provided by miner %s",
                commitment.data.repo_id,
                commitment.hotkey,
            )
            return

        async with self.async_session() as session:
            upload_result = await session.execute(
                select(SubmissionUpload).where(
                    SubmissionUpload.submission_id == submission_id
                )
            )
            upload: Optional[SubmissionUpload] = upload_result.scalar_one_or_none()

            if not upload:
                logger.error(
                    "No pending upload found for submission %s (miner %s)",
                    submission_id,
                    commitment.hotkey,
                )
                return

            if upload.status == SubmissionUploadStatus.PROCESSED:
                logger.info(
                    "Submission %s already processed; ignoring duplicate commitment",
                    submission_id,
                )
                return

            if upload.miner_hotkey != commitment.hotkey:
                logger.error(
                    "Miner %s attempted to commit submission %s owned by %s",
                    commitment.hotkey,
                    submission_id,
                    upload.miner_hotkey,
                )
                return

            if upload.competition_id != competition.id:
                logger.error(
                    "Submission %s was prepared for competition %s but miner committed to %s",
                    submission_id,
                    upload.competition_id,
                    competition.id,
                )
                return

            existing_submission = await session.execute(
                select(MinerSubmission).where(MinerSubmission.id == submission_id)
            )
            if existing_submission.scalar_one_or_none():
                logger.info("Submission %s already registered; skipping", submission_id)
                upload.status = SubmissionUploadStatus.PROCESSED
                await session.commit()
                return

            metadata = self.submission_storage.head_object(upload.artifact_object_key)
            if not metadata:
                logger.error(
                    "Artifact %s not found for submission %s",
                    upload.artifact_object_key,
                    submission_id,
                )
                return

            if metadata.sha256 and metadata.sha256 != upload.artifact_sha256:
                logger.error(
                    "Artifact checksum mismatch for submission %s", submission_id
                )
                return

            actual_size = metadata.size_bytes or upload.artifact_size_bytes

            if actual_size != upload.artifact_size_bytes:
                logger.error(
                    "Artifact size mismatch for submission %s (expected %s, got %s)",
                    submission_id,
                    upload.artifact_size_bytes,
                    actual_size,
                )
                return

            upload.uploaded_at = metadata.last_modified
            upload.status = SubmissionUploadStatus.PROCESSED

            submission = MinerSubmission(
                id=submission_id,
                miner_hotkey=commitment.hotkey,
                competition_id=competition.id,
                hf_repo_id=f"r2:{submission_id}",
                version=upload.version,
                commitment_block=block_num,
                artifact_object_key=upload.artifact_object_key,
                artifact_sha256=upload.artifact_sha256,
                artifact_size_bytes=actual_size,
                holdout_release_at=datetime.now(timezone.utc)
                + timedelta(seconds=upload.holdout_seconds),
            )

            session.add(submission)
            await session.flush()

            submission_event = SubmissionReceivedEvent(
                submission_id=submission.id,
                competition_id=submission.competition_id,
                miner_hotkey=submission.miner_hotkey,
                hf_repo_id=submission.hf_repo_id,
                block_number=block_num,
                created_at=submission.created_at or datetime.now(timezone.utc),
            )
            await event_broadcaster.broadcast_event(
                EventType.SUBMISSION_RECEIVED, submission_event
            )

            jobs: List[BackendEvaluationJob] = []
            for benchmark in competition.benchmarks:
                if "provider" not in benchmark or "benchmark_name" not in benchmark:
                    logger.error(
                        "Benchmark missing provider or benchmark_name: %s", benchmark
                    )
                    continue

                job = BackendEvaluationJob(
                    id=next(self.id_generator),
                    submission_id=submission.id,
                    competition_id=competition.id,
                    miner_hotkey=submission.miner_hotkey,
                    hf_repo_id=submission.hf_repo_id,
                    env_provider=benchmark["provider"],
                    benchmark_name=benchmark["benchmark_name"],
                    config=benchmark.get("config", {}),
                    artifact_object_key=submission.artifact_object_key,
                    artifact_sha256=submission.artifact_sha256,
                    artifact_size_bytes=submission.artifact_size_bytes,
                )
                jobs.append(job)

            if not jobs:
                logger.error(
                    "No evaluation jobs generated for submission %s", submission_id
                )
                await session.commit()
                return

            session.add_all(jobs)
            await session.commit()

            connected_validator_hotkeys = tuple(
                dict.fromkeys(self.validator_connections.values())
            )

            for job in jobs:
                job_event = JobCreatedEvent(
                    job_id=str(job.id),
                    competition_id=job.competition_id,
                    submission_id=job.submission_id,
                    miner_hotkey=job.miner_hotkey,
                    hf_repo_id=job.hf_repo_id,
                    env_provider=job.env_provider,
                    benchmark_name=job.benchmark_name,
                    config=job.config if job.config else {},
                    status=EvaluationStatus.QUEUED,
                    validator_statuses={
                        hotkey: EvaluationStatus.QUEUED
                        for hotkey in connected_validator_hotkeys
                    },
                )
                await event_broadcaster.broadcast_event(
                    EventType.JOB_CREATED, job_event
                )
                await self._broadcast_job(job)

            await self._broadcast_stats_update()

            logger.info(
                "Processed R2 submission %s from miner %s",
                submission_id,
                commitment.hotkey,
            )

    async def _broadcast_stats_update(self):
        """Broadcast updated statistics to all clients."""
        if not self.async_session:
            return

        try:
            async with self.async_session() as session:
                # Get competitions
                comp_result = await session.execute(select(Competition))
                competitions = comp_result.scalars().all()
                active_comps = [c for c in competitions if c.active]
                total_points = sum(c.points for c in active_comps)

                # Get validators
                val_result = await session.execute(
                    select(ValidatorConnection).where(ValidatorConnection.is_connected)
                )
                connected_validators = len(val_result.scalars().all())

                # Get submissions count
                sub_result = await session.execute(
                    select(func.count(MinerSubmission.id))
                )
                total_submissions = sub_result.scalar() or 0

                # Get jobs count
                job_result = await session.execute(
                    select(func.count(BackendEvaluationJob.id))
                )
                total_jobs = job_result.scalar() or 0

                # Get completed jobs count (latest status is COMPLETED)
                latest_status_subquery = (
                    select(
                        BackendEvaluationJobStatus.job_id,
                        func.max(BackendEvaluationJobStatus.created_at).label(
                            "max_created_at"
                        ),
                    )
                    .group_by(BackendEvaluationJobStatus.job_id)
                    .subquery()
                )

                completed_jobs_result = await session.execute(
                    select(func.count(BackendEvaluationJob.id.distinct()))
                    .select_from(BackendEvaluationJob)
                    .join(
                        BackendEvaluationJobStatus,
                        BackendEvaluationJob.id == BackendEvaluationJobStatus.job_id,
                    )
                    .join(
                        latest_status_subquery,
                        and_(
                            BackendEvaluationJobStatus.job_id
                            == latest_status_subquery.c.job_id,
                            BackendEvaluationJobStatus.created_at
                            == latest_status_subquery.c.max_created_at,
                        ),
                    )
                    .where(
                        BackendEvaluationJobStatus.status == EvaluationStatus.COMPLETED
                    )
                )
                completed_jobs = completed_jobs_result.scalar() or 0

                # Get failed jobs count (latest status is FAILED, CANCELLED, or TIMEOUT)
                failed_jobs_result = await session.execute(
                    select(func.count(BackendEvaluationJob.id.distinct()))
                    .select_from(BackendEvaluationJob)
                    .join(
                        BackendEvaluationJobStatus,
                        BackendEvaluationJob.id == BackendEvaluationJobStatus.job_id,
                    )
                    .join(
                        latest_status_subquery,
                        and_(
                            BackendEvaluationJobStatus.job_id
                            == latest_status_subquery.c.job_id,
                            BackendEvaluationJobStatus.created_at
                            == latest_status_subquery.c.max_created_at,
                        ),
                    )
                    .where(
                        BackendEvaluationJobStatus.status.in_(
                            [
                                EvaluationStatus.FAILED,
                                EvaluationStatus.CANCELLED,
                                EvaluationStatus.TIMEOUT,
                            ]
                        )
                    )
                )
                failed_jobs = failed_jobs_result.scalar() or 0

                # Get results count
                result_count = await session.execute(
                    select(func.count(BackendEvaluationResult.id))
                )
                total_results = result_count.scalar() or 0

                # Get backend state
                state_result = await session.execute(
                    select(BackendState).where(BackendState.id == 1)
                )
                state = state_result.scalar_one_or_none()

                # Calculate competition percentages
                comp_percentages = {}
                for comp in active_comps:
                    percentage = (
                        (comp.points / total_points * 100) if total_points > 0 else 0
                    )
                    comp_percentages[comp.id] = percentage

                # Create and broadcast StatsUpdatedEvent
                stats_event = StatsUpdatedEvent(
                    total_competitions=len(competitions),
                    active_competitions=len(active_comps),
                    total_points=total_points,
                    connected_validators=connected_validators,
                    total_submissions=total_submissions,
                    total_jobs=total_jobs,
                    total_results=total_results,
                    completed_jobs=completed_jobs,
                    failed_jobs=failed_jobs,
                    last_seen_block=state.last_seen_block if state else 0,
                    competition_percentages=comp_percentages,
                )

                await event_broadcaster.broadcast_event(
                    EventType.STATS_UPDATED, stats_event
                )

        except Exception as e:
            logger.error(f"Failed to broadcast stats update: {e}")

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

                # Broadcast status change event to clients using the model
                status_event = JobStatusChangedEvent(
                    job_id=str(job_id),
                    validator_hotkey=validator_hotkey,
                    status=status.value,
                    detail=detail,
                    created_at=status_record.created_at,
                )
                await event_broadcaster.broadcast_event(
                    EventType.JOB_STATUS_CHANGED, status_event
                )

                # If job is completed or failed, also send JOB_COMPLETED event and stats update
                if status in [
                    EvaluationStatus.COMPLETED,
                    EvaluationStatus.FAILED,
                    EvaluationStatus.CANCELLED,
                    EvaluationStatus.TIMEOUT,
                ]:
                    completed_event = JobCompletedEvent(
                        job_id=str(job_id),
                        validator_hotkey=validator_hotkey,
                        status=status.value,
                        detail=detail,
                        result_count=0,  # Will be updated when results come in
                    )
                    await event_broadcaster.broadcast_event(
                        EventType.JOB_COMPLETED, completed_event
                    )

                    # Broadcast updated stats
                    await self._broadcast_stats_update()

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

        artifact_url = None
        artifact_expires_at: Optional[datetime] = None
        if self.submission_storage and job.artifact_object_key:
            try:
                artifact_url, artifact_expires_at = (
                    self.submission_storage.generate_download_url(
                        job.artifact_object_key, self.submission_download_url_ttl
                    )
                )
            except Exception as exc:
                logger.error(
                    "Failed to generate artifact URL for job %s: %s", job.id, exc
                )
        else:
            logger.error(
                "Cannot broadcast submission %s: storage unavailable or artifact missing",
                job.submission_id,
            )
            return

        job_msg = EvalJobMessage(
            job_id=job.id,
            competition_id=job.competition_id,
            submission_id=job.submission_id,
            miner_hotkey=job.miner_hotkey,
            hf_repo_id=job.hf_repo_id,
            env_provider=job.env_provider,
            benchmark_name=job.benchmark_name,
            config=env_config,
            artifact_url=artifact_url,
            artifact_expires_at=artifact_expires_at,
            artifact_sha256=job.artifact_sha256,
            artifact_size_bytes=job.artifact_size_bytes,
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
