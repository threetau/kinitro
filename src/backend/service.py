"""

This provides REST API endpoints and WebSocket connections for:

- Competition management
- Validator connections
- Job distribution
- Result collection
"""

import asyncio
import copy
import hashlib
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import dotenv
from asyncpg.exceptions import DeadlockDetectedError
from fastapi import (
    WebSocket,
)
from fiber.chain.fetch_nodes import _get_nodes_for_uid
from fiber.chain.interface import get_substrate
from fiber.chain.models import Node
from snowflake import SnowflakeGenerator
from sqlalchemy import and_, func, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from backend.constants import (
    CHAIN_SCAN_YIELD_INTERVAL,
    DEFAULT_BURN_PCT,
    DEFAULT_CHAIN_SYNC_INTERVAL,
    DEFAULT_MAX_COMMITMENT_LOOKBACK,
    DEFAULT_OWNER_UID,
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
    EpisodeCompletedEvent,
    EpisodeStepEvent,
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
    EpisodeDataMessage,
    EpisodeStepDataMessage,
    EvalJobMessage,
    EventType,
    MessageType,
    SetWeightsMessage,
)
from core.schemas import ChainCommitmentResponse, ModelProvider
from core.storage import load_s3_config

from .config import BackendConfig
from .models import (
    BackendEvaluationJob,
    BackendEvaluationJobStatus,
    BackendEvaluationResult,
    BackendState,
    Competition,
    CompetitionLeaderCandidate,
    EpisodeData,
    EpisodeStepData,
    LeaderCandidateStatus,
    MinerSubmission,
    SS58Address,
    SubmissionUpload,
    SubmissionUploadStatus,
    ValidatorConnection,
    WeightsSnapshot,
)

dotenv.load_dotenv()

logger = get_logger(__name__)

ConnectionId = str  # Unique ID for each WebSocket connection

VALIDATOR_MESSAGE_QUEUE_MAXSIZE = 5000
VALIDATOR_MESSAGE_BATCH_SIZE = 50
VALIDATOR_MESSAGE_BATCH_INTERVAL = 0.5
DEFAULT_VALIDATOR_MESSAGE_WORKERS = max(1, (os.cpu_count() or 1) * 2 + 1)

EpisodeKey = Tuple[str, int, str, SS58Address]


def _ensure_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    if isinstance(value, str):
        return datetime.fromisoformat(value)
    raise ValueError(f"Unsupported datetime value: {value!r}")


def _extract_benchmark_spec_payload(
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    Split a stored benchmark configuration into the full benchmark spec and the
    underlying execution config used by the evaluator.
    """
    spec_copy = copy.deepcopy(dict(config))
    try:
        base_config_source = config["config"]
    except KeyError as exc:  # pragma: no cover - defensive guard
        raise ValueError("Benchmark spec is missing 'config' payload") from exc
    try:
        base_config = copy.deepcopy(dict(base_config_source))
    except TypeError as exc:  # pragma: no cover - defensive guard
        raise ValueError("'config' payload must be a mapping") from exc
    return spec_copy, base_config


def _normalize_benchmark_spec_payload(
    provider: str,
    benchmark_name: str,
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """
    Ensure a benchmark specification payload includes top-level metadata and a nested config mapping.

    Accepts either the new-style payload (with a `config` key) or a bare config mapping and
    returns a copy that always matches the new-style structure.
    """
    if payload is None:
        base_config: dict[str, Any] = {}
        return {
            "provider": provider,
            "benchmark_name": benchmark_name,
            "config": base_config,
        }

    payload_dict = dict(payload)
    if "config" in payload_dict:
        return copy.deepcopy(payload_dict)

    base_config = copy.deepcopy(payload_dict)
    return {
        "provider": provider,
        "benchmark_name": benchmark_name,
        "config": base_config,
    }


class LeaderCandidateError(Exception):
    """Base exception for leader candidate operations."""


class LeaderCandidateNotFoundError(LeaderCandidateError):
    """Raised when a leader candidate cannot be located."""


class LeaderCandidateAlreadyReviewedError(LeaderCandidateError):
    """Raised when attempting to re-review a processed leader candidate."""


class LeaderCandidateNotApprovedError(LeaderCandidateError):
    """Raised when attempting to modify an unapproved leader candidate."""


class SubmissionNotFoundError(Exception):
    """Raised when a submission cannot be located."""


class EvaluationJobNotFoundError(Exception):
    """Raised when an evaluation job cannot be located."""


class NoBenchmarksAvailableError(Exception):
    """Raised when no benchmarks are available for an evaluation rerun."""


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

        owner_uid_setting = config.settings.get("owner_uid", DEFAULT_OWNER_UID)
        try:
            self.owner_uid = int(owner_uid_setting)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid owner_uid setting %r; falling back to default %s",
                owner_uid_setting,
                DEFAULT_OWNER_UID,
            )
            self.owner_uid = DEFAULT_OWNER_UID

        burn_pct_setting = config.settings.get("burn_pct", DEFAULT_BURN_PCT)
        try:
            burn_pct = float(burn_pct_setting)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid burn_pct setting %r; falling back to default %.3f",
                burn_pct_setting,
                DEFAULT_BURN_PCT,
            )
            burn_pct = DEFAULT_BURN_PCT

        if burn_pct < 0 or burn_pct > 1:
            logger.warning(
                "Configured burn_pct %.3f out of bounds [0, 1]; clamping.",
                burn_pct,
            )
            burn_pct = max(0.0, min(1.0, burn_pct))

        self.burn_pct = burn_pct
        logger.info("Burn percentage configured at %.2f%%", self.burn_pct * 100)

        # Scoring and weight broadcast intervals
        self.score_evaluation_interval = config.settings.get(
            "score_evaluation_interval", SCORE_EVALUATION_INTERVAL
        )
        self.weight_broadcast_interval = config.settings.get(
            "weight_broadcast_interval", WEIGHT_BROADCAST_INTERVAL
        )
        timeout_setting = config.settings.get(
            "job_timeout_seconds",
            config.settings.get("job_timeout", EVAL_JOB_TIMEOUT),
        )
        self.default_job_timeout_seconds = self._resolve_job_timeout_seconds(
            timeout_setting
        )
        logger.info(
            "Default evaluation job timeout set to %s seconds",
            self.default_job_timeout_seconds,
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

        self.s3_config = load_s3_config()
        logger.debug(f"S3 Config: {self.s3_config}")
        self.submission_storage: Optional[SubmissionStorage] = (
            SubmissionStorage(self.s3_config) if self.s3_config else None
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
        self._latest_weights_snapshot: Optional[WeightsSnapshot] = None

        # ID generator
        self.id_generator = SnowflakeGenerator(42)

        # Database
        self.engine: AsyncEngine = None
        self.async_session: async_sessionmaker[AsyncSession] = None

        # Thread pool for blocking operations
        self.thread_pool = ThreadPoolExecutor(max_workers=MAX_WORKERS)

        self._default_validator_worker_count = DEFAULT_VALIDATOR_MESSAGE_WORKERS
        self.validator_worker_count = self._resolve_validator_worker_count(
            config.settings.get("validator_message_workers")
        )

        # Validator message processing
        self._validator_message_queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(
            maxsize=VALIDATOR_MESSAGE_QUEUE_MAXSIZE
        )
        self._validator_worker_tasks: List[asyncio.Task] = []
        self._validator_queue_warning_threshold = int(
            VALIDATOR_MESSAGE_QUEUE_MAXSIZE * 0.8
        )
        self._validator_queue_warning_triggered = False

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

    def _resolve_job_timeout_seconds(self, configured_value: Any) -> int:
        """Convert configured timeout to a positive integer with a sane default."""
        default_timeout = EVAL_JOB_TIMEOUT
        try:
            timeout_seconds = int(configured_value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid job timeout value %r; using default %s seconds",
                configured_value,
                default_timeout,
            )
            return default_timeout

        if timeout_seconds <= 0:
            logger.warning(
                "Configured job timeout %s is non-positive; using default %s seconds",
                timeout_seconds,
                default_timeout,
            )
            return default_timeout

        return timeout_seconds

    def _job_timeout_seconds(self, competition: Optional[Competition]) -> int:
        """Return the timeout for a competition, falling back to the default."""
        if competition and competition.job_timeout_seconds:
            try:
                value = int(competition.job_timeout_seconds)
                if value > 0:
                    return value
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid job_timeout_seconds for competition %s: %r",
                    getattr(competition, "id", "unknown"),
                    getattr(competition, "job_timeout_seconds", None),
                )
        return self.default_job_timeout_seconds

    def _resolve_validator_worker_count(self, configured_value: Any) -> int:
        """Determine validator worker pool size from config or CPU count."""

        default_workers = self._default_validator_worker_count

        if configured_value is None:
            logger.info(
                "Validator message workers defaulting to %s (cpu cores=%s)",
                default_workers,
                os.cpu_count() or 1,
            )
            return default_workers

        try:
            workers = int(configured_value)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid validator_message_workers value %r; using default %s",
                configured_value,
                default_workers,
            )
            return default_workers

        if workers <= 0:
            logger.info(
                "validator_message_workers <= 0 (got %s); using default %s",
                workers,
                default_workers,
            )
            return default_workers

        logger.info("Validator message workers set to %s", workers)
        return workers

    async def create_submission_upload(
        self,
        *,
        miner_hotkey: SS58Address,
        competition_id: str,
        version: str,
        artifact_sha256: str,
        artifact_size_bytes: int,
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

            max_size = competition.submission_max_size_bytes
            if max_size is not None and artifact_size_bytes > max_size:
                raise RuntimeError(
                    "Submission artifact exceeds the competition size limit "
                    f"({artifact_size_bytes} bytes > {max_size} bytes)"
                )

            uploads_per_window = competition.submission_uploads_per_window
            window_seconds = competition.submission_upload_window_seconds
            if uploads_per_window and window_seconds:
                window_start = datetime.utcnow() - timedelta(seconds=window_seconds)
                recent_uploads_stmt = await session.execute(
                    select(func.count(SubmissionUpload.submission_id)).where(
                        SubmissionUpload.miner_hotkey == miner_hotkey,
                        SubmissionUpload.competition_id == competition_id,
                        SubmissionUpload.created_at >= window_start,
                    )
                )
                recent_uploads = recent_uploads_stmt.scalar() or 0
                if recent_uploads >= uploads_per_window:
                    raise RuntimeError(
                        "Upload rate limit exceeded for this hotkey; "
                        "please wait before requesting another upload slot"
                    )

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

        for worker_id in range(self.validator_worker_count):
            worker_task = asyncio.create_task(self._validator_message_worker(worker_id))
            self._validator_worker_tasks.append(worker_task)

        if self._validator_worker_tasks:
            logger.info(
                "Started %d validator message workers",
                len(self._validator_worker_tasks),
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

        for idx, task in enumerate(self._validator_worker_tasks):
            tasks_to_cancel.append((task, f"validator_worker_{idx}"))

        for task, name in tasks_to_cancel:
            if task:
                logger.info(f"Cancelling {name} task")
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError as e:
                    logger.error(f"{name} task cancelled: {e}")

        self._validator_worker_tasks.clear()

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
                        current_time = datetime.now(timezone.utc)
                        # Prefilter jobs older than the minimum configured timeout; exact check happens per job below.
                        min_timeout_result = await session.execute(
                            select(func.min(Competition.job_timeout_seconds))
                        )
                        min_timeout = min_timeout_result.scalar()
                        candidates = [
                            t
                            for t in (
                                min_timeout,
                                self.default_job_timeout_seconds,
                            )
                            if t and t > 0
                        ]
                        min_timeout_seconds = min(candidates) if candidates else 1
                        prefilter_threshold = current_time - timedelta(
                            seconds=min_timeout_seconds
                        )
                        prefilter_threshold = prefilter_threshold.replace(tzinfo=None)

                        result = await session.execute(
                            select(BackendEvaluationJob).where(
                                BackendEvaluationJob.created_at < prefilter_threshold
                            )
                        )

                        candidate_jobs = result.scalars().all()

                        for job in candidate_jobs:
                            job_timeout = self._resolve_job_timeout_seconds(
                                job.timeout_seconds or self.default_job_timeout_seconds
                            )
                            stale_threshold = current_time - timedelta(
                                seconds=job_timeout
                            )

                            if job.created_at and job.created_at >= stale_threshold:
                                continue

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

    async def queue_validator_heartbeat(
        self, validator_hotkey: SS58Address, timestamp: datetime
    ) -> None:
        """Queue a validator heartbeat for asynchronous persistence."""

        enqueued = await self._enqueue_validator_message(
            {
                "type": "heartbeat",
                "validator_hotkey": validator_hotkey,
                "timestamp": timestamp,
            },
            block=False,
        )

        if not enqueued:
            logger.warning(
                "Dropping heartbeat for %s due to full validator queue",
                validator_hotkey,
            )

    async def queue_episode_data(
        self, validator_hotkey: SS58Address, message: EpisodeDataMessage
    ) -> None:
        """Queue episode data for asynchronous persistence."""

        enqueued = await self._enqueue_validator_message(
            {
                "type": "episode_data",
                "validator_hotkey": validator_hotkey,
                "message": message,
            }
        )

        if not enqueued:
            logger.error(
                "Failed to enqueue episode data for submission %s episode %s",
                message.submission_id,
                message.episode_id,
            )

    async def queue_episode_step_data(
        self, validator_hotkey: SS58Address, message: EpisodeStepDataMessage
    ) -> None:
        """Queue episode step data for asynchronous persistence."""

        enqueued = await self._enqueue_validator_message(
            {
                "type": "episode_step_data",
                "validator_hotkey": validator_hotkey,
                "message": message,
            }
        )

        if not enqueued:
            logger.error(
                "Failed to enqueue step data for submission %s episode %s step %s",
                message.submission_id,
                message.episode_id,
                message.step,
            )

    async def _enqueue_validator_message(
        self, payload: Dict[str, Any], *, block: bool = True
    ) -> bool:
        """Enqueue a validator message, optionally dropping if queue is saturated."""

        if not self._running:
            logger.debug(
                "Received validator message %s while backend not running",
                payload.get("type"),
            )
            return False

        queue_size = self._validator_message_queue.qsize()
        if queue_size > self._validator_queue_warning_threshold:
            if not self._validator_queue_warning_triggered:
                logger.warning(
                    "Validator message queue high water mark: %s/%s",
                    queue_size,
                    VALIDATOR_MESSAGE_QUEUE_MAXSIZE,
                )
                self._validator_queue_warning_triggered = True
        elif self._validator_queue_warning_triggered and queue_size < (
            self._validator_queue_warning_threshold // 2
        ):
            logger.info(
                "Validator message queue draining (current size %s)", queue_size
            )
            self._validator_queue_warning_triggered = False

        try:
            if block:
                await self._validator_message_queue.put(payload)
            else:
                self._validator_message_queue.put_nowait(payload)
            return True
        except asyncio.QueueFull:
            logger.error(
                "Validator message queue full; dropping %s message",
                payload.get("type"),
            )
            return False

    async def _validator_message_worker(self, worker_id: int) -> None:
        """Background worker that batches validator messages before persisting."""

        logger.info("Validator message worker %s started", worker_id)
        try:
            while self._running:
                try:
                    message = await self._validator_message_queue.get()
                except asyncio.CancelledError:
                    break

                batch = [message]
                loop = asyncio.get_running_loop()
                deadline = loop.time() + VALIDATOR_MESSAGE_BATCH_INTERVAL

                while len(batch) < VALIDATOR_MESSAGE_BATCH_SIZE:
                    timeout = deadline - loop.time()
                    if timeout <= 0:
                        break

                    try:
                        next_message = await asyncio.wait_for(
                            self._validator_message_queue.get(), timeout
                        )
                        batch.append(next_message)
                    except asyncio.TimeoutError:
                        break
                    except asyncio.CancelledError:
                        raise

                try:
                    await self._process_validator_batch(batch)
                except Exception as exc:  # pragma: no cover - best effort logging
                    logger.error(
                        "Validator message worker %s failed to process batch: %s",
                        worker_id,
                        exc,
                    )
                finally:
                    for _ in batch:
                        self._validator_message_queue.task_done()

        finally:
            logger.info("Validator message worker %s stopped", worker_id)

    async def _process_validator_batch(self, batch: List[Dict[str, Any]]) -> None:
        """Persist a batch of validator messages and emit related events."""

        if not batch:
            return

        if not self.async_session:
            logger.error("Database not initialized; dropping validator messages")
            return

        heartbeats, episode_payload_map, step_payload_map = (
            self._group_validator_messages(batch)
        )

        step_payload_map = {
            key: sorted(payloads, key=lambda payload: payload["message"].step)
            for key, payloads in step_payload_map.items()
        }

        all_episode_keys = sorted(
            set(episode_payload_map.keys()) | set(step_payload_map.keys()),
            key=lambda k: (k[0], k[2], k[1], k[3]),
        )

        step_table = EpisodeStepData.__table__
        placeholder_warned: set[EpisodeKey] = set()
        placeholder_keys: set[EpisodeKey] = set()

        max_retries = 5
        attempt = 0

        while True:
            try:
                pending_events: List[tuple[EventType, Any]] = []
                status_updates: List[Dict[str, Any]] = []
                status_update_keys: set[tuple[Any, SS58Address]] = set()

                async with self.async_session() as session:
                    episode_lookup: Dict[EpisodeKey, int] = {}

                    await self._apply_heartbeats(session, heartbeats)

                    for key in all_episode_keys:
                        episode_payload = episode_payload_map.get(key)
                        step_payloads = step_payload_map.get(key, [])
                        if not episode_payload and not step_payloads:
                            continue

                        await self._acquire_episode_lock(session, key)

                        current_episode_id = episode_lookup.get(key)
                        if current_episode_id is None and step_payloads:
                            existing_episode_id = await self._get_episode_id_from_db(
                                session, key
                            )
                            if existing_episode_id is not None:
                                episode_lookup[key] = existing_episode_id
                                current_episode_id = existing_episode_id

                        if episode_payload:
                            message: EpisodeDataMessage = episode_payload["message"]
                            validator_hotkey: SS58Address = episode_payload[
                                "validator_hotkey"
                            ]
                            logger.info(
                                "Applying episode summary submission=%s task=%s episode=%s validator=%s reward=%s steps=%s",
                                message.submission_id,
                                message.task_id,
                                message.episode_id,
                                validator_hotkey,
                                message.final_reward,
                                message.steps,
                            )
                            current_episode_id = await self._ensure_episode_record(
                                message,
                                validator_hotkey,
                                episode_lookup,
                                session,
                            )
                            if key in placeholder_keys:
                                logger.info(
                                    "Overwriting placeholder episode with summary submission=%s task=%s episode=%s validator=%s",
                                    message.submission_id,
                                    message.task_id,
                                    message.episode_id,
                                    validator_hotkey,
                                )
                                placeholder_keys.discard(key)

                            pending_events.append(
                                (
                                    EventType.EPISODE_COMPLETED,
                                    EpisodeCompletedEvent(
                                        job_id=message.job_id,
                                        submission_id=message.submission_id,
                                        validator_hotkey=validator_hotkey,
                                        episode_id=message.episode_id,
                                        env_name=message.env_name,
                                        benchmark_name=message.benchmark_name,
                                        final_reward=message.final_reward,
                                        success=message.success,
                                        steps=message.steps,
                                        start_time=_ensure_datetime(message.start_time),
                                        end_time=_ensure_datetime(message.end_time),
                                        extra_metrics=message.extra_metrics,
                                        created_at=datetime.now(timezone.utc),
                                    ),
                                ),
                            )

                            status_result = await session.execute(
                                select(BackendEvaluationJobStatus).where(
                                    BackendEvaluationJobStatus.job_id == message.job_id,
                                    BackendEvaluationJobStatus.validator_hotkey
                                    == validator_hotkey,
                                    BackendEvaluationJobStatus.status
                                    == EvaluationStatus.RUNNING,
                                )
                            )
                            if not status_result.scalar_one_or_none():
                                status_key = (message.job_id, validator_hotkey)
                                if status_key not in status_update_keys:
                                    status_updates.append(
                                        {
                                            "job_id": message.job_id,
                                            "validator_hotkey": validator_hotkey,
                                            "status": EvaluationStatus.RUNNING,
                                            "detail": f"Started processing episodes (episode {message.episode_id})",
                                        }
                                    )
                                    status_update_keys.add(status_key)

                        if not episode_payload and step_payloads:
                            if key not in placeholder_warned:
                                submission_id, episode_no, task_id, validator_key = key
                                logger.warning(
                                    "Episode summary missing for submission=%s task=%s episode=%s validator=%s; using placeholder values",
                                    submission_id,
                                    task_id,
                                    episode_no,
                                    validator_key,
                                )
                                placeholder_warned.add(key)

                        if current_episode_id is None and step_payloads:
                            placeholder_episode = self._placeholder_episode_from_step(
                                step_payloads[0]["message"]
                            )
                            logger.info(
                                "Creating placeholder episode for submission=%s task=%s episode=%s validator=%s",
                                placeholder_episode.submission_id,
                                placeholder_episode.task_id,
                                placeholder_episode.episode_id,
                                step_payloads[0]["validator_hotkey"],
                            )
                            placeholder_keys.add(key)
                            current_episode_id = await self._ensure_episode_record(
                                placeholder_episode,
                                step_payloads[0]["validator_hotkey"],
                                episode_lookup,
                                session,
                            )

                        for step_payload in step_payloads:
                            step_message: EpisodeStepDataMessage = step_payload[
                                "message"
                            ]
                            validator_hotkey = step_payload["validator_hotkey"]

                            episode_lookup_id = episode_lookup.get(key)
                            if episode_lookup_id is None:
                                episode_lookup_id = await self._get_episode_id_from_db(
                                    session, key
                                )
                                if episode_lookup_id is not None:
                                    episode_lookup[key] = episode_lookup_id
                            if episode_lookup_id is None:
                                placeholder_episode = (
                                    self._placeholder_episode_from_step(step_message)
                                )
                                logger.info(
                                    "Creating placeholder episode for submission=%s task=%s episode=%s validator=%s",
                                    placeholder_episode.submission_id,
                                    placeholder_episode.task_id,
                                    placeholder_episode.episode_id,
                                    validator_hotkey,
                                )
                                placeholder_keys.add(key)
                                episode_lookup_id = await self._ensure_episode_record(
                                    placeholder_episode,
                                    validator_hotkey,
                                    episode_lookup,
                                    session,
                                )

                            step_values = {
                                "id": next(self.id_generator),
                                "episode_id": episode_lookup_id,
                                "submission_id": step_message.submission_id,
                                "validator_hotkey": validator_hotkey,
                                "task_id": step_message.task_id,
                                "step": step_message.step,
                                "action": step_message.action,
                                "reward": step_message.reward,
                                "done": step_message.done,
                                "truncated": step_message.truncated,
                                "observation_refs": step_message.observation_refs,
                                "info": step_message.info,
                                "timestamp": _ensure_datetime(
                                    step_message.step_timestamp
                                ),
                            }

                            step_insert = (
                                insert(step_table)
                                .values(**step_values)
                                .on_conflict_do_update(
                                    index_elements=["episode_id", "step"],
                                    set_={
                                        "submission_id": step_values["submission_id"],
                                        "validator_hotkey": step_values[
                                            "validator_hotkey"
                                        ],
                                        "task_id": step_values["task_id"],
                                        "action": step_values["action"],
                                        "reward": step_values["reward"],
                                        "done": step_values["done"],
                                        "truncated": step_values["truncated"],
                                        "observation_refs": step_values[
                                            "observation_refs"
                                        ],
                                        "info": step_values["info"],
                                        "timestamp": step_values["timestamp"],
                                        "updated_at": func.now(),
                                    },
                                )
                            )

                            await session.execute(step_insert)

                            pending_events.append(
                                (
                                    EventType.EPISODE_STEP,
                                    EpisodeStepEvent(
                                        submission_id=step_message.submission_id,
                                        validator_hotkey=validator_hotkey,
                                        episode_id=step_message.episode_id,
                                        step=step_message.step,
                                        action=step_message.action,
                                        reward=step_message.reward,
                                        done=step_message.done,
                                        truncated=step_message.truncated,
                                        observation_refs=step_message.observation_refs,
                                        info=step_message.info,
                                    ),
                                )
                            )

                        if episode_payload is None and step_payloads:
                            submission_id, episode_no, task_id, validator_key = key
                            if episode_lookup.get(key) is None:
                                logger.warning(
                                    "Episode summary missing for submission=%s task=%s episode=%s validator=%s; using placeholder values",
                                    submission_id,
                                    task_id,
                                    episode_no,
                                    validator_key,
                                )

                    await session.commit()

                for update in status_updates:
                    await self._update_job_status(
                        update["job_id"],
                        update["validator_hotkey"],
                        update["status"],
                        update["detail"],
                    )

                for event_type, event_payload in pending_events:
                    await event_broadcaster.broadcast_event(event_type, event_payload)

                if heartbeats:
                    logger.debug("Processed %s heartbeat updates", len(heartbeats))
                if episode_payload_map:
                    logger.info(
                        "Persisted %s episode records", len(episode_payload_map)
                    )
                if step_payload_map:
                    logger.debug(
                        "Persisted %s episode step records",
                        sum(len(payloads) for payloads in step_payload_map.values()),
                    )

                break
            except DBAPIError as exc:
                if (
                    isinstance(exc.orig, DeadlockDetectedError)
                    and attempt < max_retries
                ):
                    attempt += 1
                    delay = min(0.5 * attempt, 3.0)
                    logger.warning(
                        "Deadlock detected while processing validator batch (attempt %s/%s); retrying in %.2fs",
                        attempt,
                        max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise

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
            logger.trace(
                f"Miner {result.miner_hotkey} excluded from competition {competition.id}: "
                f"success_rate={result.success_rate:.3f} < min_threshold={competition.min_success_rate:.3f}"
            )
            return False

        if result.avg_reward < competition.min_avg_reward:
            logger.trace(
                f"Miner {result.miner_hotkey} excluded from competition {competition.id}: "
                f"avg_reward={result.avg_reward:.3f} < min_threshold={competition.min_avg_reward}"
            )
            return False

        return True

    async def _queue_leader_candidate(
        self,
        session: AsyncSession,
        competition: Competition,
        result: BackendEvaluationResult,
    ) -> bool:
        """Persist a leader candidate if not already recorded for this result."""

        if result.avg_reward is None:
            logger.debug(
                "Skipping leader candidate creation without avg_reward: competition=%s result_id=%s",
                competition.id,
                result.id,
            )
            return False

        existing_candidate_result = await session.execute(
            select(CompetitionLeaderCandidate).where(
                CompetitionLeaderCandidate.evaluation_result_id == result.id
            )
        )
        existing_candidate = existing_candidate_result.scalar_one_or_none()
        if existing_candidate:
            logger.debug(
                "Leader candidate already exists for evaluation result %s (competition=%s)",
                result.id,
                competition.id,
            )
            return False

        candidate = CompetitionLeaderCandidate(
            id=next(self.id_generator),
            competition_id=competition.id,
            miner_hotkey=result.miner_hotkey,
            evaluation_result_id=result.id,
            avg_reward=result.avg_reward,
            success_rate=result.success_rate,
            score=result.score,
            total_episodes=result.total_episodes,
        )
        session.add(candidate)
        return True

    async def approve_leader_candidate(
        self,
        candidate_id: int,
        admin_api_key_id: int,
        reason: Optional[str] = None,
    ) -> CompetitionLeaderCandidate:
        """Approve a pending leader candidate and promote them to current leader."""

        if not self.async_session:
            raise RuntimeError("Database not initialized")

        async with self.async_session() as session:
            candidate = await session.get(CompetitionLeaderCandidate, candidate_id)
            if not candidate:
                raise LeaderCandidateNotFoundError(
                    f"Leader candidate {candidate_id} not found"
                )

            if candidate.status != LeaderCandidateStatus.PENDING:
                raise LeaderCandidateAlreadyReviewedError(
                    f"Leader candidate {candidate_id} has already been reviewed"
                )

            competition = await session.get(Competition, candidate.competition_id)
            if not competition:
                raise LeaderCandidateNotFoundError(
                    f"Competition {candidate.competition_id} not found for candidate"
                )

            now = datetime.now(timezone.utc)

            candidate.status = LeaderCandidateStatus.APPROVED
            candidate.status_reason = reason
            candidate.reviewed_by_api_key_id = admin_api_key_id
            candidate.reviewed_at = now

            competition.current_leader_hotkey = candidate.miner_hotkey
            competition.current_leader_reward = candidate.avg_reward
            competition.leader_updated_at = now

            await session.commit()
            await session.refresh(candidate)

        try:
            await self._broadcast_stats_update()
        except Exception as exc:
            logger.error(
                "Failed to broadcast stats after leader candidate approval: %s",
                exc,
            )

        return candidate

    async def reject_leader_candidate(
        self,
        candidate_id: int,
        admin_api_key_id: int,
        reason: Optional[str] = None,
    ) -> CompetitionLeaderCandidate:
        """Reject a pending leader candidate."""

        if not self.async_session:
            raise RuntimeError("Database not initialized")

        async with self.async_session() as session:
            candidate = await session.get(CompetitionLeaderCandidate, candidate_id)
            if not candidate:
                raise LeaderCandidateNotFoundError(
                    f"Leader candidate {candidate_id} not found"
                )

            if candidate.status != LeaderCandidateStatus.PENDING:
                raise LeaderCandidateAlreadyReviewedError(
                    f"Leader candidate {candidate_id} has already been reviewed"
                )

            now = datetime.now(timezone.utc)

            candidate.status = LeaderCandidateStatus.REJECTED
            candidate.status_reason = reason
            candidate.reviewed_by_api_key_id = admin_api_key_id
            candidate.reviewed_at = now

            await session.commit()
            await session.refresh(candidate)

        try:
            await self._broadcast_stats_update()
        except Exception as exc:
            logger.error(
                "Failed to broadcast stats after leader candidate rejection: %s",
                exc,
            )

        return candidate

    async def unapprove_leader_candidate(
        self,
        candidate_id: int,
        admin_api_key_id: int,
        reason: Optional[str] = None,
    ) -> CompetitionLeaderCandidate:
        """Revert an approved leader candidate back to pending state."""

        if not self.async_session:
            raise RuntimeError("Database not initialized")

        async with self.async_session() as session:
            candidate = await session.get(CompetitionLeaderCandidate, candidate_id)
            if not candidate:
                raise LeaderCandidateNotFoundError(
                    f"Leader candidate {candidate_id} not found"
                )

            if candidate.status != LeaderCandidateStatus.APPROVED:
                raise LeaderCandidateNotApprovedError(
                    f"Leader candidate {candidate_id} is not approved"
                )

            competition = await session.get(Competition, candidate.competition_id)
            if not competition:
                raise LeaderCandidateNotFoundError(
                    f"Competition {candidate.competition_id} not found for candidate"
                )

            previous_reviewed_at = candidate.reviewed_at
            was_current_leader = (
                competition.current_leader_hotkey == candidate.miner_hotkey
                and competition.leader_updated_at == previous_reviewed_at
            )

            candidate.status = LeaderCandidateStatus.PENDING
            candidate.status_reason = reason
            candidate.reviewed_by_api_key_id = None
            candidate.reviewed_at = None
            await session.flush()

            if was_current_leader:
                fallback_stmt = (
                    select(CompetitionLeaderCandidate)
                    .where(
                        CompetitionLeaderCandidate.competition_id
                        == candidate.competition_id,
                        CompetitionLeaderCandidate.status
                        == LeaderCandidateStatus.APPROVED,
                        CompetitionLeaderCandidate.id != candidate.id,
                    )
                    .order_by(CompetitionLeaderCandidate.reviewed_at.desc())
                    .limit(1)
                )
                fallback_result = await session.execute(fallback_stmt)
                fallback_candidate = fallback_result.scalar_one_or_none()

                if fallback_candidate:
                    competition.current_leader_hotkey = fallback_candidate.miner_hotkey
                    competition.current_leader_reward = fallback_candidate.avg_reward
                    competition.leader_updated_at = fallback_candidate.reviewed_at
                else:
                    competition.current_leader_hotkey = None
                    competition.current_leader_reward = None
                    competition.leader_updated_at = None

            await session.commit()
            await session.refresh(candidate)
            logger.info(
                "Admin %s unapproved leader candidate %s (competition=%s)",
                admin_api_key_id,
                candidate_id,
                candidate.competition_id,
            )

        try:
            await self._broadcast_stats_update()
        except Exception as exc:
            logger.error(
                "Failed to broadcast stats after leader candidate unapproval: %s",
                exc,
            )

        return candidate

    async def _score_evaluations(self) -> dict[SS58Address, float]:
        """
        Score completed evaluations with winner-takes-all per competition.

        Scoring logic:
        - Miners must meet minimum success rate threshold per competition to be considered
        - Miners must pass minimum avg reward threshold per competition
        - Eligible challengers above the approved leader's success rate (ordered by success_rate, then avg_reward) are queued for admin review
          - If the current leader improves or matches their approved success rate with a new result, that result is also queued
        - Current leader retains position until admin approval
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

                # Find eligible challengers and order them by success rate / avg reward
                eligible_results: List[BackendEvaluationResult] = [
                    result
                    for result in eval_results
                    if self._is_miner_eligible(result, competition)
                ]

                eligible_results.sort(
                    key=lambda res: (
                        res.success_rate
                        if res.success_rate is not None
                        else float("-inf"),
                        res.avg_reward if res.avg_reward is not None else float("-inf"),
                    ),
                    reverse=True,
                )

                if not eligible_results:
                    if competition.current_leader_hotkey:
                        logger.info(
                            "Competition %s: Current leader %s retains position (no eligible challengers)",
                            competition.id,
                            competition.current_leader_hotkey,
                        )
                    else:
                        logger.info(
                            "Competition %s: No eligible miners found", competition.id
                        )
                    continue

                current_leader = competition.current_leader_hotkey

                if current_leader is None:
                    queued_any = False
                    for res in eligible_results:
                        created_candidate = await self._queue_leader_candidate(
                            session, competition, res
                        )
                        if created_candidate:
                            queued_any = True
                            logger.info(
                                "Competition %s: Queued leader candidate %s (success_rate=%.3f, avg_reward=%.3f)",
                                competition.id,
                                res.miner_hotkey,
                                res.success_rate or 0.0,
                                res.avg_reward or 0.0,
                            )
                    if not queued_any:
                        logger.debug(
                            "Competition %s: All eligible results already queued as candidates",
                            competition.id,
                        )
                else:
                    leader_success_rate_stmt = (
                        select(
                            CompetitionLeaderCandidate.success_rate,
                            CompetitionLeaderCandidate.evaluation_result_id,
                        )
                        .where(
                            CompetitionLeaderCandidate.competition_id == competition.id,
                            CompetitionLeaderCandidate.miner_hotkey == current_leader,
                            CompetitionLeaderCandidate.status
                            == LeaderCandidateStatus.APPROVED,
                        )
                        .order_by(
                            CompetitionLeaderCandidate.reviewed_at.desc(),
                            CompetitionLeaderCandidate.updated_at.desc(),
                        )
                        .limit(1)
                    )
                    leader_success_rate_result = await session.execute(
                        leader_success_rate_stmt
                    )
                    leader_success_rate_row = leader_success_rate_result.first()
                    leader_success_rate = (
                        leader_success_rate_row[0] if leader_success_rate_row else None
                    )
                    leader_success_eval_id = (
                        leader_success_rate_row[1] if leader_success_rate_row else None
                    )
                    baseline_leader_success_rate = (
                        leader_success_rate if leader_success_rate is not None else -1.0
                    )

                    leader_best = next(
                        (
                            res
                            for res in eligible_results
                            if res.miner_hotkey == current_leader
                        ),
                        None,
                    )
                    if (
                        leader_best
                        and leader_best.avg_reward is not None
                        and leader_best.avg_reward != competition.current_leader_reward
                    ):
                        competition.current_leader_reward = leader_best.avg_reward
                        competition.leader_updated_at = datetime.now(timezone.utc)
                        logger.info(
                            "Competition %s: Updated leader %s reward to %.3f",
                            competition.id,
                            current_leader,
                            leader_best.avg_reward,
                        )

                    challengers: list[BackendEvaluationResult] = []
                    for res in eligible_results:
                        if (
                            res.success_rate is not None
                            and res.success_rate > baseline_leader_success_rate
                        ):
                            challengers.append(res)

                    if (
                        leader_best
                        and leader_best.success_rate is not None
                        and leader_best.success_rate >= baseline_leader_success_rate
                        and leader_best.id != leader_success_eval_id
                    ):
                        challengers.append(leader_best)

                    # Deduplicate challengers by evaluation_result_id while preserving order
                    seen_eval_ids: set[int] = set()
                    unique_challengers: list[BackendEvaluationResult] = []
                    for res in challengers:
                        if res.id in seen_eval_ids:
                            continue
                        seen_eval_ids.add(res.id)
                        unique_challengers.append(res)

                    for challenger in unique_challengers:
                        created_candidate = await self._queue_leader_candidate(
                            session, competition, challenger
                        )
                        if created_candidate:
                            logger.info(
                                "Competition %s: Challenger %s queued for admin review (success_rate=%.3f, avg_reward=%.3f, current leader=%s success_rate=%s)",
                                competition.id,
                                challenger.miner_hotkey,
                                challenger.success_rate or 0.0,
                                challenger.avg_reward or 0.0,
                                current_leader,
                                f"{leader_success_rate:.3f}"
                                if leader_success_rate is not None
                                else "unknown",
                            )
                        else:
                            logger.debug(
                                "Competition %s: Challenger %s already recorded as candidate",
                                competition.id,
                                challenger.miner_hotkey,
                            )

                # Award points only to the currently approved leader
                award_hotkey = competition.current_leader_hotkey
                if not award_hotkey:
                    logger.debug(
                        "Competition %s: Skipping score award (no approved leader)",
                        competition.id,
                    )
                    continue

                base_score = competition.points / total_points if total_points else 0
                if base_score == 0:
                    logger.debug(
                        "Competition %s: Skipping zero-point competition in scoring",
                        competition.id,
                    )
                    continue

                if award_hotkey in miner_scores:
                    logger.warning(
                        "Miner %s already won competition - skipping score from %s. Previous score: %.4f, would have added: %.4f",
                        award_hotkey,
                        competition.id,
                        miner_scores[award_hotkey],
                        base_score * (1 - self.burn_pct),
                    )
                    continue

                awarded_score = base_score * (1 - self.burn_pct)
                burned_score = base_score - awarded_score

                if awarded_score <= 0:
                    logger.info(
                        "Competition %s: Burned entire %.4f normalized score for %s (burn_pct=%.2f%%)",
                        competition.id,
                        base_score,
                        award_hotkey,
                        self.burn_pct * 100,
                    )
                    continue

                miner_scores[award_hotkey] = awarded_score
                if burned_score > 0:
                    logger.info(
                        "Competition %s: Awarded %.4f normalized score to %s (burned %.4f; burn_pct=%.2f%%)",
                        competition.id,
                        awarded_score,
                        award_hotkey,
                        burned_score,
                        self.burn_pct * 100,
                    )
                else:
                    logger.info(
                        "Competition %s: Awarded %.4f normalized score to %s",
                        competition.id,
                        awarded_score,
                        award_hotkey,
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
                    "Score evaluation complete. %s miners scored.",
                    len(miner_scores),
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

            total_weight = sum(weights_dict.values())
            if total_weight > 1.0:
                logger.warning(
                    "Total miner weight %.6f exceeds 1.0 before owner allocation",
                    total_weight,
                )

            owner_weight = max(0.0, 1.0 - total_weight)
            if owner_weight > 0:
                weights_dict[self.owner_uid] = (
                    weights_dict.get(self.owner_uid, 0.0) + owner_weight
                )
                if all(node.node_id != self.owner_uid for node in self.nodes.values()):
                    logger.warning(
                        "Owner UID %s not found in node list; assigning %.4f weight without hotkey mapping",
                        self.owner_uid,
                        owner_weight,
                    )
                else:
                    logger.info(
                        "Owner UID %s assigned remaining normalized score %.4f (burn_pct=%.2f%%)",
                        self.owner_uid,
                        owner_weight,
                        self.burn_pct * 100,
                    )

            if not weights_dict:
                logger.info("No miner scores to broadcast")
                return

            # Populate missing entries with 0.0 weight for all nodes
            for node in self.nodes.values():
                weights_dict.setdefault(node.node_id, 0.0)

            snapshot_total_weight = float(sum(weights_dict.values()))
            self._latest_weights_snapshot = WeightsSnapshot(
                updated_at=datetime.now(timezone.utc),
                total_weight=snapshot_total_weight,
                weights=weights_dict.copy(),
            )

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

    def get_latest_weights_snapshot(self) -> Optional[WeightsSnapshot]:
        """Return the most recent weight broadcast snapshot, if available."""
        snapshot = self._latest_weights_snapshot
        if not snapshot:
            return None

        return snapshot.model_copy(deep=True)

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

            if provider == ModelProvider.S3:
                await self._process_s3_commitment(
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

    async def _process_s3_commitment(
        self,
        commitment: ChainCommitmentResponse,
        block_num: int,
        competition: Competition,
    ) -> None:
        """Handle commitments referencing direct-vault submissions."""

        if not self.submission_storage:
            logger.error(
                "Submission storage not configured; cannot process S3 commitment"
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
                hf_repo_id=f"s3:{submission_id}",
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

                if isinstance(benchmark, dict):
                    benchmark_spec = copy.deepcopy(benchmark)
                else:
                    logger.warning(
                        "Benchmark specification for competition %s is not a dict (%r); "
                        "wrapping in config field",
                        competition.id,
                        type(benchmark),
                    )
                    benchmark_spec = {"config": benchmark}

                job = BackendEvaluationJob(
                    id=next(self.id_generator),
                    submission_id=submission.id,
                    competition_id=competition.id,
                    miner_hotkey=submission.miner_hotkey,
                    hf_repo_id=submission.hf_repo_id,
                    env_provider=benchmark_spec.get("provider", benchmark["provider"]),
                    benchmark_name=benchmark_spec.get(
                        "benchmark_name", benchmark["benchmark_name"]
                    ),
                    config=benchmark_spec,
                    timeout_seconds=self._job_timeout_seconds(competition),
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
            await self._publish_new_jobs(jobs)

            logger.info(
                "Processed S3 submission %s from miner %s",
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

    async def rerun_submission_evaluations(
        self,
        submission_id: int,
        benchmark_names: Optional[List[str]] = None,
        requested_by_api_key_id: Optional[int] = None,
    ) -> List[BackendEvaluationJob]:
        """Re-run evaluations for a submission across its configured benchmarks."""

        if not self.async_session:
            raise RuntimeError("Database not initialized")

        benchmark_filter = (
            {name.strip() for name in benchmark_names if name.strip()}
            if benchmark_names
            else None
        )

        new_jobs: List[BackendEvaluationJob] = []

        async with self.async_session() as session:
            submission = await session.get(MinerSubmission, submission_id)
            if not submission:
                raise SubmissionNotFoundError(f"Submission {submission_id} not found")

            competition = await session.get(Competition, submission.competition_id)
            if not competition:
                raise SubmissionNotFoundError(
                    f"Competition {submission.competition_id} not found for submission {submission_id}"
                )

            benchmarks = competition.benchmarks or []
            for benchmark in benchmarks:
                provider = benchmark.get("provider")
                benchmark_name = benchmark.get("benchmark_name")

                if not provider or not benchmark_name:
                    logger.error(
                        "Submission %s rerun skipped invalid benchmark entry: %s",
                        submission_id,
                        benchmark,
                    )
                    continue

                if benchmark_filter and benchmark_name not in benchmark_filter:
                    continue
                spec_payload = _normalize_benchmark_spec_payload(
                    provider,
                    benchmark_name,
                    benchmark,
                )

                job = BackendEvaluationJob(
                    id=next(self.id_generator),
                    submission_id=submission.id,
                    competition_id=competition.id,
                    miner_hotkey=submission.miner_hotkey,
                    hf_repo_id=submission.hf_repo_id,
                    env_provider=provider,
                    benchmark_name=benchmark_name,
                    config=spec_payload,
                    timeout_seconds=self._job_timeout_seconds(competition),
                    artifact_object_key=submission.artifact_object_key,
                    artifact_sha256=submission.artifact_sha256,
                    artifact_size_bytes=submission.artifact_size_bytes,
                )
                new_jobs.append(job)

            if not new_jobs:
                raise NoBenchmarksAvailableError(
                    "No matching benchmarks available for rerun request"
                )

            session.add_all(new_jobs)
            await session.commit()

            for job in new_jobs:
                await session.refresh(job)

        await self._publish_new_jobs(new_jobs)

        logger.info(
            "Submission %s rerun triggered by API key %s; queued %s jobs",
            submission_id,
            requested_by_api_key_id,
            len(new_jobs),
        )

        return new_jobs

    async def rerun_job_evaluation(
        self,
        job_id: int,
        requested_by_api_key_id: Optional[int] = None,
    ) -> BackendEvaluationJob:
        """Re-run a specific evaluation job by cloning its configuration."""

        if not self.async_session:
            raise RuntimeError("Database not initialized")

        async with self.async_session() as session:
            existing_job = await session.get(BackendEvaluationJob, job_id)
            if not existing_job:
                raise EvaluationJobNotFoundError(f"Job {job_id} not found")

            spec_payload = _normalize_benchmark_spec_payload(
                existing_job.env_provider,
                existing_job.benchmark_name,
                existing_job.config
                if isinstance(existing_job.config, Mapping)
                else None,
            )

            new_job = BackendEvaluationJob(
                id=next(self.id_generator),
                submission_id=existing_job.submission_id,
                competition_id=existing_job.competition_id,
                miner_hotkey=existing_job.miner_hotkey,
                hf_repo_id=existing_job.hf_repo_id,
                env_provider=existing_job.env_provider,
                benchmark_name=existing_job.benchmark_name,
                config=spec_payload,
                timeout_seconds=existing_job.timeout_seconds,
                artifact_object_key=existing_job.artifact_object_key,
                artifact_sha256=existing_job.artifact_sha256,
                artifact_size_bytes=existing_job.artifact_size_bytes,
            )

            session.add(new_job)
            await session.commit()
            await session.refresh(new_job)

        await self._publish_new_jobs([new_job])

        logger.info(
            "Job %s rerun created as job %s by API key %s",
            job_id,
            new_job.id,
            requested_by_api_key_id,
        )

        return new_job

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

    async def _publish_new_jobs(self, jobs: Sequence[BackendEvaluationJob]) -> None:
        """Emit events and broadcasts for newly created jobs."""

        if not jobs:
            return

        connected_validator_hotkeys = tuple(
            dict.fromkeys(self.validator_connections.values())
        )

        for job in jobs:
            _benchmark_spec_payload, base_config_payload = (
                _extract_benchmark_spec_payload(job.config)
            )
            job_event = JobCreatedEvent(
                job_id=str(job.id),
                competition_id=job.competition_id,
                submission_id=job.submission_id,
                miner_hotkey=job.miner_hotkey,
                hf_repo_id=job.hf_repo_id,
                env_provider=job.env_provider,
                benchmark_name=job.benchmark_name,
                config=base_config_payload,
                status=EvaluationStatus.QUEUED,
                validator_statuses={
                    hotkey: EvaluationStatus.QUEUED
                    for hotkey in connected_validator_hotkeys
                },
            )

            try:
                await event_broadcaster.broadcast_event(
                    EventType.JOB_CREATED, job_event
                )
            except Exception as exc:  # pragma: no cover - broadcast best effort
                logger.error(
                    "Failed to broadcast job created event for job %s: %s",
                    job.id,
                    exc,
                )

            try:
                await self._broadcast_job(job)
            except Exception as exc:  # pragma: no cover - broadcast best effort
                logger.error("Failed to push job %s to validators: %s", job.id, exc)

        await self._broadcast_stats_update()

    async def _broadcast_job(self, job: BackendEvaluationJob):
        """Broadcast job to connected validators."""
        if not self.active_connections:
            logger.warning("No validators connected")
            return

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

        benchmark_spec_payload, base_config_payload = _extract_benchmark_spec_payload(
            job.config
        )

        timeout_seconds = job.timeout_seconds or self.default_job_timeout_seconds

        job_msg = EvalJobMessage(
            job_id=job.id,
            competition_id=job.competition_id,
            submission_id=job.submission_id,
            miner_hotkey=job.miner_hotkey,
            hf_repo_id=job.hf_repo_id,
            env_provider=job.env_provider,
            benchmark_name=job.benchmark_name,
            config=base_config_payload,
            benchmark_spec=benchmark_spec_payload,
            artifact_url=artifact_url,
            artifact_expires_at=artifact_expires_at,
            artifact_sha256=job.artifact_sha256,
            artifact_size_bytes=job.artifact_size_bytes,
            timeout_seconds=timeout_seconds,
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

    @staticmethod
    def _episode_score(message: EpisodeDataMessage) -> tuple[datetime, datetime, int]:
        return (
            _ensure_datetime(message.end_time),
            _ensure_datetime(message.start_time),
            message.steps,
        )

    def _group_validator_messages(
        self, batch: List[Dict[str, Any]]
    ) -> tuple[
        Dict[SS58Address, datetime],
        Dict[EpisodeKey, Dict[str, Any]],
        Dict[EpisodeKey, List[Dict[str, Any]]],
    ]:
        heartbeats: Dict[SS58Address, datetime] = {}
        episodes: Dict[EpisodeKey, Dict[str, Any]] = {}
        steps: Dict[EpisodeKey, List[Dict[str, Any]]] = defaultdict(list)

        for item in batch:
            message_type = item.get("type")
            if message_type == MessageType.HEARTBEAT:
                timestamp = item["timestamp"]
                hotkey = item["validator_hotkey"]
                current = heartbeats.get(hotkey)
                if current is None or timestamp > current:
                    heartbeats[hotkey] = timestamp
                continue

            if message_type == MessageType.EPISODE_DATA:
                message: EpisodeDataMessage = item["message"]
                validator_hotkey: SS58Address = item["validator_hotkey"]
                key: EpisodeKey = (
                    message.submission_id,
                    message.episode_id,
                    message.task_id,
                    validator_hotkey,
                )
                existing = episodes.get(key)
                if existing is None or self._episode_score(
                    message
                ) > self._episode_score(existing["message"]):
                    episodes[key] = item
                continue

            if message_type == MessageType.EPISODE_STEP_DATA:
                message = item["message"]
                validator_hotkey = item["validator_hotkey"]
                key = (
                    message.submission_id,
                    message.episode_id,
                    message.task_id,
                    validator_hotkey,
                )
                steps[key].append(item)
                continue

            logger.warning("Unknown validator message type: %s", message_type)

        return heartbeats, episodes, steps

    @staticmethod
    def _episode_lock_key(
        submission_id: str, task_id: str, episode_id: int
    ) -> tuple[int, int]:
        data = f"{submission_id}|{task_id}|{episode_id}".encode("utf-8")
        digest = hashlib.blake2b(data, digest_size=8).digest()
        part1 = int.from_bytes(digest[:4], "big", signed=True)
        part2 = int.from_bytes(digest[4:], "big", signed=True)
        return part1, part2

    def _placeholder_episode_from_step(
        self, step_message: EpisodeStepDataMessage
    ) -> EpisodeDataMessage:
        """Create a minimal episode payload so steps can be stored before summary arrives."""

        info = step_message.info or {}
        success = bool(info.get("success"))
        start_end = _ensure_datetime(step_message.step_timestamp)

        return EpisodeDataMessage(
            job_id=step_message.job_id,
            submission_id=step_message.submission_id,
            task_id=step_message.task_id,
            episode_id=step_message.episode_id,
            env_name=step_message.env_name,
            benchmark_name=step_message.benchmark_name,
            final_reward=info.get("_rollout_cumulative_reward", step_message.reward),
            success=success,
            steps=info.get("_rollout_total_steps", step_message.step),
            start_time=start_end,
            end_time=start_end,
            extra_metrics={"pending_summary": True},
        )

    async def _ensure_episode_record(
        self,
        message: EpisodeDataMessage,
        validator_hotkey: SS58Address,
        lookup: Dict[EpisodeKey, int],
        session: AsyncSession,
    ) -> int:
        key: EpisodeKey = (
            message.submission_id,
            message.episode_id,
            message.task_id,
            validator_hotkey,
        )
        existing_id = lookup.get(key)
        if existing_id is not None:
            return existing_id

        episode_table = EpisodeData.__table__
        episode_values = {
            "id": next(self.id_generator),
            "job_id": message.job_id,
            "submission_id": message.submission_id,
            "validator_hotkey": validator_hotkey,
            "task_id": message.task_id,
            "episode_id": message.episode_id,
            "env_name": message.env_name,
            "benchmark_name": message.benchmark_name,
            "final_reward": message.final_reward,
            "success": message.success,
            "steps": message.steps,
            "start_time": _ensure_datetime(message.start_time),
            "end_time": _ensure_datetime(message.end_time),
            "extra_metrics": message.extra_metrics,
        }

        episode_insert = (
            insert(episode_table)
            .values(**episode_values)
            .on_conflict_do_update(
                index_elements=[
                    "submission_id",
                    "task_id",
                    "episode_id",
                    "validator_hotkey",
                ],
                set_={
                    "job_id": episode_values["job_id"],
                    "validator_hotkey": episode_values["validator_hotkey"],
                    "env_name": episode_values["env_name"],
                    "benchmark_name": episode_values["benchmark_name"],
                    "final_reward": episode_values["final_reward"],
                    "success": episode_values["success"],
                    "steps": episode_values["steps"],
                    "start_time": episode_values["start_time"],
                    "end_time": episode_values["end_time"],
                    "extra_metrics": episode_values["extra_metrics"],
                    "updated_at": func.now(),
                },
            )
            .returning(episode_table.c.id)
        )

        result = await session.execute(episode_insert)
        episode_db_id = result.scalar_one()
        lookup[key] = episode_db_id
        logger.info(
            "Upserted episode summary submission=%s task=%s episode=%s validator=%s id=%s",
            message.submission_id,
            message.task_id,
            message.episode_id,
            validator_hotkey,
            episode_db_id,
        )
        return episode_db_id

    async def _get_episode_id_from_db(
        self, session: AsyncSession, key: EpisodeKey
    ) -> Optional[int]:
        submission_id, episode_id_value, task_id, validator_hotkey = key
        result = await session.execute(
            select(EpisodeData.id).where(
                EpisodeData.submission_id == submission_id,
                EpisodeData.episode_id == episode_id_value,
                EpisodeData.task_id == task_id,
                EpisodeData.validator_hotkey == validator_hotkey,
            )
        )
        episode_id = result.scalar_one_or_none()
        if episode_id is not None:
            logger.debug(
                "Loaded existing episode summary id=%s for submission=%s task=%s episode=%s validator=%s",
                episode_id,
                submission_id,
                task_id,
                episode_id_value,
                validator_hotkey,
            )
        return episode_id

    async def _apply_heartbeats(
        self, session: AsyncSession, heartbeats: Dict[SS58Address, datetime]
    ) -> None:
        if not heartbeats:
            return
        result = await session.execute(
            select(ValidatorConnection).where(
                ValidatorConnection.validator_hotkey.in_(heartbeats.keys())
            )
        )
        for validator in result.scalars():
            validator.last_heartbeat = heartbeats[validator.validator_hotkey]
            validator.is_connected = True
        await session.commit()

    async def _acquire_episode_lock(
        self, session: AsyncSession, key: EpisodeKey
    ) -> None:
        submission_id, episode_id_value, task_id, _ = key
        lock_k1, lock_k2 = self._episode_lock_key(
            submission_id, task_id, episode_id_value
        )
        await session.execute(
            text("SELECT pg_advisory_xact_lock(:k1, :k2)"),
            {"k1": lock_k1, "k2": lock_k2},
        )
