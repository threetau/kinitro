import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, cast

from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.auth import (
    UserRole,
    generate_api_key,
    get_api_key_from_db,
    hash_api_key,
)
from backend.auth_middleware import (
    ADMIN_FLAG_ATTR,
    AdminAuthMiddleware,
    ApiAuthMiddleware,
    admin_route,
)
from backend.constants import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    MIN_PAGE_LIMIT,
    SUBMISSION_SIGNATURE_MAX_AGE_SECONDS,
)
from backend.events import (
    EvaluationCompletedEvent,
    ValidatorConnectedEvent,
    ValidatorDisconnectedEvent,
)
from backend.realtime import event_broadcaster
from backend.service import (
    BackendService,
    EvaluationJobNotFoundError,
    LeaderCandidateAlreadyReviewedError,
    LeaderCandidateNotApprovedError,
    LeaderCandidateNotFoundError,
    NoBenchmarksAvailableError,
    SubmissionNotFoundError,
    WeightsSnapshot,
)
from core import __version__ as VERSION  # noqa: N812
from core.db.models import EvaluationStatus, SnowflakeId
from core.log import configure_logging, get_logger
from core.messages import (
    EpisodeDataMessage,
    EpisodeStepDataMessage,
    EvalResultMessage,
    EventType,
    JobStatusUpdateMessage,
    MessageType,
)
from core.schemas import ModelProvider

from .config import BackendConfig
from .models import (
    AgentLeaderboardEntry,
    ApiKey,
    ApiKeyCreateRequest,
    ApiKeyCreateResponse,
    ApiKeyResponse,
    BackendEvaluationJob,
    BackendEvaluationJobStatus,
    BackendEvaluationResult,
    BackendState,
    BackendStatsResponse,
    Competition,
    CompetitionCreateRequest,
    CompetitionLeaderboardResponse,
    CompetitionLeaderCandidate,
    CompetitionLeaderCandidateResponse,
    CompetitionLeaderInfo,
    CompetitionResponse,
    EpisodeData,
    EpisodeStepData,
    EvaluationLogDownloadResponse,
    EvaluationResultLogResponse,
    EvaluationResultResponse,
    JobResponse,
    JobStatusResponse,
    LeaderCandidateReviewRequest,
    LeaderCandidateStatus,
    MinerSubmission,
    MinerSubmissionResponse,
    RevealedSubmissionResponse,
    SubmissionLeaderboardEntry,
    SubmissionLeaderboardResponse,
    SubmissionRerunRequest,
    ValidatorConnection,
    ValidatorInfoResponse,
)


class SubmissionUploadRequest(BaseModel):
    competition_id: str
    version: str
    artifact_sha256: str
    artifact_size_bytes: int = Field(gt=0)
    timestamp: int = Field(ge=0)
    hotkey: str
    signature: str

    @field_validator("competition_id")
    @classmethod
    def _validate_competition_id(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("competition_id cannot be empty")
        return value

    @field_validator("version")
    @classmethod
    def _validate_version(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("version cannot be empty")
        return value

    @field_validator("artifact_sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        if not re.fullmatch(r"[0-9a-fA-F]{64}", value):
            raise ValueError(
                "artifact_sha256 must be a 64 character hexadecimal string"
            )
        return value.lower()

    @field_validator("signature")
    @classmethod
    def _validate_signature(cls, value: str) -> str:
        signature_body = value[2:] if value.startswith("0x") else value
        if not re.fullmatch(r"[0-9a-fA-F]{128}", signature_body):
            raise ValueError("signature must be a 64-byte hexadecimal string")
        return value.lower()

    @field_validator("hotkey")
    @classmethod
    def _validate_hotkey(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("hotkey cannot be empty")
        return value


class SubmissionUploadResponse(BaseModel):
    submission_id: int
    upload_url: str
    method: str
    expires_at: datetime
    headers: Dict[str, str]
    object_key: str
    artifact_sha256: str
    artifact_size_bytes: int
    commit_payload: Dict[str, Any]


def _build_submission_upload_message(payload: SubmissionUploadRequest) -> bytes:
    parts = [
        payload.hotkey,
        payload.competition_id,
        payload.version,
        payload.artifact_sha256,
        str(payload.artifact_size_bytes),
        str(payload.timestamp),
    ]
    return "|".join(parts).encode("utf-8")


# Create backend service instance
config = BackendConfig()
if config.log_file:
    configure_logging(config.log_file)

logger = get_logger(__name__)
backend_service = BackendService(config)
API_SECURITY_SCHEME_NAME = "ApiKeyAuth"


STATUS_PRIORITY = [
    EvaluationStatus.FAILED,
    EvaluationStatus.TIMEOUT,
    EvaluationStatus.CANCELLED,
    EvaluationStatus.RUNNING,
    EvaluationStatus.STARTING,
    EvaluationStatus.QUEUED,
]


def _derive_submission_status(
    statuses: Sequence[EvaluationStatus],
) -> Optional[EvaluationStatus]:
    """Collapse per-job statuses into a single submission-level status."""
    if not statuses:
        return None

    if all(status == EvaluationStatus.COMPLETED for status in statuses):
        return EvaluationStatus.COMPLETED

    for priority_status in STATUS_PRIORITY:
        if priority_status in statuses:
            return priority_status

    # Fallback - should not be reached, but preserve latest observed status
    return statuses[0]


async def _get_submission_evaluation_statuses(
    session: AsyncSession, submission_ids: Sequence[int]
) -> Dict[int, Optional[EvaluationStatus]]:
    """Fetch the aggregated evaluation status for each submission."""
    if not submission_ids:
        return {}

    job_rows = await session.execute(
        select(BackendEvaluationJob.id, BackendEvaluationJob.submission_id).where(
            BackendEvaluationJob.submission_id.in_(submission_ids)
        )
    )
    jobs_by_submission: Dict[int, List[int]] = {}
    for job_id, submission_id in job_rows:
        jobs_by_submission.setdefault(int(submission_id), []).append(int(job_id))

    if not jobs_by_submission:
        return {}

    job_ids = {
        job_id
        for submission_jobs in jobs_by_submission.values()
        for job_id in submission_jobs
    }
    job_id_values = tuple(job_ids)

    if not job_id_values:
        return {}

    latest_status_subquery = (
        select(
            BackendEvaluationJobStatus.job_id,
            func.max(BackendEvaluationJobStatus.created_at).label("max_created_at"),
        )
        .where(BackendEvaluationJobStatus.job_id.in_(job_id_values))
        .group_by(BackendEvaluationJobStatus.job_id)
        .subquery()
    )

    status_rows = await session.execute(
        select(
            BackendEvaluationJob.submission_id,
            BackendEvaluationJob.id,
            BackendEvaluationJobStatus.status,
        )
        .select_from(BackendEvaluationJob)
        .join(
            BackendEvaluationJobStatus,
            BackendEvaluationJob.id == BackendEvaluationJobStatus.job_id,
        )
        .join(
            latest_status_subquery,
            and_(
                BackendEvaluationJobStatus.job_id == latest_status_subquery.c.job_id,
                BackendEvaluationJobStatus.created_at
                == latest_status_subquery.c.max_created_at,
            ),
        )
        .where(BackendEvaluationJob.id.in_(job_id_values))
    )

    job_status_map: Dict[int, EvaluationStatus] = {}
    submission_statuses: Dict[int, List[EvaluationStatus]] = {}

    for submission_id, job_id, status_value in status_rows:
        status_enum = (
            status_value
            if isinstance(status_value, EvaluationStatus)
            else EvaluationStatus(status_value)
        )
        submission_key = int(submission_id)
        job_key = int(job_id)
        job_status_map[job_key] = status_enum
        submission_statuses.setdefault(submission_key, []).append(status_enum)

    # Default jobs without a status entry to QUEUED so they are reflected in aggregation
    for submission_id, job_ids_for_submission in jobs_by_submission.items():
        statuses = submission_statuses.setdefault(submission_id, [])
        for job_id in job_ids_for_submission:
            if job_id not in job_status_map:
                statuses.append(EvaluationStatus.QUEUED)

    aggregated_statuses: Dict[int, Optional[EvaluationStatus]] = {}
    for submission_id, statuses in submission_statuses.items():
        aggregated_statuses[submission_id] = _derive_submission_status(statuses)

    return aggregated_statuses


async def _get_leader_success_rates(
    session: AsyncSession, competitions: Sequence[Competition]
) -> Dict[str, float]:
    """Map competitions to the success rate of their current leader."""
    success_rates: Dict[str, float] = {}

    for competition in competitions:
        if not competition.current_leader_hotkey:
            continue

        result = await session.execute(
            select(CompetitionLeaderCandidate.success_rate)
            .where(
                CompetitionLeaderCandidate.competition_id == competition.id,
                CompetitionLeaderCandidate.status == LeaderCandidateStatus.APPROVED,
                CompetitionLeaderCandidate.miner_hotkey
                == competition.current_leader_hotkey,
            )
            .order_by(
                CompetitionLeaderCandidate.reviewed_at.desc(),
                CompetitionLeaderCandidate.updated_at.desc(),
            )
            .limit(1)
        )
        success_rate = result.scalar_one_or_none()
        if success_rate is not None:
            success_rates[competition.id] = float(success_rate)

    return success_rates


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage backend service lifecycle."""
    # Initialize the service but don't start background tasks yet
    await backend_service.startup()

    # Set backend service reference in event broadcaster
    event_broadcaster.set_backend_service(backend_service)

    # Start the event broadcaster
    await event_broadcaster.start()

    # Start background tasks after FastAPI is ready
    asyncio.create_task(backend_service.start_background_tasks())

    yield

    # Stop the event broadcaster
    await event_broadcaster.stop()

    await backend_service.shutdown()


# Create FastAPI app
app = FastAPI(
    title="Kinitro Backend API",
    description="Central coordination service for Kinitro system",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(AdminAuthMiddleware, backend_service=backend_service)
app.add_middleware(ApiAuthMiddleware, backend_service=backend_service)


def get_admin_user(request: Request) -> ApiKey:
    """Return the authenticated admin user set by the middleware."""
    return cast(ApiKey, request.state.api_user)


# ============================================================================
# HTTP API Endpoints
# ============================================================================


@app.get("/")
async def root() -> dict:
    """Root endpoint."""
    return {
        "service": "Kinitro Backend",
        "version": VERSION,
        "docs": "/docs",
        "health": "/health",
    }


def custom_openapi() -> dict:
    """Customize OpenAPI schema to advertise API key security."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )

    components = openapi_schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes[API_SECURITY_SCHEME_NAME] = {
        "type": "apiKey",
        "in": "header",
        "name": "X-API-Key",
        "description": "API key used for authenticated endpoints.",
    }

    paths = openapi_schema.get("paths", {})

    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if not endpoint or not getattr(endpoint, ADMIN_FLAG_ATTR, False):
            continue

        path_item = paths.get(route.path)
        if not path_item:
            continue

        for method in route.methods or []:
            method_key = method.lower()
            operation = path_item.get(method_key)
            if not operation:
                continue
            security = operation.setdefault("security", [])
            if {API_SECURITY_SCHEME_NAME: []} not in security:
                security.append({API_SECURITY_SCHEME_NAME: []})

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "kinitro-backend",
        "chain_connected": backend_service.substrate is not None,
        "database_connected": backend_service.engine is not None,
    }


@app.get("/weights", response_model=WeightsSnapshot)
async def get_latest_weights() -> WeightsSnapshot:
    """Expose the most recent weight snapshot for validators and monitoring tools."""
    snapshot = backend_service.get_latest_weights_snapshot()
    if not snapshot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Weights not available yet",
        )
    return snapshot


@app.post(
    "/submissions/request-upload",
    response_model=SubmissionUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def request_submission_upload(
    payload: SubmissionUploadRequest,
) -> SubmissionUploadResponse:
    """Create a presigned upload slot for a miner submission."""

    now = datetime.now(timezone.utc)
    timestamp_dt = datetime.fromtimestamp(payload.timestamp, tz=timezone.utc)
    if abs((now - timestamp_dt).total_seconds()) > SUBMISSION_SIGNATURE_MAX_AGE_SECONDS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signature timestamp is outside the allowed window",
        )

    signature_message = _build_submission_upload_message(payload)

    try:
        signature_valid = backend_service.verify_hotkey_signature(
            payload.hotkey, signature_message, payload.signature
        )
    except RuntimeError as exc:
        logger.error("Signature verification backend unavailable: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signature verification backend unavailable",
        ) from exc

    if not signature_valid:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid signature",
        )

    try:
        upload_record, presigned = await backend_service.create_submission_upload(
            miner_hotkey=payload.hotkey,
            competition_id=payload.competition_id,
            version=payload.version,
            artifact_sha256=payload.artifact_sha256,
            artifact_size_bytes=payload.artifact_size_bytes,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    commit_payload = {
        "provider": ModelProvider.S3.value,
        "submission_id": str(upload_record.submission_id),
        "artifact_sha256": upload_record.artifact_sha256,
        "artifact_size_bytes": upload_record.artifact_size_bytes,
        "competition_id": upload_record.competition_id,
    }

    return SubmissionUploadResponse(
        submission_id=upload_record.submission_id,
        upload_url=presigned.url,
        method=presigned.method,
        expires_at=presigned.expires_at,
        headers=presigned.headers,
        object_key=upload_record.artifact_object_key,
        artifact_sha256=upload_record.artifact_sha256,
        artifact_size_bytes=upload_record.artifact_size_bytes,
        commit_payload=commit_payload,
    )


@app.get("/stats", response_model=BackendStatsResponse)
async def get_stats() -> BackendStatsResponse:
    """Get comprehensive backend statistics."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
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
        sub_result = await session.execute(select(func.count(MinerSubmission.id)))
        total_submissions = sub_result.scalar() or 0

        # Get jobs count
        job_result = await session.execute(select(func.count(BackendEvaluationJob.id)))
        total_jobs = job_result.scalar() or 0

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
            percentage = (comp.points / total_points * 100) if total_points > 0 else 0
            comp_percentages[comp.id] = percentage

        return BackendStatsResponse(
            total_competitions=len(competitions),
            active_competitions=len(active_comps),
            total_points=total_points,
            connected_validators=connected_validators,
            total_submissions=total_submissions,
            total_jobs=total_jobs,
            total_results=total_results,
            last_seen_block=state.last_seen_block if state else 0,
            competition_percentages=comp_percentages,
        )


# Competition endpoints
@app.post("/competitions", response_model=CompetitionResponse)
@admin_route
async def create_competition(competition: CompetitionCreateRequest):
    """Create a new competition."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        db_competition = Competition(
            id=uuid.uuid4().hex,
            name=competition.name,
            description=competition.description,
            benchmarks=competition.benchmarks,
            points=competition.points,
            min_avg_reward=competition.min_avg_reward,
            win_margin_pct=competition.win_margin_pct,
            min_success_rate=competition.min_success_rate,
            submission_holdout_seconds=competition.submission_holdout_seconds,
            submission_max_size_bytes=competition.submission_max_size_bytes,
            submission_upload_window_seconds=competition.submission_upload_window_seconds,
            submission_uploads_per_window=competition.submission_uploads_per_window,
            active=True,
            start_time=competition.start_time,
            end_time=competition.end_time,
        )

        session.add(db_competition)
        await session.commit()
        await session.refresh(db_competition)

        return CompetitionResponse.model_validate(db_competition)


@app.get("/competitions", response_model=List[CompetitionResponse])
async def list_competitions(
    active_only: bool = Query(False, description="Filter for active competitions only"),
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
):
    """List all competitions."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        query = select(Competition)
        if active_only:
            query = query.where(Competition.active)
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        competitions = result.scalars().all()

        success_rate_map = await _get_leader_success_rates(session, competitions)

        responses: List[CompetitionResponse] = []
        for competition in competitions:
            response = CompetitionResponse.model_validate(competition)
            response.current_leader_success_rate = success_rate_map.get(competition.id)
            responses.append(response)

        return responses


@app.get(
    "/competitions/leaderboard",
    response_model=CompetitionLeaderboardResponse,
)
async def get_competition_leaderboard(
    active_only: bool = Query(
        True, description="Filter for active competitions only when true"
    ),
):
    """Aggregate current competition leaders into an agent leaderboard."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        query = select(Competition)
        if active_only:
            query = query.where(Competition.active)

        result = await session.execute(query)
        competitions = result.scalars().all()

        leader_submission_map: Dict[str, Optional[str]] = {}
        leader_success_rate_map: Dict[str, Optional[float]] = {}
        for competition in competitions:
            if not competition.current_leader_hotkey:
                continue

            submission_stmt = (
                select(
                    BackendEvaluationJob.submission_id,
                    CompetitionLeaderCandidate.success_rate,
                )
                .select_from(CompetitionLeaderCandidate)
                .join(
                    BackendEvaluationResult,
                    CompetitionLeaderCandidate.evaluation_result_id
                    == BackendEvaluationResult.id,
                )
                .join(
                    BackendEvaluationJob,
                    BackendEvaluationResult.job_id == BackendEvaluationJob.id,
                )
                .where(
                    CompetitionLeaderCandidate.competition_id == competition.id,
                    CompetitionLeaderCandidate.status == LeaderCandidateStatus.APPROVED,
                    CompetitionLeaderCandidate.miner_hotkey
                    == competition.current_leader_hotkey,
                )
                .order_by(
                    CompetitionLeaderCandidate.reviewed_at.desc(),
                    CompetitionLeaderCandidate.updated_at.desc(),
                )
                .limit(1)
            )

            submission_result = await session.execute(submission_stmt)
            leader_row = submission_result.first()
            if leader_row is None:
                continue

            submission_id, success_rate = leader_row
            if submission_id is not None:
                leader_submission_map[competition.id] = str(submission_id)
            if success_rate is not None:
                leader_success_rate_map[competition.id] = float(success_rate)

    total_points = sum(comp.points for comp in competitions)
    competition_infos = [
        CompetitionLeaderInfo(
            competition_id=comp.id,
            competition_name=comp.name,
            points=comp.points,
            current_leader_hotkey=comp.current_leader_hotkey,
            current_leader_submission_id=leader_submission_map.get(comp.id),
            current_leader_reward=comp.current_leader_reward,
            current_leader_success_rate=leader_success_rate_map.get(comp.id),
            leader_updated_at=comp.leader_updated_at,
        )
        for comp in competitions
    ]

    per_miner_points: dict[str, int] = {}
    per_miner_competitions: dict[str, List[str]] = {}
    per_miner_submissions: dict[str, Dict[str, str]] = {}

    for comp in competitions:
        leader_hotkey = comp.current_leader_hotkey
        if not leader_hotkey:
            continue

        per_miner_points[leader_hotkey] = (
            per_miner_points.get(leader_hotkey, 0) + comp.points
        )
        per_miner_competitions.setdefault(leader_hotkey, []).append(comp.id)
        submission_id = leader_submission_map.get(comp.id)
        if submission_id:
            per_miner_submissions.setdefault(leader_hotkey, {})[comp.id] = submission_id

    sorted_leaders = sorted(
        per_miner_points.items(),
        key=lambda item: (-item[1], item[0]),
    )

    leaderboard_entries: List[AgentLeaderboardEntry] = []
    for index, (hotkey, points) in enumerate(sorted_leaders, start=1):
        normalized_score = (points / total_points) if total_points else 0.0
        leaderboard_entries.append(
            AgentLeaderboardEntry(
                rank=index,
                miner_hotkey=hotkey,
                total_points=points,
                normalized_score=normalized_score,
                competitions=sorted(per_miner_competitions.get(hotkey, [])),
                competition_submission_ids=dict(per_miner_submissions.get(hotkey, {})),
            )
        )

    return CompetitionLeaderboardResponse(
        total_competitions=len(competitions),
        total_points=total_points,
        leaders=leaderboard_entries,
        competitions=competition_infos,
    )


@app.get(
    "/leaderboards/submissions",
    response_model=SubmissionLeaderboardResponse,
)
async def get_submission_leaderboard(
    competition_id: Optional[str] = Query(
        None, description="Filter evaluation results by competition ID"
    ),
    miner_hotkey: Optional[str] = Query(
        None, description="Filter evaluation results by miner hotkey"
    ),
    env_provider: Optional[str] = Query(
        None, description="Filter evaluation results by provider"
    ),
    benchmark: Optional[str] = Query(
        None, description="Filter evaluation results by benchmark name"
    ),
    validator_hotkey: Optional[str] = Query(
        None, description="Filter evaluation results by validator hotkey"
    ),
    sort_by: Literal["avg_reward", "success_rate", "score"] = Query(
        "avg_reward",
        description="Primary metric to sort by",
    ),
    sort_direction: Literal["asc", "desc"] = Query(
        "desc",
        description="Sort direction for the primary metric",
    ),
    min_results: int = Query(
        1, ge=1, description="Minimum number of evaluation results per submission"
    ),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
):
    """Rank submissions by aggregated evaluation metrics."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    metric_column_map = {
        "avg_reward": BackendEvaluationResult.avg_reward,
        "success_rate": BackendEvaluationResult.success_rate,
        "score": BackendEvaluationResult.score,
    }

    async with backend_service.async_session() as session:
        filters = [
            metric_column_map[sort_by].is_not(None),
        ]

        if competition_id:
            filters.append(BackendEvaluationResult.competition_id == competition_id)
        if miner_hotkey:
            filters.append(BackendEvaluationResult.miner_hotkey == miner_hotkey)
        if env_provider:
            filters.append(BackendEvaluationJob.env_provider == env_provider)
        if benchmark:
            filters.append(BackendEvaluationJob.benchmark_name == benchmark)
        if validator_hotkey:
            filters.append(BackendEvaluationResult.validator_hotkey == validator_hotkey)

        base_stmt = (
            select(
                BackendEvaluationJob.submission_id.label("submission_id"),
                BackendEvaluationJob.competition_id.label("competition_id"),
                BackendEvaluationResult.miner_hotkey.label("miner_hotkey"),
                MinerSubmission.version.label("version"),
                MinerSubmission.hf_repo_id.label("hf_repo_id"),
                func.avg(BackendEvaluationResult.avg_reward).label("avg_reward"),
                func.avg(BackendEvaluationResult.success_rate).label("success_rate"),
                func.avg(BackendEvaluationResult.score).label("score"),
                func.count(BackendEvaluationResult.id).label("result_count"),
                func.coalesce(
                    func.sum(BackendEvaluationResult.total_episodes), 0
                ).label("total_episodes"),
                func.max(BackendEvaluationResult.result_time).label("last_result_time"),
            )
            .join(
                BackendEvaluationJob,
                BackendEvaluationResult.job_id == BackendEvaluationJob.id,
            )
            .join(
                MinerSubmission,
                MinerSubmission.id == BackendEvaluationJob.submission_id,
            )
            .group_by(
                BackendEvaluationJob.submission_id,
                BackendEvaluationJob.competition_id,
                BackendEvaluationResult.miner_hotkey,
                MinerSubmission.version,
                MinerSubmission.hf_repo_id,
            )
        )

        if filters:
            base_stmt = base_stmt.where(*filters)

        if min_results > 1:
            base_stmt = base_stmt.having(
                func.count(BackendEvaluationResult.id) >= min_results
            )

        aggregation_subquery = base_stmt.subquery()

        sort_column_map = {
            "avg_reward": aggregation_subquery.c.avg_reward,
            "success_rate": aggregation_subquery.c.success_rate,
            "score": aggregation_subquery.c.score,
        }
        sort_column = sort_column_map[sort_by]
        primary_order = (
            sort_column.desc() if sort_direction == "desc" else sort_column.asc()
        )

        if sort_by == "avg_reward":
            secondary_column = aggregation_subquery.c.success_rate
        elif sort_by == "success_rate":
            secondary_column = aggregation_subquery.c.avg_reward
        else:
            secondary_column = aggregation_subquery.c.avg_reward

        secondary_order = secondary_column.desc()

        ranked_stmt = (
            select(*aggregation_subquery.c)
            .order_by(
                primary_order,
                secondary_order,
                aggregation_subquery.c.submission_id,
            )
            .offset(offset)
            .limit(limit)
        )

        results = await session.execute(ranked_stmt)
        rows = results.all()

        count_stmt = select(func.count()).select_from(aggregation_subquery)
        total_submissions = (await session.execute(count_stmt)).scalar() or 0

    entries: List[SubmissionLeaderboardEntry] = []
    for index, row in enumerate(rows, start=offset + 1):
        avg_reward_val = float(row.avg_reward) if row.avg_reward is not None else None
        success_rate_val = (
            float(row.success_rate) if row.success_rate is not None else None
        )
        score_val = float(row.score) if row.score is not None else None
        total_episodes_val = (
            int(row.total_episodes) if row.total_episodes is not None else None
        )
        entries.append(
            SubmissionLeaderboardEntry(
                rank=index,
                submission_id=str(row.submission_id),
                competition_id=row.competition_id,
                miner_hotkey=row.miner_hotkey,
                hf_repo_id=row.hf_repo_id,
                version=row.version,
                avg_reward=avg_reward_val,
                success_rate=success_rate_val,
                score=score_val,
                total_results=int(row.result_count),
                total_episodes=total_episodes_val,
                last_result_time=row.last_result_time,
            )
        )

    return SubmissionLeaderboardResponse(
        total_submissions=total_submissions,
        offset=offset,
        limit=limit,
        sort_by=sort_by,
        sort_direction=sort_direction,
        entries=entries,
    )


@app.get("/competitions/{competition_id}", response_model=CompetitionResponse)
async def get_competition(competition_id: str):
    """Get a specific competition by ID."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        competition = result.scalar_one_or_none()

        if not competition:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Competition not found"
            )

        success_rate_map = await _get_leader_success_rates(session, [competition])
        response = CompetitionResponse.model_validate(competition)
        response.current_leader_success_rate = success_rate_map.get(competition.id)

        return response


@app.patch("/competitions/{competition_id}/activate")
@admin_route
async def activate_competition(competition_id: str):
    """Activate a competition."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        competition = result.scalar_one_or_none()

        if not competition:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Competition not found"
            )

        competition.active = True
        await session.commit()

        return {"status": "activated", "competition_id": competition_id}


@app.patch("/competitions/{competition_id}/deactivate")
@admin_route
async def deactivate_competition(competition_id: str):
    """Deactivate a competition."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        competition = result.scalar_one_or_none()

        if not competition:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Competition not found"
            )

        competition.active = False
        await session.commit()

        return {"status": "deactivated", "competition_id": competition_id}


@app.delete("/competitions/{competition_id}")
@admin_route
async def delete_competition(competition_id: str):
    """Delete a competition (soft delete by deactivating)."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        competition = result.scalar_one_or_none()

        if not competition:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Competition not found"
            )

        competition.active = False
        await session.commit()

        return {"status": "deleted", "competition_id": competition_id}


# Validator endpoints
@app.get("/validators", response_model=List[ValidatorInfoResponse])
async def list_validators(
    connected_only: bool = Query(
        False, description="Filter for connected validators only"
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
):
    """List all validators."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        query = select(ValidatorConnection)
        if connected_only:
            query = query.where(ValidatorConnection.is_connected)
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        validators = result.scalars().all()

        return [ValidatorInfoResponse.model_validate(v) for v in validators]


@app.get("/validators/{validator_hotkey}", response_model=ValidatorInfoResponse)
async def get_validator(validator_hotkey: str):
    """Get a specific validator by hotkey."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(ValidatorConnection).where(
                ValidatorConnection.validator_hotkey == validator_hotkey
            )
        )
        validator = result.scalar_one_or_none()

        if not validator:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Validator not found"
            )

        return ValidatorInfoResponse.model_validate(validator)


# Submission endpoints
@app.get("/submissions", response_model=List[MinerSubmissionResponse])
async def list_submissions(
    competition_id: Optional[str] = Query(None, description="Filter by competition ID"),
    miner_hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    sort_by: Literal[
        "created_at",
        "submission_time",
        "commitment_block",
        "version",
    ] = Query(
        "created_at",
        description="Primary field to sort by",
    ),
    sort_direction: Literal["asc", "desc"] = Query(
        "desc",
        description="Sort direction for the primary field",
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
    include_status: bool = Query(
        False,
        description="When true, include aggregated evaluation status for each submission",
    ),
):
    """List miner submissions."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        sort_column_map = {
            "created_at": MinerSubmission.created_at,
            "submission_time": MinerSubmission.submission_time,
            "commitment_block": MinerSubmission.commitment_block,
            "version": MinerSubmission.version,
        }

        query = select(MinerSubmission)

        if competition_id:
            query = query.where(MinerSubmission.competition_id == competition_id)
        if miner_hotkey:
            query = query.where(MinerSubmission.miner_hotkey == miner_hotkey)

        primary_column = sort_column_map[sort_by]
        primary_order = (
            primary_column.desc() if sort_direction == "desc" else primary_column.asc()
        )

        query = query.order_by(primary_order, MinerSubmission.id.desc())
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        submissions = result.scalars().all()

        status_map: Dict[int, Optional[EvaluationStatus]] = {}
        if include_status:
            status_map = await _get_submission_evaluation_statuses(
                session, [int(s.id) for s in submissions]
            )

        responses: List[MinerSubmissionResponse] = []
        for submission in submissions:
            response = MinerSubmissionResponse.model_validate(submission)
            if include_status:
                response.evaluation_status = status_map.get(int(submission.id))
            responses.append(response)

        return responses


@app.get("/submissions/revealed", response_model=List[RevealedSubmissionResponse])
async def list_revealed_submissions(
    competition_id: Optional[str] = Query(None, description="Filter by competition ID"),
    miner_hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    only_active_urls: bool = Query(
        False,
        description="When true, return only entries whose public URL has not expired",
    ),
    sort_by: Literal[
        "released_at",
        "holdout_release_at",
        "created_at",
    ] = Query(
        "released_at",
        description="Primary field to sort revealed submissions by",
    ),
    sort_direction: Literal["asc", "desc"] = Query(
        "desc",
        description="Sort direction for the primary field",
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
    include_status: bool = Query(
        False,
        description="When true, include aggregated evaluation status for each submission",
    ),
):
    """List submissions whose hold-out period has ended."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        sort_column_map = {
            "released_at": MinerSubmission.released_at,
            "holdout_release_at": MinerSubmission.holdout_release_at,
            "created_at": MinerSubmission.created_at,
        }

        query = select(MinerSubmission).where(
            MinerSubmission.released_at.is_not(None),
            MinerSubmission.public_artifact_url.is_not(None),
        )

        if competition_id:
            query = query.where(MinerSubmission.competition_id == competition_id)
        if miner_hotkey:
            query = query.where(MinerSubmission.miner_hotkey == miner_hotkey)
        if only_active_urls:
            now = datetime.now(timezone.utc)
            query = query.where(
                MinerSubmission.public_artifact_url_expires_at.is_not(None),
                MinerSubmission.public_artifact_url_expires_at > now,
            )

        primary_column = sort_column_map[sort_by]
        primary_order = (
            primary_column.desc() if sort_direction == "desc" else primary_column.asc()
        )

        query = query.order_by(primary_order, MinerSubmission.id.desc())
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        submissions = result.scalars().all()

        status_map: Dict[int, Optional[EvaluationStatus]] = {}
        if include_status:
            status_map = await _get_submission_evaluation_statuses(
                session, [int(s.id) for s in submissions]
            )

        responses: List[RevealedSubmissionResponse] = []
        for submission in submissions:
            response = RevealedSubmissionResponse.model_validate(submission)
            if include_status:
                response.evaluation_status = status_map.get(int(submission.id))
            responses.append(response)

        return responses


@app.post(
    "/submissions/{submission_id}/rerun",
    response_model=List[JobResponse],
)
@admin_route
async def rerun_submission_evaluations(
    submission_id: int,
    request: Request,
    parameters: Optional[SubmissionRerunRequest] = Body(
        default=None,
        description="Optional benchmark filters to restrict rerun scope",
    ),
):
    """Re-run evaluations for a submission (admin only)."""

    params = parameters or SubmissionRerunRequest()
    admin_user = get_admin_user(request)

    try:
        jobs = await backend_service.rerun_submission_evaluations(
            submission_id,
            benchmark_names=params.benchmarks,
            requested_by_api_key_id=getattr(admin_user, "id", None),
        )
    except SubmissionNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except NoBenchmarksAvailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return [JobResponse.model_validate(job) for job in jobs]


@app.get(
    "/submissions/revealed/{submission_id}",
    response_model=RevealedSubmissionResponse,
)
async def get_revealed_submission(
    submission_id: int,
    include_status: bool = Query(
        False,
        description="When true, include aggregated evaluation status for the submission",
    ),
):
    """Get a specific revealed submission by ID."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        result = await session.execute(
            select(MinerSubmission).where(MinerSubmission.id == submission_id)
        )
        submission = result.scalar_one_or_none()

        if (
            not submission
            or submission.released_at is None
            or submission.public_artifact_url is None
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Revealed submission not found",
            )

        response = RevealedSubmissionResponse.model_validate(submission)
        if include_status:
            status_map = await _get_submission_evaluation_statuses(
                session, [int(submission.id)]
            )
            response.evaluation_status = status_map.get(int(submission.id))

        return response


@app.get("/submissions/{submission_id}", response_model=MinerSubmissionResponse)
async def get_submission(
    submission_id: int,
    include_status: bool = Query(
        False,
        description="When true, include aggregated evaluation status for the submission",
    ),
):
    """Get a specific submission by ID."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(MinerSubmission).where(MinerSubmission.id == submission_id)
        )
        submission = result.scalar_one_or_none()

        if not submission:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Submission not found"
            )

        response = MinerSubmissionResponse.model_validate(submission)
        if include_status:
            status_map = await _get_submission_evaluation_statuses(
                session, [int(submission.id)]
            )
            response.evaluation_status = status_map.get(int(submission.id))

        return response


# Job endpoints
@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    competition_id: Optional[str] = Query(None, description="Filter by competition ID"),
    miner_hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
):
    """List evaluation jobs."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        query = select(BackendEvaluationJob)

        if competition_id:
            query = query.where(BackendEvaluationJob.competition_id == competition_id)
        if miner_hotkey:
            query = query.where(BackendEvaluationJob.miner_hotkey == miner_hotkey)

        query = query.order_by(BackendEvaluationJob.created_at.desc())
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        jobs = result.scalars().all()

        return [JobResponse.model_validate(j) for j in jobs]


@app.post("/jobs/{job_id}/rerun", response_model=JobResponse)
@admin_route
async def rerun_job(job_id: SnowflakeId, request: Request) -> JobResponse:
    """Re-run a specific evaluation job (admin only)."""

    admin_user = get_admin_user(request)

    try:
        job = await backend_service.rerun_job_evaluation(
            int(job_id),
            requested_by_api_key_id=getattr(admin_user, "id", None),
        )
    except EvaluationJobNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return JobResponse.model_validate(job)


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: SnowflakeId):
    """Get a specific job by ID."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(BackendEvaluationJob).where(BackendEvaluationJob.id == job_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
            )

        return JobResponse.model_validate(job)


@app.get("/jobs/{job_id}/status", response_model=List[JobStatusResponse])
async def get_job_status(job_id: int):
    """Get status updates for a specific job."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        # Check if job exists
        job_result = await session.execute(
            select(BackendEvaluationJob).where(BackendEvaluationJob.id == job_id)
        )
        job = job_result.scalar_one_or_none()

        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Job not found"
            )

        # Get status updates
        status_result = await session.execute(
            select(BackendEvaluationJobStatus)
            .where(BackendEvaluationJobStatus.job_id == job_id)
            .order_by(BackendEvaluationJobStatus.created_at.desc())
        )
        status_updates = status_result.scalars().all()

        return [JobStatusResponse.model_validate(status) for status in status_updates]


@app.get("/job-status", response_model=List[JobStatusResponse])
async def list_job_status(
    job_id: Optional[int] = Query(None, description="Filter by job ID"),
    validator_hotkey: Optional[str] = Query(
        None, description="Filter by validator hotkey"
    ),
    status: Optional[str] = Query(None, description="Filter by status"),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
):
    """Get job status updates with optional filters."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        query = select(BackendEvaluationJobStatus)

        # Apply filters
        if job_id is not None:
            query = query.where(BackendEvaluationJobStatus.job_id == job_id)
        if validator_hotkey is not None:
            query = query.where(
                BackendEvaluationJobStatus.validator_hotkey == validator_hotkey
            )
        if status is not None:
            # Convert string to enum
            try:
                status_enum = EvaluationStatus(status.upper())
                query = query.where(BackendEvaluationJobStatus.status == status_enum)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid status: {status}",
                )

        # Apply pagination and ordering
        query = (
            query.order_by(BackendEvaluationJobStatus.created_at.desc())
            .limit(limit)
            .offset(offset)
        )

        result = await session.execute(query)
        status_updates = result.scalars().all()

        return [JobStatusResponse.model_validate(status) for status in status_updates]


# Result endpoints
@app.get("/results", response_model=List[EvaluationResultResponse])
async def list_results(
    job_id: Optional[int] = Query(None, description="Filter by job ID"),
    competition_id: Optional[str] = Query(None, description="Filter by competition ID"),
    miner_hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    validator_hotkey: Optional[str] = Query(
        None, description="Filter by validator hotkey"
    ),
    benchmark: Optional[str] = Query(None, description="Filter by benchmark"),
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
):
    """List evaluation results."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        query = select(BackendEvaluationResult)

        if job_id:
            query = query.where(BackendEvaluationResult.job_id == job_id)
        if competition_id:
            query = query.where(
                BackendEvaluationResult.competition_id == competition_id
            )
        if miner_hotkey:
            query = query.where(BackendEvaluationResult.miner_hotkey == miner_hotkey)
        if validator_hotkey:
            query = query.where(
                BackendEvaluationResult.validator_hotkey == validator_hotkey
            )
        if benchmark:
            query = query.where(BackendEvaluationResult.benchmark == benchmark)

        query = query.order_by(BackendEvaluationResult.result_time.desc())
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        results = result.scalars().all()

        return [EvaluationResultResponse.model_validate(r) for r in results]


@app.get("/results/{result_id}", response_model=EvaluationResultResponse)
async def get_result(result_id: int):
    """Get a specific result by ID."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(BackendEvaluationResult).where(
                BackendEvaluationResult.id == result_id
            )
        )
        eval_result = result.scalar_one_or_none()

        if not eval_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Result not found"
            )

        return EvaluationResultResponse.model_validate(eval_result)


async def _resolve_log_response(
    eval_result: BackendEvaluationResult,
) -> EvaluationResultLogResponse:
    """Internal helper to build the log response payload."""

    extra_data = eval_result.extra_data or {}
    summary = extra_data.get("summary")
    pod_logs = extra_data.get("pod_logs")
    log_artifact = extra_data.get("log_artifact")

    def _sanitize_artifact_metadata(data: Dict[str, Any]) -> Dict[str, Any]:
        allowed_keys = {"object_key", "uploaded_at", "public_url"}
        return {key: data[key] for key in allowed_keys if data.get(key) is not None}

    def _build_download_info(
        artifact: Dict[str, Any],
    ) -> Optional[EvaluationLogDownloadResponse]:
        public_url = artifact.get("public_url")
        if public_url:
            return EvaluationLogDownloadResponse(url=public_url)

        object_key = artifact.get("object_key")
        if not object_key:
            return None

        if not backend_service.submission_storage:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Log storage not configured; cannot generate download URL",
            )

        try:
            download_url, expires_at = (
                backend_service.submission_storage.generate_download_url(
                    object_key,
                    backend_service.submission_download_url_ttl,
                )
            )
            return EvaluationLogDownloadResponse(
                url=download_url, expires_at=expires_at
            )
        except Exception as exc:  # pragma: no cover - S3 failure path
            logger.exception(
                "Failed to generate log download URL for result %s",
                eval_result.id,
            )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail="Failed to generate log download URL",
            ) from exc

    artifact_metadata: Optional[Dict[str, Any]] = None
    download_info: Optional[EvaluationLogDownloadResponse] = None
    if isinstance(log_artifact, dict):
        artifact_metadata = _sanitize_artifact_metadata(log_artifact)
        download_info = _build_download_info(log_artifact)

    if not (log_artifact or pod_logs or summary):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No evaluator logs available for this result",
        )

    return EvaluationResultLogResponse(
        result_id=eval_result.id,
        job_id=eval_result.job_id,
        summary=summary,
        download=download_info,
        artifact_metadata=artifact_metadata,
        inline_logs=pod_logs,
    )


@app.get("/results/{result_id}/logs", response_model=EvaluationResultLogResponse)
async def get_result_logs(result_id: int):
    """Retrieve evaluator log metadata and download information for a result."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        result = await session.execute(
            select(BackendEvaluationResult).where(
                BackendEvaluationResult.id == result_id
            )
        )
        eval_result = result.scalar_one_or_none()

        if not eval_result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Result not found"
            )

        return await _resolve_log_response(eval_result)


@app.get("/jobs/{job_id}/logs", response_model=List[EvaluationResultLogResponse])
async def get_job_logs(job_id: int):
    """Retrieve evaluator logs for all results associated with a job."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        results = await session.execute(
            select(BackendEvaluationResult)
            .where(BackendEvaluationResult.job_id == job_id)
            .order_by(BackendEvaluationResult.result_time.desc())
        )
        eval_results = results.scalars().all()

        if not eval_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No evaluation results found for this job",
            )

        responses: List[EvaluationResultLogResponse] = []
        for result_row in eval_results:
            try:
                responses.append(await _resolve_log_response(result_row))
            except HTTPException as exc:
                if exc.status_code == status.HTTP_404_NOT_FOUND:
                    continue
                raise

        if not responses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No evaluator logs available for this job",
            )

        return responses


@app.get(
    "/submissions/{submission_id}/logs",
    response_model=List[EvaluationResultLogResponse],
)
async def get_submission_logs(submission_id: int):
    """Retrieve evaluator logs for all jobs (and their results) tied to a submission."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        jobs = await session.execute(
            select(BackendEvaluationJob.id).where(
                BackendEvaluationJob.submission_id == submission_id
            )
        )
        job_ids = [row[0] for row in jobs.all()]

        if not job_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No jobs found for this submission",
            )

        results = await session.execute(
            select(BackendEvaluationResult)
            .where(BackendEvaluationResult.job_id.in_(job_ids))
            .order_by(
                BackendEvaluationResult.job_id,
                BackendEvaluationResult.result_time.desc(),
            )
        )
        eval_results = results.scalars().all()

        if not eval_results:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No evaluation results found for this submission",
            )

        responses: List[EvaluationResultLogResponse] = []
        for result_row in eval_results:
            try:
                responses.append(await _resolve_log_response(result_row))
            except HTTPException as exc:
                if exc.status_code == status.HTTP_404_NOT_FOUND:
                    continue
                raise

        if not responses:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No evaluator logs available for this submission",
            )

        return responses


# ============================================================================
# Episode Data API Endpoints
# ============================================================================


class EpisodeDataResponse(BaseModel):
    """Response model for episode data."""

    id: str
    job_id: str
    submission_id: str
    validator_hotkey: Optional[str]
    episode_id: str
    env_name: str
    benchmark_name: str
    final_reward: float
    success: bool
    steps: int
    start_time: datetime
    end_time: datetime
    extra_metrics: Optional[dict]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @field_validator("id", "job_id", "episode_id", mode="before")
    @classmethod
    def _convert_ids(cls, value):
        return str(value)


class EpisodeStepDataResponse(BaseModel):
    """Response model for episode step data."""

    id: str
    episode_id: str
    submission_id: str
    validator_hotkey: Optional[str]
    step: int
    action: dict
    reward: float
    done: bool
    truncated: bool
    observation_refs: dict
    info: Optional[dict]
    timestamp: datetime
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True

    @field_validator("id", "episode_id", mode="before")
    @classmethod
    def _convert_ids(cls, value):
        return str(value)


@app.get("/episodes", response_model=List[EpisodeDataResponse])
async def get_episodes(
    job_id: Optional[int] = Query(None, description="Filter by job ID"),
    submission_id: Optional[str] = Query(None, description="Filter by submission ID"),
    validator_hotkey: Optional[str] = Query(
        None, description="Filter by validator hotkey"
    ),
    success: Optional[bool] = Query(None, description="Filter by success status"),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
):
    """Get episode data with optional filters."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        query = select(EpisodeData)

        # Apply filters
        if job_id is not None:
            query = query.where(EpisodeData.job_id == job_id)
        if submission_id is not None:
            query = query.where(EpisodeData.submission_id == submission_id)
        if validator_hotkey is not None:
            query = query.where(EpisodeData.validator_hotkey == validator_hotkey)
        if success is not None:
            query = query.where(EpisodeData.success == success)

        # Apply pagination
        query = (
            query.limit(limit).offset(offset).order_by(EpisodeData.created_at.desc())
        )

        result = await session.execute(query)
        episodes = result.scalars().all()

        return [EpisodeDataResponse.model_validate(ep) for ep in episodes]


@app.get("/episodes/{episode_id}", response_model=EpisodeDataResponse)
async def get_episode(episode_id: int):
    """Get a specific episode by ID."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        result = await session.execute(
            select(EpisodeData).where(EpisodeData.id == episode_id)
        )
        episode = result.scalar_one_or_none()

        if not episode:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Episode not found"
            )

        return EpisodeDataResponse.model_validate(episode)


@app.get("/episodes/{episode_id}/steps", response_model=List[EpisodeStepDataResponse])
async def get_episode_steps(
    episode_id: int,
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
):
    """Get all steps for a specific episode."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        # Check episode exists
        episode_result = await session.execute(
            select(EpisodeData).where(EpisodeData.id == episode_id)
        )
        if not episode_result.scalar_one_or_none():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Episode not found"
            )

        # Get steps
        query = (
            select(EpisodeStepData)
            .where(EpisodeStepData.episode_id == episode_id)
            .order_by(EpisodeStepData.step)
            .limit(limit)
            .offset(offset)
        )

        result = await session.execute(query)
        steps = result.scalars().all()

        return [EpisodeStepDataResponse.model_validate(step) for step in steps]


@app.get("/steps", response_model=List[EpisodeStepDataResponse])
async def get_steps(
    submission_id: Optional[str] = Query(None, description="Filter by submission ID"),
    validator_hotkey: Optional[str] = Query(
        None, description="Filter by validator hotkey"
    ),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
):
    """Get episode steps with optional filters."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        query = select(EpisodeStepData)

        # Apply filters
        if submission_id is not None:
            query = query.where(EpisodeStepData.submission_id == submission_id)
        if validator_hotkey is not None:
            query = query.where(EpisodeStepData.validator_hotkey == validator_hotkey)

        # Apply pagination
        query = query.limit(limit).offset(offset).order_by(EpisodeStepData.step)

        result = await session.execute(query)
        steps = result.scalars().all()

        return [EpisodeStepDataResponse.model_validate(step) for step in steps]


# ============================================================================
# API Key Management Endpoints
# ============================================================================


@app.post("/admin/api-keys", response_model=ApiKeyCreateResponse)
@admin_route
async def create_api_key(key_request: ApiKeyCreateRequest):
    """Create a new API key (admin only)."""
    # Validate role
    try:
        UserRole(key_request.role)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid role: {key_request.role}. Must be one of: admin, validator, viewer",
        )

    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    # Generate API key
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)

    async with backend_service.async_session() as session:
        # Create API key record
        db_api_key = ApiKey(
            id=next(backend_service.id_generator),
            name=key_request.name,
            description=key_request.description,
            key_hash=key_hash,
            role=key_request.role,
            associated_hotkey=key_request.associated_hotkey,
            expires_at=key_request.expires_at,
        )

        session.add(db_api_key)
        await session.commit()
        await session.refresh(db_api_key)

        # Return response with the actual API key (only shown once)
        return ApiKeyCreateResponse(
            id=db_api_key.id,
            name=db_api_key.name,
            description=db_api_key.description,
            role=db_api_key.role,
            associated_hotkey=db_api_key.associated_hotkey,
            is_active=db_api_key.is_active,
            expires_at=db_api_key.expires_at,
            created_at=db_api_key.created_at,
            api_key=api_key,
        )


@app.get(
    "/admin/leader-candidates",
    response_model=List[CompetitionLeaderCandidateResponse],
)
@admin_route
async def list_leader_candidates(
    competition_id: Optional[str] = Query(None, description="Filter by competition"),
    status: Optional[LeaderCandidateStatus] = Query(
        None, description="Filter by review status"
    ),
    miner_hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
    offset: int = Query(0, ge=0),
):
    """List leader candidates for review (admin only)."""

    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        stmt = select(CompetitionLeaderCandidate).order_by(
            CompetitionLeaderCandidate.created_at.desc()
        )

        if competition_id:
            stmt = stmt.where(
                CompetitionLeaderCandidate.competition_id == competition_id
            )
        if status:
            stmt = stmt.where(CompetitionLeaderCandidate.status == status)
        if miner_hotkey:
            stmt = stmt.where(CompetitionLeaderCandidate.miner_hotkey == miner_hotkey)

        stmt = stmt.offset(offset).limit(limit)

        result = await session.execute(stmt)
        candidates = result.scalars().all()

        return [
            CompetitionLeaderCandidateResponse.model_validate(candidate)
            for candidate in candidates
        ]


@app.post(
    "/admin/leader-candidates/{candidate_id}/approve",
    response_model=CompetitionLeaderCandidateResponse,
)
@admin_route
async def approve_leader_candidate(
    candidate_id: int,
    request: Request,
    decision: Optional[LeaderCandidateReviewRequest] = Body(
        default=None, description="Optional approval notes"
    ),
):
    """Approve a pending leader candidate (admin only)."""

    decision = decision or LeaderCandidateReviewRequest()

    admin_user = get_admin_user(request)

    try:
        candidate = await backend_service.approve_leader_candidate(
            candidate_id,
            admin_user.id,
            decision.reason,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except LeaderCandidateNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except LeaderCandidateAlreadyReviewedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    return CompetitionLeaderCandidateResponse.model_validate(candidate)


@app.post(
    "/admin/leader-candidates/{candidate_id}/unapprove",
    response_model=CompetitionLeaderCandidateResponse,
)
@admin_route
async def unapprove_leader_candidate(
    candidate_id: int,
    request: Request,
    decision: Optional[LeaderCandidateReviewRequest] = Body(
        default=None, description="Optional notes explaining the unapproval"
    ),
):
    """Revert an approved leader candidate back to pending (admin only)."""

    decision = decision or LeaderCandidateReviewRequest()
    admin_user = get_admin_user(request)

    try:
        candidate = await backend_service.unapprove_leader_candidate(
            candidate_id,
            admin_user.id,
            decision.reason,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except LeaderCandidateNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except LeaderCandidateNotApprovedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    return CompetitionLeaderCandidateResponse.model_validate(candidate)


@app.post(
    "/admin/leader-candidates/{candidate_id}/reject",
    response_model=CompetitionLeaderCandidateResponse,
)
@admin_route
async def reject_leader_candidate(
    candidate_id: int,
    request: Request,
    decision: Optional[LeaderCandidateReviewRequest] = Body(
        default=None, description="Optional rejection notes"
    ),
):
    """Reject a pending leader candidate (admin only)."""

    decision = decision or LeaderCandidateReviewRequest()

    admin_user = get_admin_user(request)

    try:
        candidate = await backend_service.reject_leader_candidate(
            candidate_id,
            admin_user.id,
            decision.reason,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc
    except LeaderCandidateNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)
        ) from exc
    except LeaderCandidateAlreadyReviewedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail=str(exc)
        ) from exc

    return CompetitionLeaderCandidateResponse.model_validate(candidate)


@app.get("/admin/api-keys", response_model=List[ApiKeyResponse])
@admin_route
async def list_api_keys(
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
):
    """List all API keys (admin only)."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        query = (
            select(ApiKey).offset(skip).limit(limit).order_by(ApiKey.created_at.desc())
        )
        result = await session.execute(query)
        api_keys = result.scalars().all()

        return [ApiKeyResponse.model_validate(key) for key in api_keys]


@app.get("/admin/api-keys/{key_id}", response_model=ApiKeyResponse)
@admin_route
async def get_api_key(key_id: int):
    """Get a specific API key by ID (admin only)."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
            )

        return ApiKeyResponse.model_validate(api_key)


@app.patch("/admin/api-keys/{key_id}/activate")
@admin_route
async def activate_api_key(key_id: int):
    """Activate an API key (admin only)."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
            )

        api_key.is_active = True
        await session.commit()

        return {"status": "activated", "key_id": key_id}


@app.patch("/admin/api-keys/{key_id}/deactivate")
@admin_route
async def deactivate_api_key(key_id: int):
    """Deactivate an API key (admin only)."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
            )

        api_key.is_active = False
        await session.commit()

        return {"status": "deactivated", "key_id": key_id}


@app.delete("/admin/api-keys/{key_id}")
@admin_route
async def delete_api_key(key_id: int):
    """Delete an API key (admin only)."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )

    async with backend_service.async_session() as session:
        result = await session.execute(select(ApiKey).where(ApiKey.id == key_id))
        api_key = result.scalar_one_or_none()

        if not api_key:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="API key not found"
            )

        await session.delete(api_key)
        await session.commit()

        return {"status": "deleted", "key_id": key_id}


# ============================================================================
# WebSocket Endpoint for Validators
# ============================================================================


@app.websocket("/ws/validator")
async def validator_websocket(websocket: WebSocket):
    """WebSocket endpoint for validator connections."""
    await websocket.accept()
    # generate a connection id
    connection_id = uuid.uuid4().hex

    try:
        # Wait for registration
        data = await websocket.receive_text()
        message = json.loads(data)

        if message.get("message_type") != MessageType.REGISTER:
            await websocket.send_text(json.dumps({"error": "Must register first"}))
            await websocket.close()
            return

        # Check for API key in registration message
        api_key = message.get("api_key")
        if not api_key:
            await websocket.send_text(json.dumps({"error": "Missing API key"}))
            await websocket.close()
            return

        # Validate API key
        api_key_obj = await get_api_key_from_db(api_key, backend_service)
        if not api_key_obj:
            await websocket.send_text(
                json.dumps({"error": "Invalid, expired, or inactive API key"})
            )
            await websocket.close()
            return

        # Check if API key has validator role
        if (
            api_key_obj.role != UserRole.VALIDATOR
            and api_key_obj.role != UserRole.ADMIN
        ):
            await websocket.send_text(
                json.dumps(
                    {"error": "API key does not have access to validator endpoints"}
                )
            )
            await websocket.close()
            return

        validator_hotkey = message.get("hotkey")
        if not validator_hotkey:
            await websocket.send_text(json.dumps({"error": "Missing hotkey"}))
            await websocket.close()
            return

        # If API key has an associated hotkey, verify it matches
        if (
            api_key_obj.associated_hotkey
            and api_key_obj.associated_hotkey != validator_hotkey
        ):
            await websocket.send_text(
                json.dumps({"error": "Hotkey does not match API key association"})
            )
            await websocket.close()
            return

        # Register validator
        if not backend_service.async_session:
            await websocket.send_text(json.dumps({"error": "Database not initialized"}))
            await websocket.close()
            return
        async with backend_service.async_session() as session:
            result = await session.execute(
                select(ValidatorConnection).where(
                    ValidatorConnection.validator_hotkey == validator_hotkey
                )
            )
            validator_conn = result.scalar_one_or_none()

            if not validator_conn:
                validator_conn = ValidatorConnection(
                    id=next(backend_service.id_generator),
                    validator_hotkey=validator_hotkey,
                    connection_id=connection_id,
                    api_key_id=api_key_obj.id,
                    is_connected=True,
                )
                session.add(validator_conn)
            else:
                validator_conn.connection_id = connection_id
                validator_conn.api_key_id = api_key_obj.id
                validator_conn.last_connected_at = datetime.now(timezone.utc)
                validator_conn.last_heartbeat = datetime.now(timezone.utc)
                validator_conn.is_connected = True

            await session.commit()

        # Store connection
        backend_service.active_connections[connection_id] = websocket
        backend_service.validator_connections[connection_id] = validator_hotkey

        # Send acknowledgment
        await websocket.send_text(
            json.dumps(
                {
                    "message_type": MessageType.REGISTRATION_ACK,
                    "status": "registered",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        )

        logger.info(f"Validator registered: {validator_hotkey} ({connection_id})")

        # Broadcast validator connected event
        event = ValidatorConnectedEvent(
            validator_hotkey=validator_hotkey,
            connection_id=connection_id,
            connected_at=datetime.now(timezone.utc),
        )
        await event_broadcaster.broadcast_event(EventType.VALIDATOR_CONNECTED, event)

        # Broadcast updated stats (validator count changed)
        await backend_service._broadcast_stats_update()

        # Handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            message_type = message.get("message_type")

            if message_type == MessageType.HEARTBEAT:
                await backend_service.queue_validator_heartbeat(
                    validator_hotkey, datetime.now(timezone.utc)
                )

                await websocket.send_text(
                    json.dumps(
                        {
                            "message_type": MessageType.HEARTBEAT_ACK,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                )

            elif message_type == MessageType.EVAL_RESULT:
                # Handle evaluation result
                result_msg = EvalResultMessage(**message)

                if not backend_service.async_session:
                    logger.error("Database not initialized")
                    continue
                async with backend_service.async_session() as session:
                    # Find job
                    job_result = await session.execute(
                        select(BackendEvaluationJob).where(
                            BackendEvaluationJob.id == result_msg.job_id
                        )
                    )
                    backend_job = job_result.scalar_one_or_none()

                    if backend_job:
                        # Create result
                        eval_result = BackendEvaluationResult(
                            id=next(backend_service.id_generator),
                            job_id=result_msg.job_id,
                            validator_hotkey=validator_hotkey,
                            miner_hotkey=result_msg.miner_hotkey,
                            competition_id=result_msg.competition_id,
                            env_provider=result_msg.env_provider,
                            benchmark=result_msg.benchmark_name,
                            score=result_msg.score,
                            success_rate=result_msg.success_rate,
                            avg_reward=result_msg.avg_reward,
                            total_episodes=result_msg.total_episodes,
                            logs=result_msg.logs,
                            error=result_msg.error,
                            extra_data=result_msg.extra_data,
                            env_specs=result_msg.env_specs,
                        )

                        session.add(eval_result)

                        # Update validator stats
                        val_result = await session.execute(
                            select(ValidatorConnection).where(
                                ValidatorConnection.validator_hotkey == validator_hotkey
                            )
                        )
                        validator_conn = val_result.scalar_one_or_none()
                        if validator_conn:
                            validator_conn.total_results_received += 1
                            if result_msg.error:
                                validator_conn.total_errors += 1

                        await session.commit()

                        # Update job status based on result
                        result_status = getattr(result_msg, "status", None)
                        if result_status is None:
                            result_status = (
                                EvaluationStatus.FAILED
                                if result_msg.error
                                else EvaluationStatus.COMPLETED
                            )

                        if result_status == EvaluationStatus.COMPLETED:
                            detail = (
                                f"Evaluation completed with score {result_msg.score}"
                            )
                        else:
                            detail = result_msg.error or result_status.value

                        await backend_service._update_job_status(
                            result_msg.job_id,
                            validator_hotkey,
                            result_status,
                            detail,
                        )

                        logger.info(
                            f"Stored result from {validator_hotkey} for job {result_msg.job_id}"
                        )

                        # Create evaluation completed event
                        # Pydantic will automatically handle datetime to ISO conversion
                        eval_event = EvaluationCompletedEvent(
                            job_id=eval_result.job_id,
                            validator_hotkey=eval_result.validator_hotkey,
                            miner_hotkey=eval_result.miner_hotkey,
                            competition_id=eval_result.competition_id,
                            benchmark_name=eval_result.benchmark,
                            score=eval_result.score,
                            success_rate=eval_result.success_rate,
                            avg_reward=eval_result.avg_reward,
                            total_episodes=eval_result.total_episodes,
                            result_time=eval_result.result_time,
                            created_at=eval_result.created_at,
                        )
                        await event_broadcaster.broadcast_event(
                            EventType.EVALUATION_COMPLETED, eval_event
                        )

                        # Send acknowledgment
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "message_type": MessageType.RESULT_ACK,
                                    "job_id": str(result_msg.job_id),
                                    "status": "received",
                                }
                            )
                        )

            elif message_type == MessageType.JOB_STATUS_UPDATE:
                status_msg = JobStatusUpdateMessage(**message)

                if status_msg.validator_hotkey != validator_hotkey:
                    logger.warning(
                        "Validator %s attempted to update job %s with mismatched hotkey %s",
                        validator_hotkey,
                        status_msg.job_id,
                        status_msg.validator_hotkey,
                    )
                    continue

                logger.info(
                    "Received job status update for job %s from %s: %s",
                    status_msg.job_id,
                    validator_hotkey,
                    status_msg.status,
                )

                await backend_service._update_job_status(
                    status_msg.job_id,
                    validator_hotkey,
                    status_msg.status,
                    status_msg.detail,
                )

            elif message_type == MessageType.EPISODE_DATA:
                episode_msg = EpisodeDataMessage(**message)
                await backend_service.queue_episode_data(validator_hotkey, episode_msg)

                logger.debug(
                    "Queued episode data from %s for episode %s for submission %s",
                    validator_hotkey,
                    episode_msg.episode_id,
                    episode_msg.submission_id,
                )

            elif message_type == MessageType.EPISODE_STEP_DATA:
                step_msg = EpisodeStepDataMessage(**message)
                await backend_service.queue_episode_step_data(
                    validator_hotkey, step_msg
                )

                logger.debug(
                    "Queued step data from %s for episode %s step %s",
                    validator_hotkey,
                    step_msg.episode_id,
                    step_msg.step,
                )

    except WebSocketDisconnect:
        logger.info(f"Validator disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Error in validator WebSocket: {e}")
    finally:
        # Cleanup
        if connection_id in backend_service.active_connections:
            del backend_service.active_connections[connection_id]
        if connection_id in backend_service.validator_connections:
            hotkey = backend_service.validator_connections[connection_id]
            del backend_service.validator_connections[connection_id]

            # Update database
            if backend_service.async_session:
                async with backend_service.async_session() as session:
                    result = await session.execute(
                        select(ValidatorConnection).where(
                            ValidatorConnection.validator_hotkey == hotkey
                        )
                    )
                    validator_conn = result.scalar_one_or_none()
                    if validator_conn:
                        validator_conn.is_connected = False
                        await session.commit()

            # Broadcast validator disconnected event
            disconnected_event = ValidatorDisconnectedEvent(
                validator_hotkey=hotkey,
                connection_id=connection_id,
                disconnected_at=datetime.now(timezone.utc),
            )
            await event_broadcaster.broadcast_event(
                EventType.VALIDATOR_DISCONNECTED, disconnected_event
            )

            # Broadcast updated stats (validator count changed)
            await backend_service._broadcast_stats_update()


# ============================================================================
# WebSocket Endpoint for Frontend Clients
# ============================================================================


@app.websocket("/ws/client")
async def client_websocket(websocket: WebSocket):
    """WebSocket endpoint for frontend client connections (no auth required)."""
    await websocket.accept()
    # Generate a connection ID
    connection_id = uuid.uuid4().hex

    try:
        # Add client to broadcaster
        await event_broadcaster.add_client(connection_id, websocket)

        # Send connection acknowledgment
        await websocket.send_text(
            json.dumps(
                {
                    "message_type": MessageType.REGISTRATION_ACK,
                    "status": "connected",
                    "connection_id": connection_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        )

        logger.info(f"Client connected: {connection_id}")

        # Handle messages
        while True:
            data = await websocket.receive_text()
            await event_broadcaster.handle_client_message(connection_id, data)

    except WebSocketDisconnect:
        logger.info(f"Client disconnected: {connection_id}")
    except Exception as e:
        logger.error(f"Error in client WebSocket: {e}")
    finally:
        # Cleanup
        await event_broadcaster.remove_client(connection_id)
