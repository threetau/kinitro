import asyncio
import json
import re
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import func, select

from backend.auth import (
    UserRole,
    create_auth_dependency,
    create_role_dependencies,
    generate_api_key,
    get_api_key_from_db,
    hash_api_key,
)
from backend.constants import (
    DEFAULT_PAGE_LIMIT,
    MAX_PAGE_LIMIT,
    MIN_PAGE_LIMIT,
    SUBMISSION_SIGNATURE_MAX_AGE_SECONDS,
)
from backend.events import (
    EpisodeCompletedEvent,
    EpisodeStepEvent,
    EvaluationCompletedEvent,
    ValidatorConnectedEvent,
    ValidatorDisconnectedEvent,
)
from backend.realtime import event_broadcaster
from backend.service import BackendService
from core import __version__ as VERSION
from core.db.models import EvaluationStatus, SnowflakeId
from core.log import get_logger
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
    CompetitionResponse,
    EpisodeData,
    EpisodeStepData,
    EvaluationResultResponse,
    JobResponse,
    JobStatusResponse,
    MinerSubmission,
    MinerSubmissionResponse,
    ValidatorConnection,
    ValidatorInfoResponse,
)

logger = get_logger(__name__)


class SubmissionUploadRequest(BaseModel):
    competition_id: str
    version: str
    artifact_sha256: str
    artifact_size_bytes: int = Field(gt=0)
    timestamp: int = Field(ge=0)
    hotkey: str
    signature: str
    holdout_seconds: Optional[int] = None

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
    holdout_seconds: int
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
    if payload.holdout_seconds is not None:
        parts.append(str(payload.holdout_seconds))
    return "|".join(parts).encode("utf-8")


# Create backend service instance
config = BackendConfig()
backend_service = BackendService(config)

# Set up authentication dependencies
get_current_user = create_auth_dependency(backend_service)
require_admin, require_validator, require_auth = create_role_dependencies(
    get_current_user
)


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


@app.get("/health")
async def health_check() -> dict:
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "kinitro-backend",
        "chain_connected": backend_service.substrate is not None,
        "database_connected": backend_service.engine is not None,
    }


@app.post(
    "/submissions/request-upload",
    response_model=SubmissionUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def request_submission_upload(
    payload: SubmissionUploadRequest,
) -> SubmissionUploadResponse:
    """Create a presigned upload slot for a miner submission."""

    if payload.holdout_seconds is not None and payload.holdout_seconds < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="holdout_seconds must be non-negative",
        )

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
            holdout_seconds=payload.holdout_seconds,
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    commit_payload = {
        "provider": ModelProvider.R2.value,
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
        holdout_seconds=upload_record.holdout_seconds,
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
async def create_competition(
    competition: CompetitionCreateRequest, admin_user: "ApiKey" = Depends(require_admin)
):
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
            active=True,
            start_time=competition.start_time,
            end_time=competition.end_time,
            submission_holdout_seconds=competition.submission_holdout_seconds,
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

        return [CompetitionResponse.model_validate(c) for c in competitions]


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

        return CompetitionResponse.model_validate(competition)


@app.patch("/competitions/{competition_id}/activate")
async def activate_competition(
    competition_id: str, admin_user: "ApiKey" = Depends(require_admin)
):
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
async def deactivate_competition(
    competition_id: str, admin_user: "ApiKey" = Depends(require_admin)
):
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
async def delete_competition(
    competition_id: str, admin_user: "ApiKey" = Depends(require_admin)
):
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
    skip: int = Query(0, ge=0),
    limit: int = Query(DEFAULT_PAGE_LIMIT, ge=MIN_PAGE_LIMIT, le=MAX_PAGE_LIMIT),
):
    """List miner submissions."""
    if not backend_service.async_session:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized",
        )
    async with backend_service.async_session() as session:
        query = select(MinerSubmission)

        if competition_id:
            query = query.where(MinerSubmission.competition_id == competition_id)
        if miner_hotkey:
            query = query.where(MinerSubmission.miner_hotkey == miner_hotkey)

        query = query.order_by(MinerSubmission.created_at.desc())
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        submissions = result.scalars().all()

        return [MinerSubmissionResponse.model_validate(s) for s in submissions]


@app.get("/submissions/{submission_id}", response_model=MinerSubmissionResponse)
async def get_submission(submission_id: int):
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

        return MinerSubmissionResponse.model_validate(submission)


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
    job_id: Optional[str] = Query(None, description="Filter by job ID"),
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
async def create_api_key(
    key_request: ApiKeyCreateRequest, admin_user: ApiKey = Depends(require_admin)
):
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


@app.get("/admin/api-keys", response_model=List[ApiKeyResponse])
async def list_api_keys(
    admin_user: ApiKey = Depends(require_admin),
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
async def get_api_key(
    key_id: int,
    admin_user: ApiKey = Depends(require_admin),
):
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
async def activate_api_key(
    key_id: int,
    admin_user: ApiKey = Depends(require_admin),
):
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
async def deactivate_api_key(
    key_id: int,
    admin_user: ApiKey = Depends(require_admin),
):
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
async def delete_api_key(
    key_id: int,
    admin_user: ApiKey = Depends(require_admin),
):
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
                # Update heartbeat
                if not backend_service.async_session:
                    logger.error("Database not initialized")
                    continue
                async with backend_service.async_session() as session:
                    result = await session.execute(
                        select(ValidatorConnection).where(
                            ValidatorConnection.validator_hotkey == validator_hotkey
                        )
                    )
                    validator_conn = result.scalar_one_or_none()
                    if validator_conn:
                        validator_conn.last_heartbeat = datetime.now(timezone.utc)
                        await session.commit()

                # Send heartbeat ack
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
                        if result_msg.error:
                            await backend_service._update_job_status(
                                result_msg.job_id,
                                validator_hotkey,
                                EvaluationStatus.FAILED,
                                f"Evaluation failed: {result_msg.error}",
                            )
                        else:
                            await backend_service._update_job_status(
                                result_msg.job_id,
                                validator_hotkey,
                                EvaluationStatus.COMPLETED,
                                f"Evaluation completed with score {result_msg.score}",
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
                # Handle episode data
                episode_msg = EpisodeDataMessage(**message)

                if not backend_service.async_session:
                    logger.error("Database not initialized")
                    continue

                async with backend_service.async_session() as session:
                    # Create episode record
                    episode_record = EpisodeData(
                        id=next(backend_service.id_generator),
                        job_id=episode_msg.job_id,
                        submission_id=episode_msg.submission_id,
                        validator_hotkey=validator_hotkey,
                        task_id=episode_msg.task_id,
                        episode_id=episode_msg.episode_id,
                        env_name=episode_msg.env_name,
                        benchmark_name=episode_msg.benchmark_name,
                        final_reward=episode_msg.final_reward,
                        success=episode_msg.success,
                        steps=episode_msg.steps,
                        start_time=datetime.fromisoformat(episode_msg.start_time)
                        if isinstance(episode_msg.start_time, str)
                        else episode_msg.start_time,
                        end_time=datetime.fromisoformat(episode_msg.end_time)
                        if isinstance(episode_msg.end_time, str)
                        else episode_msg.end_time,
                        extra_metrics=episode_msg.extra_metrics,
                    )

                    session.add(episode_record)
                    await session.commit()

                    # Broadcast episode completed event to clients using the model
                    episode_event = EpisodeCompletedEvent(
                        job_id=episode_msg.job_id,
                        submission_id=episode_msg.submission_id,
                        validator_hotkey=validator_hotkey,
                        episode_id=episode_msg.episode_id,
                        env_name=episode_msg.env_name,
                        benchmark_name=episode_msg.benchmark_name,
                        final_reward=episode_msg.final_reward,
                        success=episode_msg.success,
                        steps=episode_msg.steps,
                        start_time=episode_msg.start_time
                        if isinstance(episode_msg.start_time, datetime)
                        else datetime.fromisoformat(episode_msg.start_time),
                        end_time=episode_msg.end_time
                        if isinstance(episode_msg.end_time, datetime)
                        else datetime.fromisoformat(episode_msg.end_time),
                        extra_metrics=episode_msg.extra_metrics,
                        created_at=datetime.now(timezone.utc),
                    )
                    await event_broadcaster.broadcast_event(
                        EventType.EPISODE_COMPLETED, episode_event
                    )

                    # Check if this is the first episode for this job-validator combination
                    # If so, update status to RUNNING
                    status_result = await session.execute(
                        select(BackendEvaluationJobStatus).where(
                            BackendEvaluationJobStatus.job_id == episode_msg.job_id,
                            BackendEvaluationJobStatus.validator_hotkey
                            == validator_hotkey,
                            BackendEvaluationJobStatus.status
                            == EvaluationStatus.RUNNING,
                        )
                    )
                    existing_running_status = status_result.scalar_one_or_none()

                    if not existing_running_status:
                        # First episode data received, mark as RUNNING
                        await backend_service._update_job_status(
                            episode_msg.job_id,
                            validator_hotkey,
                            EvaluationStatus.RUNNING,
                            f"Started processing episodes (episode {episode_msg.episode_id})",
                        )

                    logger.info(
                        f"Stored episode data from {validator_hotkey} for episode {episode_msg.episode_id}"
                    )

            elif message_type == MessageType.EPISODE_STEP_DATA:
                # Handle episode step data
                step_msg = EpisodeStepDataMessage(**message)

                if not backend_service.async_session:
                    logger.error("Database not initialized")
                    continue

                async with backend_service.async_session() as session:
                    # Find the episode record
                    episode_result = await session.execute(
                        select(EpisodeData).where(
                            EpisodeData.submission_id == step_msg.submission_id,
                            EpisodeData.episode_id == step_msg.episode_id,
                            EpisodeData.task_id == step_msg.task_id,
                        )
                    )
                    episode_record = episode_result.scalar_one_or_none()

                    if episode_record:
                        # Create step record
                        step_record = EpisodeStepData(
                            id=next(backend_service.id_generator),
                            episode_id=episode_record.id,  # Use the database episode ID
                            submission_id=step_msg.submission_id,
                            validator_hotkey=validator_hotkey,
                            task_id=step_msg.task_id,
                            step=step_msg.step,
                            action=step_msg.action,
                            reward=step_msg.reward,
                            done=step_msg.done,
                            truncated=step_msg.truncated,
                            observation_refs=step_msg.observation_refs,
                            info=step_msg.info,
                            timestamp=datetime.fromisoformat(step_msg.step_timestamp)
                            if isinstance(step_msg.step_timestamp, str)
                            else step_msg.step_timestamp,
                        )

                        session.add(step_record)
                        await session.commit()

                        # Broadcast episode step event to clients
                        step_event = EpisodeStepEvent(
                            submission_id=step_msg.submission_id,
                            validator_hotkey=validator_hotkey,
                            episode_id=step_msg.episode_id,
                            step=step_msg.step,
                            action=step_msg.action,
                            reward=step_msg.reward,
                            done=step_msg.done,
                            truncated=step_msg.truncated,
                            observation_refs=step_msg.observation_refs,
                            info=step_msg.info,
                        )
                        await event_broadcaster.broadcast_event(
                            EventType.EPISODE_STEP, step_event
                        )

                        logger.info(
                            f"Stored step data from {validator_hotkey} for episode {step_msg.episode_id}, step {step_msg.step}"
                        )
                    else:
                        logger.warning(
                            f"Episode not found for step data: submission {step_msg.submission_id}, episode {step_msg.episode_id}"
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
