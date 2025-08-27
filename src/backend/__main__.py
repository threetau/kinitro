"""
FastAPI backend application for Kinitro.

This provides REST API endpoints and WebSocket connections for:
- Competition management
- Validator connections
- Job distribution
- Result collection
"""

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from pydantic import BaseModel, Field
from snowflake import SnowflakeGenerator
from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker

from core.chain import query_commitments_from_substrate
from core.log import get_logger
from core.schemas import ChainCommitmentResponse

from .config import BackendConfig
from .models import (
    BackendEvaluationJob,
    BackendEvaluationResult,
    BackendState,
    Competition,
    MinerSubmission,
    ValidatorConnection,
)

logger = get_logger(__name__)


# Pydantic models for API requests/responses
class CompetitionCreate(BaseModel):
    """Request model for creating a competition."""

    name: str
    description: Optional[str] = None
    benchmarks: List[str]
    points: int = Field(gt=0)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None


class CompetitionResponse(BaseModel):
    """Response model for competition data."""

    id: str
    name: str
    description: Optional[str]
    benchmarks: List[str]
    points: int
    active: bool
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class ValidatorInfo(BaseModel):
    """Response model for validator information."""

    validator_hotkey: str
    connection_id: str
    is_connected: bool
    first_connected_at: datetime
    last_heartbeat: datetime
    total_jobs_sent: int
    total_results_received: int
    total_errors: int

    class Config:
        from_attributes = True


class MinerSubmissionResponse(BaseModel):
    """Response model for miner submission data."""

    id: int
    miner_hotkey: str
    competition_id: str
    hf_repo_id: str
    version: str
    commitment_block: int
    submission_time: datetime
    created_at: datetime

    class Config:
        from_attributes = True


class JobResponse(BaseModel):
    """Response model for job data."""

    id: int
    job_id: str
    submission_id: int
    competition_id: str
    miner_hotkey: str
    hf_repo_id: str
    benchmarks: List[str]
    broadcast_time: Optional[datetime]
    validators_sent: int
    validators_completed: int
    created_at: datetime

    class Config:
        from_attributes = True


class EvaluationResultResponse(BaseModel):
    """Response model for evaluation result data."""

    id: int
    job_id: str
    validator_hotkey: str
    miner_hotkey: str
    competition_id: str
    benchmark: str
    score: float
    success_rate: Optional[float]
    avg_reward: Optional[float]
    total_episodes: Optional[int]
    error: Optional[str]
    result_time: datetime

    class Config:
        from_attributes = True


class BackendStats(BaseModel):
    """Response model for backend statistics."""

    total_competitions: int
    active_competitions: int
    total_points: int
    connected_validators: int
    total_submissions: int
    total_jobs: int
    total_results: int
    last_seen_block: int
    competition_percentages: Dict[str, float]


class EvalJobMessage(BaseModel):
    """Message for broadcasting evaluation jobs to validators."""

    message_type: str = "eval_job"
    job_id: str
    competition_id: str
    miner_hotkey: str
    hf_repo_id: str
    benchmarks: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvalResultMessage(BaseModel):
    """Message for receiving evaluation results from validators."""

    message_type: str = "eval_result"
    job_id: str
    validator_hotkey: str
    miner_hotkey: str
    competition_id: str
    benchmark: str
    score: float
    success_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    total_episodes: Optional[int] = None
    logs: Optional[str] = None
    error: Optional[str] = None
    extra_data: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class BackendService:
    """Core backend service logic."""

    def __init__(self, config: BackendConfig):
        self.config = config
        self.db_url = config.settings.get("database_url")

        # Chain monitoring configuration
        self.max_commitment_lookback = config.settings.get(
            "max_commitment_lookback", 360
        )
        self.chain_sync_interval = config.settings.get("chain_sync_interval", 30)
        self.min_stake_threshold = config.settings.get("min_stake_threshold", 0.0)

        # Chain connection objects
        self.substrate = None
        self.metagraph = None

        # WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        self.validator_connections: Dict[str, str] = {}  # connection_id -> hotkey

        # Background tasks
        self._running = False
        self._chain_monitor_task = None
        self._heartbeat_monitor_task = None

        # ID generator
        self.id_generator = SnowflakeGenerator(42)

        # Database
        self.engine = None
        self.async_session = None

    async def startup(self):
        """Initialize the backend service."""
        logger.info("Starting Kinitro Backend Service")

        # Initialize chain connection
        await self._init_chain()

        # Initialize database
        await self._init_database()

        # Load backend state
        await self._load_backend_state()

        # Start background tasks
        self._running = True
        self._chain_monitor_task = asyncio.create_task(self._monitor_chain())
        self._heartbeat_monitor_task = asyncio.create_task(
            self._monitor_validator_heartbeats()
        )

        logger.info("Kinitro Backend Service started successfully")

    async def shutdown(self):
        """Shutdown the backend service."""
        logger.info("Shutting down Kinitro Backend Service")

        self._running = False

        # Cancel background tasks
        if self._chain_monitor_task:
            self._chain_monitor_task.cancel()
            try:
                await self._chain_monitor_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_monitor_task:
            self._heartbeat_monitor_task.cancel()
            try:
                await self._heartbeat_monitor_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connections
        for ws in self.active_connections.values():
            await ws.close()

        # Close database
        if self.engine:
            await self.engine.dispose()

        logger.info("Backend Service shut down")

    async def _init_chain(self):
        """Initialize blockchain connection."""
        try:
            logger.info("Initializing blockchain connection...")

            self.substrate = get_substrate(
                subtensor_network=self.config.settings["subtensor_network"],
                subtensor_address=self.config.settings["subtensor_address"],
            )

            self.metagraph = Metagraph(
                netuid=self.config.settings["netuid"],
                substrate=self.substrate,
            )

            logger.info("Blockchain connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connection: {e}")
            # Don't raise - allow backend to run without chain connection for testing

    async def _init_database(self):
        """Initialize database connection."""
        self.engine = create_async_engine(
            self.db_url, echo=False, pool_pre_ping=True, pool_size=20, max_overflow=0
        )

        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        logger.info("Database connection initialized")

    async def _load_backend_state(self):
        """Load or initialize backend state."""
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

    async def _monitor_chain(self):
        """Background task to monitor blockchain for commitments."""
        while self._running:
            try:
                if self.substrate and self.metagraph:
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

                        # Query commitments
                        for block_num in range(start_block, latest_block + 1):
                            commitments = await self._query_block_commitments(block_num)
                            for commitment in commitments:
                                await self._process_commitment(
                                    commitment, block_num, active_competitions
                                )

                        # Update state
                        state.last_seen_block = latest_block
                        state.last_chain_scan = datetime.now(timezone.utc)
                        await session.commit()

                await asyncio.sleep(self.chain_sync_interval)

            except Exception as e:
                logger.error(f"Error monitoring chain: {e}")
                await asyncio.sleep(self.chain_sync_interval)

    async def _monitor_validator_heartbeats(self):
        """Monitor validator heartbeats and cleanup stale connections."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                timeout_threshold = current_time - timedelta(minutes=2)

                async with self.async_session() as session:
                    # Find stale validators
                    result = await session.execute(
                        select(ValidatorConnection).where(
                            and_(
                                ValidatorConnection.is_connected,
                                ValidatorConnection.last_heartbeat < timeout_threshold,
                            )
                        )
                    )

                    for validator in result.scalars():
                        logger.warning(
                            f"Marking validator as disconnected: {validator.validator_hotkey}"
                        )
                        validator.is_connected = False

                        # Close WebSocket if exists
                        for conn_id, hotkey in list(self.validator_connections.items()):
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
                await asyncio.sleep(30)

    async def _get_latest_block(self) -> int:
        """Get latest block from chain."""
        try:
            if not self.substrate:
                return 0
            return self.substrate.get_block_number()
        except Exception as e:
            logger.error(f"Failed to get latest block: {e}")
            return 0

    async def _sync_metagraph(self):
        """Sync metagraph nodes."""
        try:
            if self.metagraph:
                self.metagraph.sync_nodes()
                logger.debug("Metagraph synced")
        except Exception as e:
            logger.error(f"Failed to sync metagraph: {e}")

    async def _query_block_commitments(
        self, block_num: int
    ) -> List[ChainCommitmentResponse]:
        """Query commitments for a block."""
        commitments = []

        try:
            if not self.metagraph:
                return []

            # Query each miner
            for node in self.metagraph.nodes.values():
                if node.stake >= self.min_stake_threshold:
                    try:
                        miner_commitments = query_commitments_from_substrate(
                            self.config, node.hotkey, block=block_num
                        )
                        if miner_commitments:
                            commitments.extend(miner_commitments)
                    except Exception as e:
                        logger.debug(f"Failed to query {node.hotkey}: {e}")
                        continue

        except Exception as e:
            logger.error(f"Failed to query block {block_num}: {e}")

        return commitments

    async def _process_commitment(
        self,
        commitment: ChainCommitmentResponse,
        block_num: int,
        active_competitions: dict,
    ):
        """Process a commitment from the chain."""
        try:
            competition_id = getattr(commitment.data, "competition_id", None)

            if not competition_id or competition_id not in active_competitions:
                logger.warning(f"Unknown competition {competition_id}")
                return

            competition = active_competitions[competition_id]

            async with self.async_session() as session:
                # Check if submission exists
                existing = await session.execute(
                    select(MinerSubmission).where(
                        and_(
                            MinerSubmission.miner_hotkey == commitment.hotkey,
                            MinerSubmission.competition_id == competition_id,
                            MinerSubmission.version == commitment.data.version,
                        )
                    )
                )

                if existing.scalar_one_or_none():
                    return

                # Create submission
                submission = MinerSubmission(
                    id=next(self.id_generator),
                    miner_hotkey=commitment.hotkey,
                    competition_id=competition_id,
                    hf_repo_id=commitment.data.repo_id,
                    version=commitment.data.version,
                    commitment_block=block_num,
                )

                session.add(submission)
                await session.flush()

                # Create job
                job_id = str(uuid.uuid4())
                eval_job = BackendEvaluationJob(
                    id=next(self.id_generator),
                    job_id=job_id,
                    submission_id=submission.id,
                    competition_id=competition_id,
                    miner_hotkey=submission.miner_hotkey,
                    hf_repo_id=submission.hf_repo_id,
                    benchmarks=competition.benchmarks,
                )

                session.add(eval_job)
                await session.commit()

                # Broadcast to validators
                await self._broadcast_job(eval_job)

                logger.info(f"Processed commitment from {commitment.hotkey}")

        except Exception as e:
            logger.error(f"Failed to process commitment: {e}")

    async def _broadcast_job(self, job: BackendEvaluationJob):
        """Broadcast job to connected validators."""
        if not self.active_connections:
            logger.warning("No validators connected")
            return

        job_msg = EvalJobMessage(
            job_id=job.job_id,
            competition_id=job.competition_id,
            miner_hotkey=job.miner_hotkey,
            hf_repo_id=job.hf_repo_id,
            benchmarks=job.benchmarks,
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

        # Update job stats
        async with self.async_session() as session:
            job_result = await session.execute(
                select(BackendEvaluationJob).where(BackendEvaluationJob.id == job.id)
            )
            db_job = job_result.scalar_one()
            db_job.broadcast_time = datetime.now(timezone.utc)
            db_job.validators_sent = broadcast_count
            await session.commit()

        logger.info(f"Broadcasted job {job.job_id} to {broadcast_count} validators")


# Create backend service instance
config = BackendConfig()
backend_service = BackendService(config)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage backend service lifecycle."""
    await backend_service.startup()
    yield
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
async def root():
    """Root endpoint."""
    return {
        "service": "Kinitro Backend",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "kinitro-backend",
        "chain_connected": backend_service.substrate is not None,
        "database_connected": backend_service.engine is not None,
    }


@app.get("/stats", response_model=BackendStats)
async def get_stats():
    """Get comprehensive backend statistics."""
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

        return BackendStats(
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
async def create_competition(competition: CompetitionCreate):
    """Create a new competition."""
    async with backend_service.async_session() as session:
        db_competition = Competition(
            id=str(uuid.uuid4()),
            name=competition.name,
            description=competition.description,
            benchmarks=competition.benchmarks,
            points=competition.points,
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
    limit: int = Query(100, ge=1, le=1000),
):
    """List all competitions."""
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
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        competition = result.scalar_one_or_none()

        if not competition:
            raise HTTPException(status_code=404, detail="Competition not found")

        return CompetitionResponse.model_validate(competition)


@app.patch("/competitions/{competition_id}/activate")
async def activate_competition(competition_id: str):
    """Activate a competition."""
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        competition = result.scalar_one_or_none()

        if not competition:
            raise HTTPException(status_code=404, detail="Competition not found")

        competition.active = True
        await session.commit()

        return {"status": "activated", "competition_id": competition_id}


@app.patch("/competitions/{competition_id}/deactivate")
async def deactivate_competition(competition_id: str):
    """Deactivate a competition."""
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        competition = result.scalar_one_or_none()

        if not competition:
            raise HTTPException(status_code=404, detail="Competition not found")

        competition.active = False
        await session.commit()

        return {"status": "deactivated", "competition_id": competition_id}


@app.delete("/competitions/{competition_id}")
async def delete_competition(competition_id: str):
    """Delete a competition (soft delete by deactivating)."""
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(Competition).where(Competition.id == competition_id)
        )
        competition = result.scalar_one_or_none()

        if not competition:
            raise HTTPException(status_code=404, detail="Competition not found")

        competition.active = False
        await session.commit()

        return {"status": "deleted", "competition_id": competition_id}


# Validator endpoints
@app.get("/validators", response_model=List[ValidatorInfo])
async def list_validators(
    connected_only: bool = Query(
        False, description="Filter for connected validators only"
    ),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """List all validators."""
    async with backend_service.async_session() as session:
        query = select(ValidatorConnection)
        if connected_only:
            query = query.where(ValidatorConnection.is_connected)
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        validators = result.scalars().all()

        return [ValidatorInfo.model_validate(v) for v in validators]


@app.get("/validators/{validator_hotkey}", response_model=ValidatorInfo)
async def get_validator(validator_hotkey: str):
    """Get a specific validator by hotkey."""
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(ValidatorConnection).where(
                ValidatorConnection.validator_hotkey == validator_hotkey
            )
        )
        validator = result.scalar_one_or_none()

        if not validator:
            raise HTTPException(status_code=404, detail="Validator not found")

        return ValidatorInfo.model_validate(validator)


# Submission endpoints
@app.get("/submissions", response_model=List[MinerSubmissionResponse])
async def list_submissions(
    competition_id: Optional[str] = Query(None, description="Filter by competition ID"),
    miner_hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """List miner submissions."""
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
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(MinerSubmission).where(MinerSubmission.id == submission_id)
        )
        submission = result.scalar_one_or_none()

        if not submission:
            raise HTTPException(status_code=404, detail="Submission not found")

        return MinerSubmissionResponse.model_validate(submission)


# Job endpoints
@app.get("/jobs", response_model=List[JobResponse])
async def list_jobs(
    competition_id: Optional[str] = Query(None, description="Filter by competition ID"),
    miner_hotkey: Optional[str] = Query(None, description="Filter by miner hotkey"),
    pending_only: bool = Query(False, description="Show only incomplete jobs"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
):
    """List evaluation jobs."""
    async with backend_service.async_session() as session:
        query = select(BackendEvaluationJob)

        if competition_id:
            query = query.where(BackendEvaluationJob.competition_id == competition_id)
        if miner_hotkey:
            query = query.where(BackendEvaluationJob.miner_hotkey == miner_hotkey)
        if pending_only:
            query = query.where(
                BackendEvaluationJob.validators_completed
                < BackendEvaluationJob.validators_sent
            )

        query = query.order_by(BackendEvaluationJob.created_at.desc())
        query = query.offset(skip).limit(limit)

        result = await session.execute(query)
        jobs = result.scalars().all()

        return [JobResponse.model_validate(j) for j in jobs]


@app.get("/jobs/{job_id}", response_model=JobResponse)
async def get_job(job_id: str):
    """Get a specific job by ID."""
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(BackendEvaluationJob).where(BackendEvaluationJob.job_id == job_id)
        )
        job = result.scalar_one_or_none()

        if not job:
            raise HTTPException(status_code=404, detail="Job not found")

        return JobResponse.model_validate(job)


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
    limit: int = Query(100, ge=1, le=1000),
):
    """List evaluation results."""
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
    async with backend_service.async_session() as session:
        result = await session.execute(
            select(BackendEvaluationResult).where(
                BackendEvaluationResult.id == result_id
            )
        )
        eval_result = result.scalar_one_or_none()

        if not eval_result:
            raise HTTPException(status_code=404, detail="Result not found")

        return EvaluationResultResponse.model_validate(eval_result)


# ============================================================================
# WebSocket Endpoint for Validators
# ============================================================================


@app.websocket("/ws/validator")
async def validator_websocket(websocket: WebSocket):
    """WebSocket endpoint for validator connections."""
    await websocket.accept()
    connection_id = f"{websocket.client.host}:{websocket.client.port}"

    try:
        # Wait for registration
        data = await websocket.receive_text()
        message = json.loads(data)

        if message.get("message_type") != "register":
            await websocket.send_text(json.dumps({"error": "Must register first"}))
            await websocket.close()
            return

        validator_hotkey = message.get("hotkey")
        if not validator_hotkey:
            await websocket.send_text(json.dumps({"error": "Missing hotkey"}))
            await websocket.close()
            return

        # Register validator
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
                    is_connected=True,
                )
                session.add(validator_conn)
            else:
                validator_conn.connection_id = connection_id
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
                    "message_type": "registration_ack",
                    "status": "registered",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            )
        )

        logger.info(f"Validator registered: {validator_hotkey} ({connection_id})")

        # Handle messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            message_type = message.get("message_type")

            if message_type == "heartbeat":
                # Update heartbeat
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
                            "message_type": "heartbeat_ack",
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                        }
                    )
                )

            elif message_type == "eval_result":
                # Handle evaluation result
                result_msg = EvalResultMessage(**message)

                async with backend_service.async_session() as session:
                    # Find job
                    job_result = await session.execute(
                        select(BackendEvaluationJob).where(
                            BackendEvaluationJob.job_id == result_msg.job_id
                        )
                    )
                    backend_job = job_result.scalar_one_or_none()

                    if backend_job:
                        # Create result
                        eval_result = BackendEvaluationResult(
                            id=next(backend_service.id_generator),
                            job_id=result_msg.job_id,
                            backend_job_id=backend_job.id,
                            validator_hotkey=validator_hotkey,
                            miner_hotkey=result_msg.miner_hotkey,
                            competition_id=result_msg.competition_id,
                            benchmark=result_msg.benchmark,
                            score=result_msg.score,
                            success_rate=result_msg.success_rate,
                            avg_reward=result_msg.avg_reward,
                            total_episodes=result_msg.total_episodes,
                            logs=result_msg.logs,
                            error=result_msg.error,
                            extra_data=result_msg.extra_data,
                        )

                        session.add(eval_result)

                        # Update job stats
                        backend_job.validators_completed += 1

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

                        logger.info(
                            f"Stored result from {validator_hotkey} for job {result_msg.job_id}"
                        )

                        # Send acknowledgment
                        await websocket.send_text(
                            json.dumps(
                                {
                                    "message_type": "result_ack",
                                    "job_id": result_msg.job_id,
                                    "status": "received",
                                }
                            )
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8080)
