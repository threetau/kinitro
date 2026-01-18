"""API routes for the backend service."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from kinitro.backend.models import (
    EnvironmentInfo,
    EvaluationCycle,
    HealthResponse,
    MinerInfo,
    MinerScore,
    ScoresResponse,
    StatusResponse,
    WeightsResponse,
    WeightsU16,
)
from kinitro.backend.storage import Storage
from kinitro.environments import get_all_environment_ids

router = APIRouter()

# Storage instance will be set by the app
_storage: Storage | None = None
_scheduler = None  # Will be set by app


def set_storage(storage: Storage) -> None:
    """Set the storage instance for routes."""
    global _storage
    _storage = storage


def set_scheduler(scheduler) -> None:
    """Set the scheduler instance for routes."""
    global _scheduler
    _scheduler = scheduler


async def get_session():
    """Dependency to get database session."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Database not initialized")
    async with _storage.session() as session:
        yield session


# =============================================================================
# Health & Status
# =============================================================================


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    db_status = "connected" if _storage is not None else "disconnected"
    return HealthResponse(
        status="ok",
        version="0.1.0",
        database=db_status,
    )


@router.get("/v1/status", response_model=StatusResponse)
async def get_status(session: AsyncSession = Depends(get_session)):
    """Get current backend status."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    # Get current/latest cycles
    current_cycle = await _storage.get_running_cycle(session)
    latest_cycle = await _storage.get_latest_cycle(session, completed_only=True)
    total_cycles = await _storage.count_cycles(session)
    total_miners = await _storage.count_unique_miners(session)

    return StatusResponse(
        current_cycle=EvaluationCycle.model_validate(current_cycle) if current_cycle else None,
        last_completed_cycle=EvaluationCycle.model_validate(latest_cycle) if latest_cycle else None,
        total_cycles=total_cycles,
        total_miners_evaluated=total_miners,
        environments=get_all_environment_ids(),
        is_evaluating=current_cycle is not None,
    )


# =============================================================================
# Weights
# =============================================================================


@router.get("/v1/weights/latest", response_model=WeightsResponse)
async def get_latest_weights(session: AsyncSession = Depends(get_session)):
    """
    Get the most recently computed weights.

    These weights are ready to be submitted to the chain by validators.
    """
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    weights_orm = await _storage.get_latest_weights(session)
    if weights_orm is None:
        raise HTTPException(status_code=404, detail="No weights available yet")

    # Get the associated cycle for metadata
    cycle = await _storage.get_cycle(session, weights_orm.cycle_id)

    return WeightsResponse(
        cycle_id=weights_orm.cycle_id,
        block_number=weights_orm.block_number,
        timestamp=weights_orm.created_at,
        weights={int(k): float(v) for k, v in weights_orm.weights_json.items()},
        weights_u16=WeightsU16(
            uids=weights_orm.weights_u16_json["uids"],
            values=weights_orm.weights_u16_json["values"],
        ),
        metadata={
            "n_miners_evaluated": cycle.n_miners if cycle else None,
            "n_environments": cycle.n_environments if cycle else None,
            "evaluation_duration_seconds": cycle.duration_seconds if cycle else None,
        },
    )


@router.get("/v1/weights/{block_number}", response_model=WeightsResponse)
async def get_weights_for_block(
    block_number: int,
    session: AsyncSession = Depends(get_session),
):
    """Get weights computed at a specific block."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    weights_orm = await _storage.get_weights_for_block(session, block_number)
    if weights_orm is None:
        raise HTTPException(
            status_code=404,
            detail=f"No weights found for block {block_number}",
        )

    cycle = await _storage.get_cycle(session, weights_orm.cycle_id)

    return WeightsResponse(
        cycle_id=weights_orm.cycle_id,
        block_number=weights_orm.block_number,
        timestamp=weights_orm.created_at,
        weights={int(k): float(v) for k, v in weights_orm.weights_json.items()},
        weights_u16=WeightsU16(
            uids=weights_orm.weights_u16_json["uids"],
            values=weights_orm.weights_u16_json["values"],
        ),
        metadata={
            "n_miners_evaluated": cycle.n_miners if cycle else None,
            "n_environments": cycle.n_environments if cycle else None,
            "evaluation_duration_seconds": cycle.duration_seconds if cycle else None,
        },
    )


# =============================================================================
# Scores
# =============================================================================


@router.get("/v1/scores/latest", response_model=ScoresResponse)
async def get_latest_scores(session: AsyncSession = Depends(get_session)):
    """Get scores from the most recent completed evaluation cycle."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    cycle = await _storage.get_latest_cycle(session, completed_only=True)
    if cycle is None:
        raise HTTPException(status_code=404, detail="No completed evaluations yet")

    scores_orm = await _storage.get_scores_for_cycle(session, cycle.id)

    # Convert to response format
    scores = [
        MinerScore(
            uid=s.uid,
            hotkey=s.hotkey,
            env_id=s.env_id,
            success_rate=s.success_rate,
            mean_reward=s.mean_reward,
            episodes_completed=s.episodes_completed,
            episodes_failed=s.episodes_failed,
        )
        for s in scores_orm
    ]

    # Build miner summary
    miner_summary: dict[int, dict[str, float]] = {}
    for s in scores_orm:
        if s.uid not in miner_summary:
            miner_summary[s.uid] = {}
        miner_summary[s.uid][s.env_id] = s.success_rate

    return ScoresResponse(
        cycle=EvaluationCycle.model_validate(cycle),
        scores=scores,
        miner_summary=miner_summary,
    )


@router.get("/v1/scores/{cycle_id}", response_model=ScoresResponse)
async def get_scores_for_cycle(
    cycle_id: int,
    session: AsyncSession = Depends(get_session),
):
    """Get scores for a specific evaluation cycle."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    cycle = await _storage.get_cycle(session, cycle_id)
    if cycle is None:
        raise HTTPException(status_code=404, detail=f"Cycle {cycle_id} not found")

    scores_orm = await _storage.get_scores_for_cycle(session, cycle_id)

    scores = [
        MinerScore(
            uid=s.uid,
            hotkey=s.hotkey,
            env_id=s.env_id,
            success_rate=s.success_rate,
            mean_reward=s.mean_reward,
            episodes_completed=s.episodes_completed,
            episodes_failed=s.episodes_failed,
        )
        for s in scores_orm
    ]

    miner_summary: dict[int, dict[str, float]] = {}
    for s in scores_orm:
        if s.uid not in miner_summary:
            miner_summary[s.uid] = {}
        miner_summary[s.uid][s.env_id] = s.success_rate

    return ScoresResponse(
        cycle=EvaluationCycle.model_validate(cycle),
        scores=scores,
        miner_summary=miner_summary,
    )


# =============================================================================
# Miners & Environments
# =============================================================================


@router.get("/v1/miners", response_model=list[MinerInfo])
async def list_miners(session: AsyncSession = Depends(get_session)):
    """List all miners that have been evaluated."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    # Get latest cycle's scores to get miner info
    cycle = await _storage.get_latest_cycle(session, completed_only=True)
    if cycle is None:
        return []

    scores = await _storage.get_scores_for_cycle(session, cycle.id)

    # Aggregate by miner
    miners_dict: dict[int, MinerInfo] = {}
    for s in scores:
        if s.uid not in miners_dict:
            miners_dict[s.uid] = MinerInfo(
                uid=s.uid,
                hotkey=s.hotkey,
                last_evaluated_block=cycle.block_number,
                avg_success_rate=0.0,
                environments_evaluated=[],
            )
        miners_dict[s.uid].environments_evaluated.append(s.env_id)

    # Calculate average success rate per miner
    for uid, miner in miners_dict.items():
        miner_scores = [s.success_rate for s in scores if s.uid == uid]
        if miner_scores:
            miner.avg_success_rate = sum(miner_scores) / len(miner_scores)

    return list(miners_dict.values())


@router.get("/v1/environments", response_model=list[EnvironmentInfo])
async def list_environments(session: AsyncSession = Depends(get_session)):
    """List all evaluation environments."""
    if _storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")

    env_ids = get_all_environment_ids()

    # Get latest cycle for stats
    cycle = await _storage.get_latest_cycle(session, completed_only=True)

    env_stats: dict[str, dict] = {env_id: {"count": 0, "total_sr": 0.0} for env_id in env_ids}

    if cycle:
        scores = await _storage.get_scores_for_cycle(session, cycle.id)
        for s in scores:
            if s.env_id in env_stats:
                env_stats[s.env_id]["count"] += 1
                env_stats[s.env_id]["total_sr"] += s.success_rate

    result = []
    for env_id in env_ids:
        parts = env_id.split("/")
        env_name = parts[0] if parts else env_id
        task_name = parts[1] if len(parts) > 1 else ""

        stats = env_stats[env_id]
        avg_sr = stats["total_sr"] / stats["count"] if stats["count"] > 0 else None

        result.append(
            EnvironmentInfo(
                env_id=env_id,
                env_name=env_name,
                task_name=task_name,
                n_evaluations=stats["count"],
                avg_success_rate=avg_sr,
            )
        )

    return result
