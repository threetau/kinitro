"""Health and status routes."""

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from kinitro.backend.models import (
    EvaluationCycle,
    HealthResponse,
    StatusResponse,
)
from kinitro.api.deps import get_session, get_storage
from kinitro.backend.storage import Storage
from kinitro.environments import get_all_environment_ids

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(session: AsyncSession = Depends(get_session)):
    """Health check endpoint."""
    try:
        await session.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception:
        db_status = "disconnected"
    return HealthResponse(
        status="ok",
        version="0.1.0",
        database=db_status,
    )


@router.get("/v1/status", response_model=StatusResponse)
async def get_status(
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
):
    """Get current backend status."""
    # Get current/latest cycles
    current_cycle = await storage.get_running_cycle(session)
    latest_cycle = await storage.get_latest_cycle(session, completed_only=True)
    total_cycles = await storage.count_cycles(session)
    total_miners = await storage.count_unique_miners(session)

    return StatusResponse(
        current_cycle=EvaluationCycle.model_validate(current_cycle) if current_cycle else None,
        last_completed_cycle=EvaluationCycle.model_validate(latest_cycle) if latest_cycle else None,
        total_cycles=total_cycles,
        total_miners_evaluated=total_miners,
        environments=get_all_environment_ids(),
        is_evaluating=current_cycle is not None,
    )
