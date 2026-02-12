"""Scores routes for viewing evaluation results."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from kinitro.api.deps import get_session, get_storage
from kinitro.backend.models import (
    EvaluationCycle,
    EvaluationCycleORM,
    MinerScore,
    MinerScoreORM,
    ScoresResponse,
)
from kinitro.backend.storage import Storage

router = APIRouter(prefix="/v1/scores", tags=["Scores"])


def _build_scores_response(
    cycle: EvaluationCycleORM, scores_orm: list[MinerScoreORM]
) -> ScoresResponse:
    """Build a ScoresResponse from a cycle ORM object and its scores."""
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


@router.get("/latest", response_model=ScoresResponse)
async def get_latest_scores(
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
):
    """Get scores from the most recent completed evaluation cycle."""
    cycle = await storage.get_latest_cycle(session, completed_only=True)
    if cycle is None:
        raise HTTPException(status_code=404, detail="No completed evaluations yet")

    scores_orm = await storage.get_scores_for_cycle(session, cycle.id)
    return _build_scores_response(cycle, scores_orm)


@router.get("/{cycle_id}", response_model=ScoresResponse)
async def get_scores_for_cycle(
    cycle_id: int,
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
):
    """Get scores for a specific evaluation cycle."""
    cycle = await storage.get_cycle(session, cycle_id)
    if cycle is None:
        raise HTTPException(status_code=404, detail=f"Cycle {cycle_id} not found")

    scores_orm = await storage.get_scores_for_cycle(session, cycle_id)
    return _build_scores_response(cycle, scores_orm)
