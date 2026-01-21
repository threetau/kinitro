"""Weights routes for validators to fetch computed weights."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from kinitro.api.deps import get_session, get_storage
from kinitro.backend.models import WeightsResponse, WeightsU16
from kinitro.backend.storage import Storage

router = APIRouter(prefix="/v1/weights", tags=["Weights"])


@router.get("/latest", response_model=WeightsResponse)
async def get_latest_weights(
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
):
    """
    Get the most recently computed weights.

    These weights are ready to be submitted to the chain by validators.
    """
    weights_orm = await storage.get_latest_weights(session)
    if weights_orm is None:
        raise HTTPException(status_code=404, detail="No weights available yet")

    # Get the associated cycle for metadata
    cycle = await storage.get_cycle(session, weights_orm.cycle_id)

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


@router.get("/{block_number}", response_model=WeightsResponse)
async def get_weights_for_block(
    block_number: int,
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
):
    """Get weights computed at a specific block."""
    weights_orm = await storage.get_weights_for_block(session, block_number)
    if weights_orm is None:
        raise HTTPException(
            status_code=404,
            detail=f"No weights found for block {block_number}",
        )

    cycle = await storage.get_cycle(session, weights_orm.cycle_id)

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
