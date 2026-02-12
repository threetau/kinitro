"""Miners and environments routes."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from kinitro.api.deps import get_session, get_storage
from kinitro.backend.models import EnvironmentInfo, MinerInfo
from kinitro.backend.storage import Storage
from kinitro.environments import get_all_environment_ids
from kinitro.types import EnvironmentId, EnvStatsEntry, Hotkey, MinerUID

router = APIRouter(prefix="/v1", tags=["Miners & Environments"])


@router.get("/miners", response_model=list[MinerInfo])
async def list_miners(
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> list[MinerInfo]:
    """List all miners that have been evaluated."""
    # Get latest cycle's scores to get miner info
    cycle = await storage.get_latest_cycle(session, completed_only=True)
    if cycle is None:
        return []

    scores = await storage.get_scores_for_cycle(session, cycle.id)

    # Aggregate by miner
    miners_dict: dict[MinerUID, MinerInfo] = {}
    for s in scores:
        uid = MinerUID(s.uid)
        if uid not in miners_dict:
            miners_dict[uid] = MinerInfo(
                uid=uid,
                hotkey=Hotkey(s.hotkey),
                last_evaluated_block=cycle.block_number,
                avg_success_rate=0.0,
                environments_evaluated=[],
            )
        miners_dict[uid].environments_evaluated.append(EnvironmentId(s.env_id))

    # Calculate average success rate per miner
    for uid, miner in miners_dict.items():
        miner_scores = [s.success_rate for s in scores if s.uid == int(uid)]
        if miner_scores:
            miner.avg_success_rate = sum(miner_scores) / len(miner_scores)

    return list(miners_dict.values())


@router.get("/environments", response_model=list[EnvironmentInfo])
async def list_environments(
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
) -> list[EnvironmentInfo]:
    """List all evaluation environments."""
    env_ids = get_all_environment_ids()

    # Get latest cycle for stats
    cycle = await storage.get_latest_cycle(session, completed_only=True)

    env_stats: dict[EnvironmentId, EnvStatsEntry] = {
        env_id: EnvStatsEntry(count=0, total_sr=0.0) for env_id in env_ids
    }

    if cycle:
        scores = await storage.get_scores_for_cycle(session, cycle.id)
        for s in scores:
            eid = EnvironmentId(s.env_id)
            if eid in env_stats:
                env_stats[eid]["count"] += 1
                env_stats[eid]["total_sr"] += s.success_rate

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
