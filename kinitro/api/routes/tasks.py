"""Task pool routes for executors to fetch and submit tasks."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from kinitro.backend.models import (
    Task,
    TaskFetchRequest,
    TaskFetchResponse,
    TaskPoolStats,
    TaskSubmitRequest,
    TaskSubmitResponse,
)
from kinitro.api.deps import get_session, get_storage
from kinitro.backend.storage import Storage

router = APIRouter(prefix="/v1/tasks", tags=["Tasks"])


@router.post("/fetch", response_model=TaskFetchResponse)
async def fetch_tasks(
    request: TaskFetchRequest,
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
):
    """
    Fetch tasks from the task pool.

    Executors call this endpoint to get a batch of tasks to process.
    Tasks are atomically assigned to the requesting executor using
    SELECT FOR UPDATE SKIP LOCKED to avoid race conditions.

    Returns a batch of tasks that the executor should process and
    submit results for using POST /v1/tasks/submit.
    """
    tasks_orm = await storage.fetch_tasks(
        session=session,
        executor_id=request.executor_id,
        batch_size=request.batch_size,
        env_ids=request.env_ids,
    )

    # Get total pending count for monitoring
    total_pending = await storage.count_pending_tasks(session)

    tasks = [
        Task(
            task_uuid=t.task_uuid,
            cycle_id=t.cycle_id,
            miner_uid=t.miner_uid,
            miner_hotkey=t.miner_hotkey,
            miner_endpoint=t.miner_endpoint,
            env_id=t.env_id,
            seed=t.seed,
            status=t.status,
            created_at=t.created_at,
        )
        for t in tasks_orm
    ]

    return TaskFetchResponse(
        tasks=tasks,
        total_pending=total_pending,
    )


@router.post("/submit", response_model=TaskSubmitResponse)
async def submit_tasks(
    request: TaskSubmitRequest,
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
):
    """
    Submit results for completed tasks.

    Executors call this endpoint after processing tasks to report results.
    Results are validated to ensure the executor was assigned the task.
    """
    accepted = 0
    rejected = 0
    errors = []

    for result in request.results:
        try:
            success = await storage.submit_task_result(
                session=session,
                task_uuid=result.task_uuid,
                executor_id=request.executor_id,
                success=result.success,
                score=result.score,
                total_reward=result.total_reward,
                timesteps=result.timesteps,
                error=result.error,
            )
            if success:
                accepted += 1
            else:
                rejected += 1
                errors.append(f"Task {result.task_uuid}: not found or not assigned to executor")
        except Exception as e:
            # Rollback to clear the failed transaction state so subsequent
            # operations can proceed
            await session.rollback()
            rejected += 1
            errors.append(f"Task {result.task_uuid}: {str(e)}")

    return TaskSubmitResponse(
        accepted=accepted,
        rejected=rejected,
        errors=errors,
    )


@router.get("/stats", response_model=TaskPoolStats)
async def get_task_stats(
    cycle_id: int | None = None,
    session: AsyncSession = Depends(get_session),
    storage: Storage = Depends(get_storage),
):
    """
    Get task pool statistics.

    Returns counts of tasks by status and list of active executors.
    Optionally filter by cycle_id.
    """
    # If no cycle specified, use the current running cycle
    if cycle_id is None:
        running_cycle = await storage.get_running_cycle(session)
        cycle_id = running_cycle.id if running_cycle else None

    stats = await storage.get_task_pool_stats(session, cycle_id=cycle_id)

    return TaskPoolStats(
        total_tasks=stats["total_tasks"],
        pending_tasks=stats["pending_tasks"],
        assigned_tasks=stats["assigned_tasks"],
        completed_tasks=stats["completed_tasks"],
        failed_tasks=stats["failed_tasks"],
        active_executors=stats["active_executors"],
        current_cycle_id=stats["current_cycle_id"],
    )
