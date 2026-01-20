"""PostgreSQL storage layer for evaluation results."""

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from datetime import datetime

import structlog
from sqlalchemy import desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from kinitro.backend.models import (
    Base,
    ComputedWeightsORM,
    EvaluationCycleORM,
    EvaluationCycleStatus,
    MinerScoreORM,
    TaskPoolORM,
    TaskStatus,
)

logger = structlog.get_logger()


class Storage:
    """
    Async PostgreSQL storage for evaluation results.

    Usage:
        storage = Storage("postgresql+asyncpg://user:pass@localhost/robo")
        await storage.initialize()

        async with storage.session() as session:
            cycle = await storage.create_cycle(session, block_number=12345)
    """

    def __init__(self, database_url: str):
        """
        Initialize storage.

        Args:
            database_url: PostgreSQL connection URL
                         (e.g., postgresql+asyncpg://user:pass@localhost/robo)
        """
        self._database_url = database_url
        self._engine = create_async_engine(
            database_url,
            echo=False,
            pool_size=5,
            max_overflow=10,
        )
        self._session_factory = async_sessionmaker(
            self._engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

    async def initialize(self) -> None:
        """Create database tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("database_initialized", url=self._database_url.split("@")[-1])

    async def close(self) -> None:
        """Close database connections."""
        await self._engine.dispose()

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get a database session."""
        async with self._session_factory() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise

    # =========================================================================
    # Evaluation Cycles
    # =========================================================================

    async def create_cycle(
        self,
        session: AsyncSession,
        block_number: int,
        n_miners: int | None = None,
        n_environments: int | None = None,
    ) -> EvaluationCycleORM:
        """Create a new evaluation cycle."""
        cycle = EvaluationCycleORM(
            block_number=block_number,
            started_at=datetime.utcnow(),
            status=EvaluationCycleStatus.RUNNING.value,
            n_miners=n_miners,
            n_environments=n_environments,
        )
        session.add(cycle)
        await session.flush()
        logger.info("cycle_created", cycle_id=cycle.id, block=block_number)
        return cycle

    async def complete_cycle(
        self,
        session: AsyncSession,
        cycle_id: int,
        duration_seconds: float,
    ) -> None:
        """Mark a cycle as completed."""
        result = await session.execute(
            select(EvaluationCycleORM).where(EvaluationCycleORM.id == cycle_id)
        )
        cycle = result.scalar_one_or_none()
        if cycle:
            cycle.status = EvaluationCycleStatus.COMPLETED.value
            cycle.completed_at = datetime.utcnow()
            cycle.duration_seconds = duration_seconds
            logger.info("cycle_completed", cycle_id=cycle_id, duration=duration_seconds)

    async def fail_cycle(
        self,
        session: AsyncSession,
        cycle_id: int,
        error_message: str,
    ) -> None:
        """Mark a cycle as failed."""
        result = await session.execute(
            select(EvaluationCycleORM).where(EvaluationCycleORM.id == cycle_id)
        )
        cycle = result.scalar_one_or_none()
        if cycle:
            cycle.status = EvaluationCycleStatus.FAILED.value
            cycle.completed_at = datetime.utcnow()
            cycle.error_message = error_message
            logger.error("cycle_failed", cycle_id=cycle_id, error=error_message)

    async def get_latest_cycle(
        self,
        session: AsyncSession,
        completed_only: bool = True,
    ) -> EvaluationCycleORM | None:
        """Get the most recent evaluation cycle."""
        query = select(EvaluationCycleORM).order_by(desc(EvaluationCycleORM.id))
        if completed_only:
            query = query.where(EvaluationCycleORM.status == EvaluationCycleStatus.COMPLETED.value)
        result = await session.execute(query.limit(1))
        return result.scalar_one_or_none()

    async def get_cycle(
        self,
        session: AsyncSession,
        cycle_id: int,
    ) -> EvaluationCycleORM | None:
        """Get a specific cycle by ID."""
        result = await session.execute(
            select(EvaluationCycleORM).where(EvaluationCycleORM.id == cycle_id)
        )
        return result.scalar_one_or_none()

    async def get_running_cycle(
        self,
        session: AsyncSession,
    ) -> EvaluationCycleORM | None:
        """Get the currently running cycle, if any."""
        result = await session.execute(
            select(EvaluationCycleORM)
            .where(EvaluationCycleORM.status == EvaluationCycleStatus.RUNNING.value)
            .order_by(desc(EvaluationCycleORM.id))
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def count_cycles(self, session: AsyncSession) -> int:
        """Count total evaluation cycles."""
        result = await session.execute(select(func.count()).select_from(EvaluationCycleORM))
        return result.scalar() or 0

    # =========================================================================
    # Miner Scores
    # =========================================================================

    async def add_miner_score(
        self,
        session: AsyncSession,
        cycle_id: int,
        uid: int,
        hotkey: str,
        env_id: str,
        success_rate: float,
        mean_reward: float,
        episodes_completed: int,
        episodes_failed: int,
    ) -> MinerScoreORM:
        """Add a score for a miner on an environment."""
        score = MinerScoreORM(
            cycle_id=cycle_id,
            uid=uid,
            hotkey=hotkey,
            env_id=env_id,
            success_rate=success_rate,
            mean_reward=mean_reward,
            episodes_completed=episodes_completed,
            episodes_failed=episodes_failed,
        )
        session.add(score)
        return score

    async def add_miner_scores_bulk(
        self,
        session: AsyncSession,
        cycle_id: int,
        scores: list[dict],
    ) -> None:
        """Bulk add miner scores."""
        for score_data in scores:
            score = MinerScoreORM(cycle_id=cycle_id, **score_data)
            session.add(score)
        logger.info("scores_added", cycle_id=cycle_id, count=len(scores))

    async def get_scores_for_cycle(
        self,
        session: AsyncSession,
        cycle_id: int,
    ) -> list[MinerScoreORM]:
        """Get all scores for a cycle."""
        result = await session.execute(
            select(MinerScoreORM).where(MinerScoreORM.cycle_id == cycle_id)
        )
        return list(result.scalars().all())

    async def get_miner_history(
        self,
        session: AsyncSession,
        uid: int,
        limit: int = 10,
    ) -> list[MinerScoreORM]:
        """Get recent scores for a specific miner."""
        result = await session.execute(
            select(MinerScoreORM)
            .where(MinerScoreORM.uid == uid)
            .order_by(desc(MinerScoreORM.id))
            .limit(limit)
        )
        return list(result.scalars().all())

    async def count_unique_miners(self, session: AsyncSession) -> int:
        """Count unique miners that have been evaluated."""
        result = await session.execute(select(func.count(func.distinct(MinerScoreORM.uid))))
        return result.scalar() or 0

    # =========================================================================
    # Computed Weights
    # =========================================================================

    async def save_weights(
        self,
        session: AsyncSession,
        cycle_id: int,
        block_number: int,
        weights: dict[int, float],
        weights_u16: dict[str, list[int]],
    ) -> ComputedWeightsORM:
        """Save computed weights for a cycle."""
        weights_orm = ComputedWeightsORM(
            cycle_id=cycle_id,
            block_number=block_number,
            weights_json=weights,
            weights_u16_json=weights_u16,
            created_at=datetime.utcnow(),
        )
        session.add(weights_orm)
        logger.info("weights_saved", cycle_id=cycle_id, block=block_number, n_miners=len(weights))
        return weights_orm

    async def get_latest_weights(
        self,
        session: AsyncSession,
    ) -> ComputedWeightsORM | None:
        """Get the most recently computed weights."""
        result = await session.execute(
            select(ComputedWeightsORM).order_by(desc(ComputedWeightsORM.block_number)).limit(1)
        )
        return result.scalar_one_or_none()

    async def get_weights_for_block(
        self,
        session: AsyncSession,
        block_number: int,
    ) -> ComputedWeightsORM | None:
        """Get weights computed at a specific block."""
        result = await session.execute(
            select(ComputedWeightsORM)
            .where(ComputedWeightsORM.block_number == block_number)
            .limit(1)
        )
        return result.scalar_one_or_none()

    # =========================================================================
    # Task Pool
    # =========================================================================

    async def create_task(
        self,
        session: AsyncSession,
        cycle_id: int,
        miner_uid: int,
        miner_hotkey: str,
        miner_endpoint: str,
        env_id: str,
        seed: int,
        task_uuid: str | None = None,
    ) -> TaskPoolORM:
        """Create a new task in the task pool."""
        # Build kwargs, only including task_uuid if provided
        # (passing None explicitly would violate NOT NULL constraint)
        task_kwargs = {
            "cycle_id": cycle_id,
            "miner_uid": miner_uid,
            "miner_hotkey": miner_hotkey,
            "miner_endpoint": miner_endpoint,
            "env_id": env_id,
            "seed": seed,
            "status": TaskStatus.PENDING.value,
        }
        if task_uuid is not None:
            task_kwargs["task_uuid"] = task_uuid
        task = TaskPoolORM(**task_kwargs)
        session.add(task)
        await session.flush()
        return task

    async def create_tasks_bulk(
        self,
        session: AsyncSession,
        tasks: list[dict],
    ) -> int:
        """Bulk create tasks in the task pool.

        Args:
            session: Database session
            tasks: List of task dicts with keys: task_uuid, cycle_id, miner_uid,
                   miner_hotkey, miner_endpoint, env_id, seed

        Returns:
            Number of tasks created
        """
        for task_data in tasks:
            task = TaskPoolORM(
                task_uuid=task_data["task_uuid"],
                cycle_id=task_data["cycle_id"],
                miner_uid=task_data["miner_uid"],
                miner_hotkey=task_data["miner_hotkey"],
                miner_endpoint=task_data["miner_endpoint"],
                env_id=task_data["env_id"],
                seed=task_data["seed"],
                status=TaskStatus.PENDING.value,
            )
            session.add(task)
        logger.info("tasks_created_bulk", count=len(tasks))
        return len(tasks)

    async def fetch_tasks(
        self,
        session: AsyncSession,
        executor_id: str,
        batch_size: int = 10,
        env_ids: list[str] | None = None,
    ) -> list[TaskPoolORM]:
        """Fetch and assign tasks to an executor.

        Uses SELECT FOR UPDATE SKIP LOCKED to ensure atomic assignment
        and avoid race conditions between executors.

        Args:
            session: Database session
            executor_id: ID of the executor fetching tasks
            batch_size: Maximum number of tasks to fetch
            env_ids: Optional filter by environment IDs

        Returns:
            List of assigned tasks
        """
        # Build query for pending tasks
        query = (
            select(TaskPoolORM)
            .where(TaskPoolORM.status == TaskStatus.PENDING.value)
            .order_by(TaskPoolORM.id)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )

        if env_ids:
            query = query.where(TaskPoolORM.env_id.in_(env_ids))

        result = await session.execute(query)
        tasks = list(result.scalars().all())

        # Assign tasks to executor
        now = datetime.utcnow()
        for task in tasks:
            task.status = TaskStatus.ASSIGNED.value
            task.assigned_to = executor_id
            task.assigned_at = now

        logger.info(
            "tasks_fetched",
            executor=executor_id,
            count=len(tasks),
        )
        return tasks

    async def submit_task_result(
        self,
        session: AsyncSession,
        task_uuid: str,
        executor_id: str,
        success: bool,
        score: float,
        total_reward: float = 0.0,
        timesteps: int = 0,
        error: str | None = None,
    ) -> bool:
        """Submit the result for a completed task.

        Args:
            session: Database session
            task_uuid: UUID of the task
            executor_id: ID of the executor submitting the result
            success: Whether the task completed successfully
            score: Score/success rate for the task
            total_reward: Total reward accumulated
            timesteps: Number of timesteps executed
            error: Error message if failed

        Returns:
            True if result was accepted, False if rejected
        """
        result = await session.execute(
            select(TaskPoolORM).where(TaskPoolORM.task_uuid == task_uuid).with_for_update()
        )
        task = result.scalar_one_or_none()

        if task is None:
            logger.warning("task_not_found", task_uuid=task_uuid)
            return False

        # Verify executor owns this task
        if task.assigned_to != executor_id:
            logger.warning(
                "task_executor_mismatch",
                task_uuid=task_uuid,
                expected=task.assigned_to,
                actual=executor_id,
            )
            return False

        # Update task with result
        # COMPLETED = task ran successfully, FAILED = task failed (regardless of error message)
        task.status = TaskStatus.COMPLETED.value if success else TaskStatus.FAILED.value
        task.completed_at = datetime.utcnow()
        task.result = {
            "success": success,
            "score": score,
            "total_reward": total_reward,
            "timesteps": timesteps,
            "error": error,
        }

        logger.info(
            "task_result_submitted",
            task_uuid=task_uuid,
            success=success,
            score=score,
        )
        return True

    async def get_task_pool_stats(
        self,
        session: AsyncSession,
        cycle_id: int | None = None,
    ) -> dict:
        """Get statistics about the task pool.

        Args:
            session: Database session
            cycle_id: Optional filter by cycle ID

        Returns:
            Dict with task pool statistics
        """
        # Base query
        base_filter = TaskPoolORM.cycle_id == cycle_id if cycle_id else True

        # Count by status
        result = await session.execute(
            select(TaskPoolORM.status, func.count()).where(base_filter).group_by(TaskPoolORM.status)
        )
        status_counts = dict(result.all())

        # Get active executors (assigned tasks)
        result = await session.execute(
            select(func.distinct(TaskPoolORM.assigned_to))
            .where(base_filter)
            .where(TaskPoolORM.status == TaskStatus.ASSIGNED.value)
            .where(TaskPoolORM.assigned_to.isnot(None))
        )
        active_executors = [r for r in result.scalars().all() if r is not None]

        return {
            "total_tasks": sum(status_counts.values()),
            "pending_tasks": status_counts.get(TaskStatus.PENDING.value, 0),
            "assigned_tasks": status_counts.get(TaskStatus.ASSIGNED.value, 0),
            "completed_tasks": status_counts.get(TaskStatus.COMPLETED.value, 0),
            "failed_tasks": status_counts.get(TaskStatus.FAILED.value, 0),
            "active_executors": active_executors,
            "current_cycle_id": cycle_id,
        }

    async def count_pending_tasks(
        self,
        session: AsyncSession,
        cycle_id: int | None = None,
    ) -> int:
        """Count pending tasks in the pool."""
        query = (
            select(func.count())
            .select_from(TaskPoolORM)
            .where(TaskPoolORM.status == TaskStatus.PENDING.value)
        )
        if cycle_id:
            query = query.where(TaskPoolORM.cycle_id == cycle_id)
        result = await session.execute(query)
        return result.scalar() or 0

    async def get_cycle_task_results(
        self,
        session: AsyncSession,
        cycle_id: int,
    ) -> list[TaskPoolORM]:
        """Get all completed tasks for a cycle."""
        result = await session.execute(
            select(TaskPoolORM)
            .where(TaskPoolORM.cycle_id == cycle_id)
            .where(TaskPoolORM.status.in_([TaskStatus.COMPLETED.value, TaskStatus.FAILED.value]))
        )
        return list(result.scalars().all())

    async def is_cycle_complete(
        self,
        session: AsyncSession,
        cycle_id: int,
    ) -> bool:
        """Check if all tasks in a cycle are complete."""
        pending = await session.execute(
            select(func.count())
            .select_from(TaskPoolORM)
            .where(TaskPoolORM.cycle_id == cycle_id)
            .where(TaskPoolORM.status.in_([TaskStatus.PENDING.value, TaskStatus.ASSIGNED.value]))
        )
        return (pending.scalar() or 0) == 0

    async def reassign_stale_tasks(
        self,
        session: AsyncSession,
        stale_threshold_seconds: int = 300,
    ) -> int:
        """Reassign tasks that have been assigned but not completed within threshold.

        Args:
            session: Database session
            stale_threshold_seconds: Time after which assigned tasks are considered stale

        Returns:
            Number of tasks reassigned
        """
        from datetime import timedelta

        threshold = datetime.utcnow() - timedelta(seconds=stale_threshold_seconds)

        result = await session.execute(
            select(TaskPoolORM)
            .where(TaskPoolORM.status == TaskStatus.ASSIGNED.value)
            .where(TaskPoolORM.assigned_at < threshold)
            .with_for_update()
        )
        stale_tasks = list(result.scalars().all())

        for task in stale_tasks:
            task.status = TaskStatus.PENDING.value
            task.assigned_to = None
            task.assigned_at = None

        if stale_tasks:
            logger.info("stale_tasks_reassigned", count=len(stale_tasks))

        return len(stale_tasks)
