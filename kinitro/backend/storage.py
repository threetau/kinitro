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
