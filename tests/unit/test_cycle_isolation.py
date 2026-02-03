"""Tests for cycle isolation (cancel_incomplete_cycles)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from kinitro.backend.models import (
    EvaluationCycleORM,
    EvaluationCycleStatus,
    TaskPoolORM,
    TaskStatus,
)
from kinitro.backend.storage import Storage
from kinitro.scheduler.config import SchedulerConfig


class TestCancelIncompleteCycles:
    """Tests for Storage.cancel_incomplete_cycles()."""

    @pytest.fixture
    def mock_session(self):
        """Create a mock async session."""
        session = AsyncMock()
        return session

    @pytest.mark.asyncio
    async def test_no_incomplete_cycles(self, mock_session):
        """When no incomplete cycles exist, nothing is cancelled."""
        # Mock execute to return empty result for cycles query
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute = AsyncMock(return_value=mock_result)

        storage = Storage("postgresql+asyncpg://test:test@localhost/test")

        cycles_cancelled, tasks_cancelled = await storage.cancel_incomplete_cycles(mock_session)

        assert cycles_cancelled == 0
        assert tasks_cancelled == 0

    @pytest.mark.asyncio
    async def test_cancels_running_cycle_and_tasks(self, mock_session):
        """Running cycles and their pending/assigned tasks are cancelled."""
        # Create mock cycle
        mock_cycle = MagicMock(spec=EvaluationCycleORM)
        mock_cycle.id = 1
        mock_cycle.status = EvaluationCycleStatus.RUNNING.value

        # Create mock tasks
        mock_task1 = MagicMock(spec=TaskPoolORM)
        mock_task1.status = TaskStatus.PENDING.value
        mock_task2 = MagicMock(spec=TaskPoolORM)
        mock_task2.status = TaskStatus.ASSIGNED.value

        # Mock execute - first call returns cycles, second returns tasks
        cycles_result = MagicMock()
        cycles_result.scalars.return_value.all.return_value = [mock_cycle]

        tasks_result = MagicMock()
        tasks_result.scalars.return_value.all.return_value = [mock_task1, mock_task2]

        mock_session.execute = AsyncMock(side_effect=[cycles_result, tasks_result])

        storage = Storage("postgresql+asyncpg://test:test@localhost/test")

        cycles_cancelled, tasks_cancelled = await storage.cancel_incomplete_cycles(mock_session)

        assert cycles_cancelled == 1
        assert tasks_cancelled == 2

        # Verify cycle was marked as failed
        assert mock_cycle.status == EvaluationCycleStatus.FAILED.value
        assert mock_cycle.error_message == "Cancelled on scheduler restart (cycle isolation)"
        assert mock_cycle.completed_at is not None

        # Verify tasks were marked as failed
        assert mock_task1.status == TaskStatus.FAILED.value
        assert mock_task1.result == {"error": "Cycle cancelled on scheduler restart"}
        assert mock_task2.status == TaskStatus.FAILED.value

    @pytest.mark.asyncio
    async def test_leaves_completed_tasks_unchanged(self, mock_session):
        """Completed/failed tasks from incomplete cycles are not modified."""
        # Create mock cycle
        mock_cycle = MagicMock(spec=EvaluationCycleORM)
        mock_cycle.id = 1
        mock_cycle.status = EvaluationCycleStatus.RUNNING.value

        # Only pending task (completed tasks not returned by query)
        mock_task = MagicMock(spec=TaskPoolORM)
        mock_task.status = TaskStatus.PENDING.value

        cycles_result = MagicMock()
        cycles_result.scalars.return_value.all.return_value = [mock_cycle]

        # Query only returns pending/assigned, not completed
        tasks_result = MagicMock()
        tasks_result.scalars.return_value.all.return_value = [mock_task]

        mock_session.execute = AsyncMock(side_effect=[cycles_result, tasks_result])

        storage = Storage("postgresql+asyncpg://test:test@localhost/test")

        cycles_cancelled, tasks_cancelled = await storage.cancel_incomplete_cycles(mock_session)

        # Only the pending task is cancelled
        assert tasks_cancelled == 1
        assert mock_task.status == TaskStatus.FAILED.value


class TestSchedulerConfig:
    """Tests for scheduler config cycle isolation option."""

    def test_cleanup_incomplete_cycles_default_true(self):
        """Default value for cleanup_incomplete_cycles is True."""
        config = SchedulerConfig(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            network="test",
            netuid=1,
        )

        assert config.cleanup_incomplete_cycles is True

    def test_cleanup_incomplete_cycles_can_be_disabled(self):
        """cleanup_incomplete_cycles can be set to False."""
        config = SchedulerConfig(
            database_url="postgresql+asyncpg://test:test@localhost/test",
            network="test",
            netuid=1,
            cleanup_incomplete_cycles=False,
        )

        assert config.cleanup_incomplete_cycles is False
