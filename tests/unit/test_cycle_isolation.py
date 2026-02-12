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


def _make_mock_cycle(
    cycle_id: int = 1,
    status: str = EvaluationCycleStatus.RUNNING.value,
) -> MagicMock:
    """Create a mock EvaluationCycleORM."""
    mock = MagicMock(spec=EvaluationCycleORM)
    mock.id = cycle_id
    mock.status = status
    return mock


def _make_mock_task(status: str = TaskStatus.PENDING.value) -> MagicMock:
    """Create a mock TaskPoolORM."""
    mock = MagicMock(spec=TaskPoolORM)
    mock.status = status
    return mock


def _mock_execute_results(*results_lists: list) -> AsyncMock:
    """Build an AsyncMock side_effect from lists of ORM objects per query."""
    side_effects = []
    for items in results_lists:
        result = MagicMock()
        result.scalars.return_value.all.return_value = items
        side_effects.append(result)
    return AsyncMock(side_effect=side_effects)


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
        mock_session.execute = _mock_execute_results([])

        storage = Storage("postgresql+asyncpg://test:test@localhost/test")

        cycles_cancelled, tasks_cancelled = await storage.cancel_incomplete_cycles(mock_session)

        assert cycles_cancelled == 0
        assert tasks_cancelled == 0

    @pytest.mark.asyncio
    async def test_cancels_running_cycle_and_tasks(self, mock_session):
        """Running cycles and their pending/assigned tasks are cancelled."""
        mock_cycle = _make_mock_cycle()
        mock_task1 = _make_mock_task(TaskStatus.PENDING.value)
        mock_task2 = _make_mock_task(TaskStatus.ASSIGNED.value)

        mock_session.execute = _mock_execute_results([mock_cycle], [mock_task1, mock_task2])

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
        mock_cycle = _make_mock_cycle()
        mock_task = _make_mock_task(TaskStatus.PENDING.value)

        mock_session.execute = _mock_execute_results([mock_cycle], [mock_task])

        storage = Storage("postgresql+asyncpg://test:test@localhost/test")

        cycles_cancelled, tasks_cancelled = await storage.cancel_incomplete_cycles(mock_session)

        # Only the pending task is cancelled
        assert tasks_cancelled == 1
        assert mock_task.status == TaskStatus.FAILED.value
