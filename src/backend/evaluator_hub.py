"""
Evaluator hub for Kinitro backend.

Manages direct WebSocket connections from evaluators.
Handles job routing and evaluator lifecycle.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Awaitable, Callable, Dict, List, Optional

from fastapi import WebSocket
from sqlmodel import SQLModel

from core.log import get_logger
from core.messages import EvalJobMessage

logger = get_logger(__name__)

ConnectionId = str  # Unique ID for each WebSocket connection
EvaluatorId = str  # Unique evaluator instance ID


@dataclass
class EvaluatorState:
    """In-memory state for a connected evaluator."""

    connection_id: ConnectionId
    evaluator_id: EvaluatorId
    websocket: WebSocket
    api_key_id: Optional[int]
    supported_task_types: List[str]
    max_concurrent_jobs: int
    current_job_count: int
    last_heartbeat: datetime
    capabilities: Optional[dict]


class EvaluatorHub:
    """
    Manages WebSocket connections from evaluators.

    This class is responsible for:
    - Registering and unregistering evaluator connections
    - Routing jobs to evaluators based on capacity and task type
    - Broadcasting messages to evaluators
    - Tracking evaluator state and health
    """

    def __init__(
        self,
        on_result_received: Optional[Callable[[dict], Awaitable[None]]] = None,
        on_status_update: Optional[Callable[[dict], Awaitable[None]]] = None,
    ):
        # WebSocket connections by connection_id
        self._connections: Dict[ConnectionId, EvaluatorState] = {}
        # Map evaluator_id -> connection_id for lookup
        self._evaluator_to_connection: Dict[EvaluatorId, ConnectionId] = {}
        # Round-robin index for job assignment
        self._round_robin_index: int = 0
        # Callbacks
        self._on_result_received = on_result_received
        self._on_status_update = on_status_update

    async def register(
        self,
        connection_id: ConnectionId,
        evaluator_id: EvaluatorId,
        websocket: WebSocket,
        api_key_id: Optional[int] = None,
        supported_task_types: Optional[List[str]] = None,
        max_concurrent_jobs: int = 1,
        capabilities: Optional[dict] = None,
    ) -> EvaluatorState:
        """
        Register a new evaluator connection.

        If an evaluator with the same evaluator_id is already connected,
        the old connection is closed and replaced.

        Args:
            connection_id: Unique identifier for this WebSocket connection
            evaluator_id: Unique identifier for the evaluator instance
            websocket: The WebSocket connection
            api_key_id: ID of the API key used for authentication
            supported_task_types: List of task types this evaluator can handle
            max_concurrent_jobs: Maximum concurrent jobs this evaluator can run
            capabilities: Additional evaluator capabilities (GPU info, etc.)

        Returns:
            The EvaluatorState for the registered connection
        """
        # Check if this evaluator_id is already connected
        if evaluator_id in self._evaluator_to_connection:
            old_conn_id = self._evaluator_to_connection[evaluator_id]
            logger.warning(
                f"Evaluator {evaluator_id} reconnecting, closing old connection {old_conn_id}"
            )
            await self._close_connection(old_conn_id)

        state = EvaluatorState(
            connection_id=connection_id,
            evaluator_id=evaluator_id,
            websocket=websocket,
            api_key_id=api_key_id,
            supported_task_types=supported_task_types or ["rl_rollout"],
            max_concurrent_jobs=max_concurrent_jobs,
            current_job_count=0,
            last_heartbeat=datetime.now(timezone.utc),
            capabilities=capabilities,
        )

        self._connections[connection_id] = state
        self._evaluator_to_connection[evaluator_id] = connection_id

        logger.info(
            f"Registered evaluator {evaluator_id} (connection {connection_id}) "
            f"with {max_concurrent_jobs} max concurrent jobs, "
            f"supporting task types: {state.supported_task_types}"
        )

        return state

    async def unregister(self, connection_id: ConnectionId) -> Optional[EvaluatorId]:
        """
        Unregister an evaluator connection.

        Args:
            connection_id: The connection ID to unregister

        Returns:
            The evaluator_id of the unregistered evaluator, or None if not found
        """
        state = self._connections.pop(connection_id, None)
        if not state:
            return None

        self._evaluator_to_connection.pop(state.evaluator_id, None)

        try:
            await state.websocket.close()
        except Exception as e:
            logger.debug(f"Error closing WebSocket for {connection_id}: {e}")

        logger.info(
            f"Unregistered evaluator {state.evaluator_id} (connection {connection_id})"
        )

        return state.evaluator_id

    async def _close_connection(self, connection_id: ConnectionId) -> None:
        """Close a connection without full unregistration."""
        state = self._connections.get(connection_id)
        if state:
            try:
                await state.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket for {connection_id}: {e}")

    async def send_job(self, evaluator_id: EvaluatorId, job: EvalJobMessage) -> bool:
        """
        Send a job to a specific evaluator.

        Args:
            evaluator_id: The evaluator to send the job to
            job: The job message

        Returns:
            True if the job was sent successfully
        """
        conn_id = self._evaluator_to_connection.get(evaluator_id)
        if not conn_id:
            logger.warning(f"No connection found for evaluator {evaluator_id}")
            return False

        state = self._connections.get(conn_id)
        if not state:
            logger.warning(f"No state found for connection {conn_id}")
            return False

        try:
            await state.websocket.send_text(job.model_dump_json())
            state.current_job_count += 1
            logger.debug(
                f"Sent job {job.job_id} to evaluator {evaluator_id} "
                f"(current jobs: {state.current_job_count})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to send job to evaluator {evaluator_id}: {e}")
            # Don't increment job count on failure
            return False

    async def route_job(
        self, job: EvalJobMessage, task_type: str = "rl_rollout"
    ) -> Optional[EvaluatorId]:
        """
        Route a job to an available evaluator using round-robin with capacity check.

        Args:
            job: The job message to route
            task_type: The task type required for this job

        Returns:
            The evaluator_id that accepted the job, or None if no evaluator available
        """
        # Get evaluators that support this task type and have capacity
        available = self._get_available_evaluators(task_type)
        if not available:
            logger.warning(
                f"No available evaluators for task type {task_type}, "
                f"total connected: {len(self._connections)}"
            )
            return None

        # Round-robin selection
        self._round_robin_index = self._round_robin_index % len(available)
        selected_evaluator_id = available[self._round_robin_index]
        self._round_robin_index += 1

        # Send the job
        if await self.send_job(selected_evaluator_id, job):
            return selected_evaluator_id

        return None

    async def broadcast_job(self, job: EvalJobMessage) -> int:
        """
        Broadcast a job to all connected evaluators.

        Args:
            job: The job message to broadcast

        Returns:
            Number of evaluators that received the job
        """
        sent_count = 0
        failed_connections: List[ConnectionId] = []

        for conn_id, state in list(self._connections.items()):
            try:
                await state.websocket.send_text(job.model_dump_json())
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to {state.evaluator_id}: {e}")
                failed_connections.append(conn_id)

        # Clean up failed connections
        for conn_id in failed_connections:
            await self.unregister(conn_id)

        return sent_count

    async def send_message(self, evaluator_id: EvaluatorId, message: SQLModel) -> bool:
        """
        Send an arbitrary message to a specific evaluator.

        Args:
            evaluator_id: The evaluator to send to
            message: The message to send

        Returns:
            True if sent successfully
        """
        conn_id = self._evaluator_to_connection.get(evaluator_id)
        if not conn_id:
            return False

        state = self._connections.get(conn_id)
        if not state:
            return False

        try:
            await state.websocket.send_text(message.model_dump_json())
            return True
        except Exception as e:
            logger.error(f"Failed to send message to {evaluator_id}: {e}")
            return False

    async def broadcast_message(self, message: SQLModel) -> int:
        """
        Broadcast a message to all connected evaluators.

        Args:
            message: The message to broadcast

        Returns:
            Number of evaluators that received the message
        """
        sent_count = 0
        failed_connections: List[ConnectionId] = []

        for conn_id, state in list(self._connections.items()):
            try:
                await state.websocket.send_text(message.model_dump_json())
                sent_count += 1
            except Exception as e:
                logger.error(f"Failed to broadcast to {state.evaluator_id}: {e}")
                failed_connections.append(conn_id)

        for conn_id in failed_connections:
            await self.unregister(conn_id)

        return sent_count

    def update_heartbeat(self, evaluator_id: EvaluatorId) -> bool:
        """
        Update the last heartbeat time for an evaluator.

        Args:
            evaluator_id: The evaluator ID

        Returns:
            True if the evaluator was found and updated
        """
        conn_id = self._evaluator_to_connection.get(evaluator_id)
        if not conn_id:
            return False

        state = self._connections.get(conn_id)
        if not state:
            return False

        state.last_heartbeat = datetime.now(timezone.utc)
        return True

    def job_completed(self, evaluator_id: EvaluatorId) -> bool:
        """
        Mark a job as completed for an evaluator (decrement job count).

        Args:
            evaluator_id: The evaluator ID

        Returns:
            True if the evaluator was found and updated
        """
        conn_id = self._evaluator_to_connection.get(evaluator_id)
        if not conn_id:
            return False

        state = self._connections.get(conn_id)
        if not state:
            return False

        if state.current_job_count > 0:
            state.current_job_count -= 1

        return True

    def _get_available_evaluators(self, task_type: str) -> List[EvaluatorId]:
        """
        Get list of evaluator IDs that support a task type and have capacity.

        Args:
            task_type: The task type to check for

        Returns:
            List of evaluator IDs with capacity
        """
        available = []
        for state in self._connections.values():
            if task_type in state.supported_task_types:
                if state.current_job_count < state.max_concurrent_jobs:
                    available.append(state.evaluator_id)
        return available

    def get_evaluator_ids(self) -> List[EvaluatorId]:
        """Get list of all connected evaluator IDs."""
        return list(self._evaluator_to_connection.keys())

    def get_state(self, evaluator_id: EvaluatorId) -> Optional[EvaluatorState]:
        """Get the state for an evaluator."""
        conn_id = self._evaluator_to_connection.get(evaluator_id)
        if not conn_id:
            return None
        return self._connections.get(conn_id)

    def get_connection_for_evaluator(
        self, evaluator_id: EvaluatorId
    ) -> Optional[ConnectionId]:
        """Get the connection ID for an evaluator."""
        return self._evaluator_to_connection.get(evaluator_id)

    def has_connections(self) -> bool:
        """Check if there are any active connections."""
        return bool(self._connections)

    def connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self._connections)

    def total_capacity(self) -> int:
        """Get the total job capacity across all connected evaluators."""
        return sum(state.max_concurrent_jobs for state in self._connections.values())

    def available_capacity(self) -> int:
        """Get the available job capacity across all connected evaluators."""
        return sum(
            state.max_concurrent_jobs - state.current_job_count
            for state in self._connections.values()
        )

    async def close_all(self) -> None:
        """Close all WebSocket connections."""
        for conn_id in list(self._connections.keys()):
            await self.unregister(conn_id)

        self._connections.clear()
        self._evaluator_to_connection.clear()
        logger.info("All evaluator connections closed")
