"""
Real-time update system for Kinitro Backend.

This module provides WebSocket-based real-time updates to frontend clients,
including evaluation results, job status updates, episode data streaming, etc.
"""

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from fastapi import WebSocket
from pydantic import BaseModel
from sqlalchemy import and_, func, select

from backend.constants import INITIAL_STATE_DATA_LIMIT
from backend.events import (
    BaseEvent,
    CompetitionCreatedEvent,
    EpisodeCompletedEvent,
    EvaluationCompletedEvent,
    JobCompletedEvent,
    JobCreatedEvent,
    JobStatusChangedEvent,
    StatsUpdatedEvent,
    SubmissionReceivedEvent,
    ValidatorConnectedEvent,
)
from core.log import get_logger
from core.messages import (
    EventMessage,
    EventType,
    MessageType,
    PongMessage,
    SubscribeMessage,
    SubscriptionAckMessage,
    SubscriptionRequest,
    UnsubscribeMessage,
    UnsubscriptionAckMessage,
)

if TYPE_CHECKING:
    from backend.service import BackendService

# Import models at runtime to avoid circular imports
try:
    from backend.models import (
        BackendEvaluationJob,
        BackendEvaluationJobStatus,
        BackendEvaluationResult,
        BackendState,
        Competition,
        EpisodeData,
        MinerSubmission,
        ValidatorConnection,
    )
    from core.db.models import EvaluationStatus
except ImportError:
    # Handle case where models might not be available during testing
    pass

logger = get_logger(__name__)


class ClientConnection:
    """Represents a connected client with their subscriptions."""

    def __init__(self, connection_id: str, websocket: WebSocket):
        self.connection_id = connection_id
        self.websocket = websocket
        self.subscriptions: Dict[str, SubscriptionRequest] = {}
        self.connected_at = datetime.now(timezone.utc)
        self.last_ping = datetime.now(timezone.utc)

    def add_subscription(self, subscription_id: str, request: SubscriptionRequest):
        """Add a subscription for this client."""
        self.subscriptions[subscription_id] = request

    def remove_subscription(self, subscription_id: str) -> bool:
        """Remove a subscription. Returns True if subscription existed."""
        if subscription_id in self.subscriptions:
            del self.subscriptions[subscription_id]
            return True
        return False

    def should_receive_event(
        self, event_type: EventType, event_data: Dict[str, Any]
    ) -> List[str]:
        """
        Check if this client should receive an event.
        Returns list of subscription IDs that match.
        """
        matching_subscriptions = []

        logger.debug(
            f"Client {self.connection_id} checking event {event_type} against {len(self.subscriptions)} subscriptions"
        )

        for sub_id, subscription in self.subscriptions.items():
            # Check if event type is subscribed
            if event_type not in subscription.event_types:
                logger.debug(
                    f"Subscription {sub_id} does not include event type {event_type}"
                )
                continue

            # Check filters
            matches = True
            for filter_key, filter_value in subscription.filters.items():
                if filter_key in event_data:
                    if event_data[filter_key] != filter_value:
                        matches = False
                        logger.debug(
                            f"Filter mismatch: {filter_key}={event_data[filter_key]} != {filter_value}"
                        )
                        break

            if matches:
                matching_subscriptions.append(sub_id)
                logger.debug(f"Subscription {sub_id} matches event {event_type}")

        logger.debug(
            f"Client {self.connection_id} has {len(matching_subscriptions)} matching subscriptions"
        )
        return matching_subscriptions


class RealtimeEventBroadcaster:
    """
    Central event broadcaster for real-time updates.
    Manages client connections and event distribution.
    """

    def __init__(self, backend_service: Optional["BackendService"] = None):
        self.client_connections: Dict[str, ClientConnection] = {}
        self._broadcast_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        self._broadcast_task = None
        self.backend_service = backend_service

    def set_backend_service(self, backend_service: "BackendService"):
        """Set the backend service reference for database access."""
        self.backend_service = backend_service

    async def start(self):
        """Start the event broadcaster."""
        self._running = True
        self._broadcast_task = asyncio.create_task(self._process_broadcast_queue())
        logger.info("Realtime event broadcaster started")

    async def stop(self):
        """Stop the event broadcaster."""
        self._running = False
        if self._broadcast_task:
            self._broadcast_task.cancel()
            try:
                await self._broadcast_task
            except asyncio.CancelledError:
                pass

        # Close all client connections
        for connection in self.client_connections.values():
            await connection.websocket.close()

        self.client_connections.clear()
        logger.info("Realtime event broadcaster stopped")

    async def add_client(
        self, connection_id: str, websocket: WebSocket
    ) -> ClientConnection:
        """Add a new client connection."""
        connection = ClientConnection(connection_id, websocket)
        self.client_connections[connection_id] = connection
        logger.info(f"Client {connection_id} connected")
        return connection

    async def remove_client(self, connection_id: str):
        """Remove a client connection."""
        if connection_id in self.client_connections:
            del self.client_connections[connection_id]
            logger.info(f"Client {connection_id} disconnected")

    async def broadcast_event(
        self, event_type: EventType, event_data: Union[BaseEvent, Dict[str, Any]]
    ):
        """
        Broadcast an event to all relevant clients.
        This is the main entry point for sending events.

        Args:
            event_type: The type of event to broadcast
            event_data: Event data as a Pydantic model or dictionary
        """
        # Convert Pydantic model to dictionary if needed
        if isinstance(event_data, BaseModel):
            event_data_dict = event_data.model_dump(mode="json")
        else:
            event_data_dict = event_data

        logger.debug(
            f"Broadcasting event: {event_type} to {len(self.client_connections)} clients"
        )
        await self._broadcast_queue.put((event_type, event_data_dict))

    async def _process_broadcast_queue(self):
        """Process events from the broadcast queue and send to clients."""
        while self._running:
            try:
                # Wait for events with timeout to allow checking _running flag
                event_type, event_data = await asyncio.wait_for(
                    self._broadcast_queue.get(), timeout=1.0
                )

                logger.debug(
                    f"Processing event: {event_type} for {len(self.client_connections)} clients"
                )

                # Send to all relevant clients
                disconnected_clients = []
                sent_count = 0

                for connection_id, connection in self.client_connections.items():
                    subscription_ids = connection.should_receive_event(
                        event_type, event_data
                    )

                    if subscription_ids:
                        # Client should receive this event
                        for sub_id in subscription_ids:
                            message = EventMessage(
                                event_type=event_type,
                                event_data=event_data,
                                subscription_id=sub_id,
                            )

                            try:
                                await connection.websocket.send_text(
                                    message.model_dump_json()
                                )
                                sent_count += 1
                                logger.debug(
                                    f"Sent {event_type} event to client {connection_id}"
                                )
                            except Exception as e:
                                logger.error(
                                    f"Failed to send event to client {connection_id}: {e}"
                                )
                                disconnected_clients.append(connection_id)

                # Remove disconnected clients
                for connection_id in disconnected_clients:
                    await self.remove_client(connection_id)

                if sent_count > 0:
                    logger.debug(
                        f"Successfully sent {event_type} event to {sent_count} clients"
                    )
                else:
                    logger.debug(f"No clients subscribed to {event_type} event")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing broadcast queue: {e}")

    async def handle_client_message(self, connection_id: str, message: str):
        """Handle a message from a client."""
        try:
            data = json.loads(message)
            message_type = data.get("message_type")

            connection = self.client_connections.get(connection_id)
            if not connection:
                logger.error(f"Unknown connection {connection_id}")
                return

            if message_type == MessageType.SUBSCRIBE:
                await self._handle_subscribe(connection, data)
            elif message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscribe(connection, data)
            elif message_type == MessageType.PING:
                await self._handle_ping(connection, data)
            else:
                error_msg = EventMessage(
                    event_type=EventType.STATS_UPDATED,  # Placeholder, should be error type
                    event_data={"error": f"Unknown message type: {message_type}"},
                )
                await connection.websocket.send_text(error_msg.model_dump_json())

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from client {connection_id}: {e}")
            connection = self.client_connections.get(connection_id)
            if connection:
                error_msg = EventMessage(
                    event_type=EventType.STATS_UPDATED,  # Placeholder
                    event_data={"error": "Invalid JSON format"},
                )
                await connection.websocket.send_text(error_msg.model_dump_json())
        except Exception as e:
            logger.error(f"Error handling client message: {e}")

    async def _handle_subscribe(self, connection: ClientConnection, data: dict):
        """Handle subscription request."""
        try:
            msg = SubscribeMessage.model_validate(data)

            # Generate subscription ID
            subscription_id = str(uuid.uuid4())

            # Add subscription
            connection.add_subscription(subscription_id, msg.subscription)

            # Send acknowledgment
            ack = SubscriptionAckMessage(
                subscription_id=subscription_id,
                subscribed_events=msg.subscription.event_types,
                request_id=msg.request_id,
            )
            await connection.websocket.send_text(ack.model_dump_json())

            # Send initial state data for relevant subscriptions
            await self._send_initial_state_data(
                connection, subscription_id, msg.subscription
            )

            logger.info(
                f"Client {connection.connection_id} subscribed to {msg.subscription.event_types} with subscription {subscription_id}"
            )
            logger.debug(
                f"Client {connection.connection_id} now has {len(connection.subscriptions)} subscriptions"
            )

        except Exception as e:
            logger.error(f"Failed to subscribe: {str(e)}")

    async def _handle_unsubscribe(self, connection: ClientConnection, data: dict):
        """Handle unsubscription request."""
        try:
            msg = UnsubscribeMessage.model_validate(data)

            # Remove subscription
            if connection.remove_subscription(msg.subscription_id):
                # Send acknowledgment
                ack = UnsubscriptionAckMessage(
                    subscription_id=msg.subscription_id, request_id=msg.request_id
                )
                await connection.websocket.send_text(ack.model_dump_json())

                logger.info(
                    f"Client {connection.connection_id} unsubscribed from {msg.subscription_id}"
                )
            else:
                logger.warning(
                    f"Subscription {msg.subscription_id} not found for client {connection.connection_id}"
                )

        except Exception as e:
            logger.error(f"Failed to unsubscribe: {str(e)}")

    async def _handle_ping(self, connection: ClientConnection, data: dict):
        """Handle ping message."""
        connection.last_ping = datetime.now(timezone.utc)
        pong = PongMessage(request_id=data.get("request_id"))
        await connection.websocket.send_text(pong.model_dump_json())

    async def _send_initial_state_data(
        self,
        connection: ClientConnection,
        subscription_id: str,
        subscription: SubscriptionRequest,
    ):
        """Send initial state data for applicable event types after subscription."""
        if not self.backend_service or not self.backend_service.async_session:
            logger.debug(
                "No backend service or database session available for initial state data"
            )
            return

        try:
            async with self.backend_service.async_session() as session:
                # Send initial data based on subscribed event types
                for event_type in subscription.event_types:
                    initial_data = await self._get_initial_state_for_event_type(
                        session, event_type, subscription.filters
                    )

                    if initial_data:
                        for data_item in initial_data:
                            message = EventMessage(
                                event_type=event_type,
                                event_data=data_item,
                                subscription_id=subscription_id,
                            )
                            await connection.websocket.send_text(
                                message.model_dump_json()
                            )
                            logger.debug(
                                f"Sent initial state data for {event_type} to client {connection.connection_id}"
                            )

        except Exception as e:
            logger.error(f"Failed to send initial state data: {str(e)}")

    async def _get_initial_state_for_event_type(
        self, session, event_type: EventType, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get initial state data for a specific event type."""
        try:
            if event_type == EventType.STATS_UPDATED:
                return await self._get_initial_stats_data(session)

            # Job-related events
            elif event_type == EventType.JOB_CREATED:
                return await self._get_initial_job_created_data(session, filters)
            elif event_type == EventType.JOB_STATUS_CHANGED:
                return await self._get_initial_job_status_data(session, filters)
            elif event_type == EventType.JOB_COMPLETED:
                return await self._get_initial_job_completed_data(session, filters)

            # Evaluation-related events
            elif event_type in [
                EventType.EVALUATION_STARTED,
                EventType.EVALUATION_COMPLETED,
                EventType.EVALUATION_PROGRESS,
            ]:
                return await self._get_initial_evaluation_data(session, filters)

            # Competition-related events
            elif event_type in [
                EventType.COMPETITION_ACTIVATED,
                EventType.COMPETITION_CREATED,
                EventType.COMPETITION_UPDATED,
                EventType.COMPETITION_DEACTIVATED,
            ]:
                return await self._get_initial_competition_data(session, filters)

            # Submission events
            elif event_type == EventType.SUBMISSION_RECEIVED:
                return await self._get_initial_submission_data(session, filters)

            # Episode events (but not EPISODE_STEP - too granular)
            elif event_type in [EventType.EPISODE_COMPLETED, EventType.EPISODE_STARTED]:
                return await self._get_initial_episode_data(session, filters)

            # Validator events
            elif event_type == EventType.VALIDATOR_CONNECTED:
                return await self._get_initial_validator_data(session, filters)

            # Skip these events - either not suitable for initial data or too granular
            elif event_type in [
                EventType.VALIDATOR_DISCONNECTED,
                EventType.EPISODE_STEP,
            ]:
                return []

            # Unknown event types
            else:
                logger.debug(
                    f"No initial state data handler for event type: {event_type}"
                )
                return []

        except Exception as e:
            logger.error(f"Error getting initial state data for {event_type}: {str(e)}")
            return []

    async def _get_initial_stats_data(self, session) -> List[Dict[str, Any]]:
        """Get current backend statistics."""
        try:
            # Models are imported at module level

            # Get competitions
            comp_result = await session.execute(select(Competition))
            competitions = comp_result.scalars().all()
            active_comps = [c for c in competitions if c.active]
            total_points = sum(c.points for c in active_comps)

            # Get validators
            val_result = await session.execute(
                select(ValidatorConnection).where(ValidatorConnection.is_connected)
            )
            connected_validators = len(val_result.scalars().all())

            # Get submissions count
            sub_result = await session.execute(select(func.count(MinerSubmission.id)))
            total_submissions = sub_result.scalar() or 0

            # Get jobs count
            job_result = await session.execute(
                select(func.count(BackendEvaluationJob.id))
            )
            total_jobs = job_result.scalar() or 0

            # Get completed jobs count (latest status is COMPLETED)
            latest_status_subquery = (
                select(
                    BackendEvaluationJobStatus.job_id,
                    func.max(BackendEvaluationJobStatus.created_at).label(
                        "max_created_at"
                    ),
                )
                .group_by(BackendEvaluationJobStatus.job_id)
                .subquery()
            )

            completed_jobs_result = await session.execute(
                select(func.count(BackendEvaluationJob.id.distinct()))
                .select_from(BackendEvaluationJob)
                .join(
                    BackendEvaluationJobStatus,
                    BackendEvaluationJob.id == BackendEvaluationJobStatus.job_id,
                )
                .join(
                    latest_status_subquery,
                    and_(
                        BackendEvaluationJobStatus.job_id
                        == latest_status_subquery.c.job_id,
                        BackendEvaluationJobStatus.created_at
                        == latest_status_subquery.c.max_created_at,
                    ),
                )
                .where(BackendEvaluationJobStatus.status == EvaluationStatus.COMPLETED)
            )
            completed_jobs = completed_jobs_result.scalar() or 0

            # Get failed jobs count (latest status is FAILED, CANCELLED, or TIMEOUT)
            failed_jobs_result = await session.execute(
                select(func.count(BackendEvaluationJob.id.distinct()))
                .select_from(BackendEvaluationJob)
                .join(
                    BackendEvaluationJobStatus,
                    BackendEvaluationJob.id == BackendEvaluationJobStatus.job_id,
                )
                .join(
                    latest_status_subquery,
                    and_(
                        BackendEvaluationJobStatus.job_id
                        == latest_status_subquery.c.job_id,
                        BackendEvaluationJobStatus.created_at
                        == latest_status_subquery.c.max_created_at,
                    ),
                )
                .where(
                    BackendEvaluationJobStatus.status.in_(
                        [
                            EvaluationStatus.FAILED,
                            EvaluationStatus.CANCELLED,
                            EvaluationStatus.TIMEOUT,
                        ]
                    )
                )
            )
            failed_jobs = failed_jobs_result.scalar() or 0

            # Get results count
            result_count = await session.execute(
                select(func.count(BackendEvaluationResult.id))
            )
            total_results = result_count.scalar() or 0

            # Get backend state
            state_result = await session.execute(
                select(BackendState).where(BackendState.id == 1)
            )
            state = state_result.scalar_one_or_none()

            # Calculate competition percentages
            comp_percentages = {}
            for comp in active_comps:
                percentage = (
                    (comp.points / total_points * 100) if total_points > 0 else 0
                )
                comp_percentages[comp.id] = percentage

            # Create StatsUpdatedEvent model
            stats_event = StatsUpdatedEvent(
                total_competitions=len(competitions),
                active_competitions=len(active_comps),
                total_points=total_points,
                connected_validators=connected_validators,
                total_submissions=total_submissions,
                total_jobs=total_jobs,
                total_results=total_results,
                completed_jobs=completed_jobs,
                failed_jobs=failed_jobs,
                last_seen_block=state.last_seen_block if state else 0,
                competition_percentages=comp_percentages,
            )

            # Return as dictionary for compatibility with existing code
            return [stats_event.model_dump(mode="json")]

        except Exception as e:
            logger.error(f"Error getting initial stats data: {str(e)}")
            return []

    async def _get_initial_validator_data(
        self, session, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get currently connected validators."""
        try:
            # ValidatorConnection is imported at module level

            query = select(ValidatorConnection).where(ValidatorConnection.is_connected)

            # Apply filters if any
            if filters:
                for filter_key, filter_value in filters.items():
                    if filter_key == "validator_hotkey":
                        query = query.where(
                            ValidatorConnection.validator_hotkey == filter_value
                        )

            result = await session.execute(query)
            validators = result.scalars().all()

            validator_data = []
            for validator in validators:
                # Use ValidatorConnectedEvent model for consistent structure
                event = ValidatorConnectedEvent(
                    validator_hotkey=validator.validator_hotkey,
                    connection_id=validator.connection_id,
                    connected_at=validator.last_connected_at
                    or datetime.now(timezone.utc),
                )
                data = event.model_dump(mode="json")
                validator_data.append(data)

            return validator_data

        except Exception as e:
            logger.error(f"Error getting initial validator data: {str(e)}")
            return []

    async def _get_initial_job_created_data(
        self, session, filters: Dict[str, Any], limit: int = INITIAL_STATE_DATA_LIMIT
    ) -> List[Dict[str, Any]]:
        """Get jobs that are currently queued or running (for JOB_CREATED subscriptions)."""
        try:
            # Find jobs with active statuses (QUEUED, STARTING, RUNNING)
            # We need to get the latest status for each job and filter by those statuses

            # Get all jobs with their latest status
            latest_status_subquery = (
                select(
                    BackendEvaluationJobStatus.job_id,
                    func.max(BackendEvaluationJobStatus.created_at).label(
                        "max_created_at"
                    ),
                )
                .group_by(BackendEvaluationJobStatus.job_id)
                .subquery()
            )

            # Join to get jobs with active statuses
            query = (
                select(BackendEvaluationJob)
                .join(
                    BackendEvaluationJobStatus,
                    BackendEvaluationJob.id == BackendEvaluationJobStatus.job_id,
                )
                .join(
                    latest_status_subquery,
                    and_(
                        BackendEvaluationJobStatus.job_id
                        == latest_status_subquery.c.job_id,
                        BackendEvaluationJobStatus.created_at
                        == latest_status_subquery.c.max_created_at,
                    ),
                )
                .where(
                    BackendEvaluationJobStatus.status.in_(
                        [
                            EvaluationStatus.QUEUED,
                            EvaluationStatus.STARTING,
                            EvaluationStatus.RUNNING,
                        ]
                    )
                )
            )

            # Apply filters if any
            if filters:
                for filter_key, filter_value in filters.items():
                    if filter_key == "job_id":
                        query = query.where(BackendEvaluationJob.id == filter_value)
                    elif filter_key == "competition_id":
                        query = query.where(
                            BackendEvaluationJob.competition_id == filter_value
                        )
                    elif filter_key == "miner_hotkey":
                        query = query.where(
                            BackendEvaluationJob.miner_hotkey == filter_value
                        )

            query = query.order_by(BackendEvaluationJob.created_at.desc()).limit(limit)

            result = await session.execute(query)
            jobs = result.scalars().all()

            job_data = []
            for job in jobs:
                # Use JobCreatedEvent model to match the event structure
                event = JobCreatedEvent(
                    job_id=job.id,
                    competition_id=job.competition_id,
                    submission_id=job.submission_id,
                    miner_hotkey=job.miner_hotkey,
                    hf_repo_id=job.hf_repo_id,
                    env_provider=job.env_provider,
                    benchmark_name=job.benchmark_name,
                    config=job.config if job.config else {},
                )
                job_data.append(event.model_dump(mode="json"))

            return job_data

        except Exception as e:
            logger.error(f"Error getting initial job created data: {str(e)}")
            return []

    async def _get_initial_job_status_data(
        self, session, filters: Dict[str, Any], limit: int = INITIAL_STATE_DATA_LIMIT
    ) -> List[Dict[str, Any]]:
        """Get recent job status changes (for JOB_STATUS_CHANGED subscriptions)."""
        try:
            query = select(BackendEvaluationJobStatus)

            # Apply filters if any
            if filters:
                for filter_key, filter_value in filters.items():
                    if filter_key == "job_id":
                        query = query.where(
                            BackendEvaluationJobStatus.job_id == filter_value
                        )
                    elif filter_key == "validator_hotkey":
                        query = query.where(
                            BackendEvaluationJobStatus.validator_hotkey == filter_value
                        )

            query = query.order_by(BackendEvaluationJobStatus.created_at.desc()).limit(
                limit
            )

            result = await session.execute(query)
            job_statuses = result.scalars().all()

            job_data = []
            for job_status in job_statuses:
                # Use JobStatusChangedEvent model
                event = JobStatusChangedEvent(
                    job_id=str(job_status.job_id),
                    validator_hotkey=job_status.validator_hotkey,
                    status=job_status.status.value,
                    detail=job_status.detail,
                    created_at=job_status.created_at,
                )
                job_data.append(event.model_dump(mode="json"))

            return job_data

        except Exception as e:
            logger.error(f"Error getting initial job status data: {str(e)}")
            return []

    async def _get_initial_job_completed_data(
        self, session, filters: Dict[str, Any], limit: int = INITIAL_STATE_DATA_LIMIT
    ) -> List[Dict[str, Any]]:
        """Get recently completed jobs (for JOB_COMPLETED subscriptions)."""
        try:
            # Find jobs that have completed status
            latest_status_subquery = (
                select(
                    BackendEvaluationJobStatus.job_id,
                    func.max(BackendEvaluationJobStatus.created_at).label(
                        "max_created_at"
                    ),
                )
                .group_by(BackendEvaluationJobStatus.job_id)
                .subquery()
            )

            # Join to get jobs with completed/failed/cancelled/timeout statuses
            query = (
                select(BackendEvaluationJob, BackendEvaluationJobStatus)
                .join(
                    BackendEvaluationJobStatus,
                    BackendEvaluationJob.id == BackendEvaluationJobStatus.job_id,
                )
                .join(
                    latest_status_subquery,
                    and_(
                        BackendEvaluationJobStatus.job_id
                        == latest_status_subquery.c.job_id,
                        BackendEvaluationJobStatus.created_at
                        == latest_status_subquery.c.max_created_at,
                    ),
                )
                .where(
                    BackendEvaluationJobStatus.status.in_(
                        [
                            EvaluationStatus.COMPLETED,
                            EvaluationStatus.FAILED,
                            EvaluationStatus.CANCELLED,
                            EvaluationStatus.TIMEOUT,
                        ]
                    )
                )
            )

            # Apply filters if any
            if filters:
                for filter_key, filter_value in filters.items():
                    if filter_key == "job_id":
                        query = query.where(BackendEvaluationJob.id == filter_value)
                    elif filter_key == "validator_hotkey":
                        query = query.where(
                            BackendEvaluationJobStatus.validator_hotkey == filter_value
                        )

            query = query.order_by(BackendEvaluationJobStatus.created_at.desc()).limit(
                limit
            )

            result = await session.execute(query)
            rows = result.all()

            job_data = []
            for job, status in rows:
                # Use JobCompletedEvent model
                event = JobCompletedEvent(
                    job_id=str(job.id),
                    validator_hotkey=status.validator_hotkey,
                    status=status.status.value,
                    detail=status.detail,
                    result_count=0,  # Could be enhanced to count results
                )
                job_data.append(event.model_dump(mode="json"))

            return job_data

        except Exception as e:
            logger.error(f"Error getting initial job completed data: {str(e)}")
            return []

    async def _get_initial_evaluation_data(
        self, session, filters: Dict[str, Any], limit: int = INITIAL_STATE_DATA_LIMIT
    ) -> List[Dict[str, Any]]:
        """Get recent evaluation results."""
        try:
            # BackendEvaluationResult is imported at module level

            query = select(BackendEvaluationResult)

            # Apply filters if any
            if filters:
                for filter_key, filter_value in filters.items():
                    if filter_key == "job_id":
                        query = query.where(
                            BackendEvaluationResult.job_id == filter_value
                        )
                    elif filter_key == "validator_hotkey":
                        query = query.where(
                            BackendEvaluationResult.validator_hotkey == filter_value
                        )
                    elif filter_key == "miner_hotkey":
                        query = query.where(
                            BackendEvaluationResult.miner_hotkey == filter_value
                        )
                    elif filter_key == "competition_id":
                        query = query.where(
                            BackendEvaluationResult.competition_id == filter_value
                        )

            query = query.order_by(BackendEvaluationResult.result_time.desc()).limit(
                limit
            )

            result = await session.execute(query)
            results = result.scalars().all()

            result_data = []
            for eval_result in results:
                # Use EvaluationCompletedEvent model
                event = EvaluationCompletedEvent(
                    job_id=str(eval_result.job_id),
                    validator_hotkey=eval_result.validator_hotkey,
                    miner_hotkey=eval_result.miner_hotkey,
                    competition_id=eval_result.competition_id,
                    benchmark_name=eval_result.benchmark_name or "",
                    score=eval_result.score,
                    success_rate=eval_result.success_rate,
                    avg_reward=eval_result.avg_reward,
                    total_episodes=eval_result.total_episodes,
                    result_time=eval_result.result_time,
                    created_at=eval_result.created_at or datetime.now(timezone.utc),
                )
                # Add extra fields from the database model
                data = event.model_dump(mode="json")
                result_data.append(data)

            return result_data

        except Exception as e:
            logger.error(f"Error getting initial evaluation data: {str(e)}")
            return []

    async def _get_initial_competition_data(
        self, session, filters: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Get active competitions."""
        try:
            # Competition is imported at module level

            query = select(Competition).where(Competition.active)

            # Apply filters if any
            if filters:
                for filter_key, filter_value in filters.items():
                    if filter_key == "competition_id":
                        query = query.where(Competition.id == filter_value)

            result = await session.execute(query)
            competitions = result.scalars().all()

            competition_data = []
            for competition in competitions:
                # Use CompetitionCreatedEvent model
                event = CompetitionCreatedEvent(
                    id=competition.id,
                    name=competition.name,
                    description=competition.description,
                    benchmarks=competition.benchmarks,
                    points=competition.points,
                    active=competition.active,
                    start_time=competition.start_time,
                    end_time=competition.end_time,
                    created_at=competition.created_at or datetime.now(timezone.utc),
                    updated_at=competition.updated_at or datetime.now(timezone.utc),
                )
                # Add extra fields
                data = event.model_dump(mode="json")
                competition_data.append(data)

            return competition_data

        except Exception as e:
            logger.error(f"Error getting initial competition data: {str(e)}")
            return []

    async def _get_initial_submission_data(
        self, session, filters: Dict[str, Any], limit: int = INITIAL_STATE_DATA_LIMIT
    ) -> List[Dict[str, Any]]:
        """Get recent miner submissions."""
        try:
            # MinerSubmission is imported at module level

            query = select(MinerSubmission)

            # Apply filters if any
            if filters:
                for filter_key, filter_value in filters.items():
                    if filter_key == "competition_id":
                        query = query.where(
                            MinerSubmission.competition_id == filter_value
                        )
                    elif filter_key == "miner_hotkey":
                        query = query.where(
                            MinerSubmission.miner_hotkey == filter_value
                        )
                    elif filter_key == "submission_id":
                        query = query.where(MinerSubmission.id == filter_value)

            query = query.order_by(MinerSubmission.created_at.desc()).limit(limit)

            result = await session.execute(query)
            submissions = result.scalars().all()

            submission_data = []
            for submission in submissions:
                # Use SubmissionReceivedEvent model
                event = SubmissionReceivedEvent(
                    submission_id=submission.id,
                    competition_id=submission.competition_id,
                    miner_hotkey=submission.miner_hotkey,
                    hf_repo_id=submission.hf_repo_id,
                    block_number=submission.block_number,
                    created_at=submission.created_at or datetime.now(timezone.utc),
                )
                # Add extra fields
                data = event.model_dump(mode="json")
                submission_data.append(data)

            return submission_data

        except Exception as e:
            logger.error(f"Error getting initial submission data: {str(e)}")
            return []

    async def _get_initial_episode_data(
        self, session, filters: Dict[str, Any], limit: int = INITIAL_STATE_DATA_LIMIT
    ) -> List[Dict[str, Any]]:
        """Get recent episode data."""
        try:
            # EpisodeData is imported at module level

            query = select(EpisodeData)

            # Apply filters if any
            if filters:
                for filter_key, filter_value in filters.items():
                    if filter_key == "job_id":
                        query = query.where(EpisodeData.job_id == filter_value)
                    elif filter_key == "submission_id":
                        query = query.where(EpisodeData.submission_id == filter_value)
                    elif filter_key == "episode_id":
                        query = query.where(EpisodeData.episode_id == filter_value)

            query = query.order_by(EpisodeData.created_at.desc()).limit(limit)

            result = await session.execute(query)
            episodes = result.scalars().all()

            episode_data = []
            for episode in episodes:
                # Use EpisodeCompletedEvent model
                event = EpisodeCompletedEvent(
                    job_id=str(episode.job_id),
                    submission_id=str(episode.submission_id),
                    episode_id=episode.episode_id,
                    env_name=episode.env_name,
                    benchmark_name=episode.benchmark_name,
                    total_reward=episode.total_reward,
                    success=episode.success,
                    steps=episode.steps,
                    start_time=episode.start_time or datetime.now(timezone.utc),
                    end_time=episode.end_time or datetime.now(timezone.utc),
                    extra_metrics=episode.extra_metrics,
                    created_at=episode.created_at or datetime.now(timezone.utc),
                )
                # Add extra fields
                data = event.model_dump(mode="json")
                episode_data.append(data)

            return episode_data

        except Exception as e:
            logger.error(f"Error getting initial episode data: {str(e)}")
            return []


# Global broadcaster instance
event_broadcaster = RealtimeEventBroadcaster()
