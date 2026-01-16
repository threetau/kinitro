"""
WebSocket client for evaluator to communicate directly with backend.

This module provides the BackendClient class that enables evaluators to
connect directly to the backend without going through the validator relay.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.log import get_logger
from core.messages import (
    EpisodeDataMessage,
    EpisodeStepDataMessage,
    EvalJobMessage,
    EvalResultMessage,
    EvaluatorRegisterMessage,
    HeartbeatMessage,
    JobAckMessage,
    JobStatusUpdateMessage,
    MessageType,
)

logger = get_logger(__name__)

SEND_QUEUE_MAXSIZE = 1000
SEND_QUEUE_WARN_FRACTION = 0.8


class BackendClient:
    """
    WebSocket client for evaluator to communicate with backend.

    This enables direct communication between evaluators and the backend,
    eliminating the need for a validator relay.

    Features:
    - Automatic reconnection with exponential backoff
    - Send queue with backpressure handling
    - Heartbeat keepalive
    - Job acknowledgment support
    """

    def __init__(
        self,
        backend_url: str,
        evaluator_id: str,
        api_key: Optional[str] = None,
        supported_task_types: Optional[List[str]] = None,
        max_concurrent_jobs: int = 1,
        capabilities: Optional[Dict[str, Any]] = None,
        on_job_received: Optional[Callable[[EvalJobMessage], Awaitable[None]]] = None,
        reconnect_interval: float = 5.0,
        max_reconnect_interval: float = 60.0,
        heartbeat_interval: float = 30.0,
    ):
        """
        Initialize the backend client.

        Args:
            backend_url: WebSocket URL for backend connection (e.g., ws://backend:8080/ws/evaluator)
            evaluator_id: Unique identifier for this evaluator instance
            api_key: API key for authentication (defaults to KINITRO_API_KEY env var)
            supported_task_types: List of task types this evaluator supports
            max_concurrent_jobs: Maximum concurrent jobs this evaluator can handle
            capabilities: Additional metadata about evaluator capabilities (GPU, memory, etc.)
            on_job_received: Callback for when a job is received
            reconnect_interval: Initial reconnection interval in seconds
            max_reconnect_interval: Maximum reconnection interval in seconds
            heartbeat_interval: Interval between heartbeat messages in seconds
        """
        self.backend_url = backend_url
        self.evaluator_id = evaluator_id
        self.api_key = api_key or os.environ.get("KINITRO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not provided. Set KINITRO_API_KEY environment variable "
                "or pass api_key parameter"
            )

        self.supported_task_types = supported_task_types or ["rl_rollout"]
        self.max_concurrent_jobs = max_concurrent_jobs
        self.capabilities = capabilities

        # Callback
        self.on_job_received = on_job_received

        # Connection settings
        self.reconnect_interval = reconnect_interval
        self.max_reconnect_interval = max_reconnect_interval
        self.heartbeat_interval = heartbeat_interval

        # Connection state
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.connected = False
        self._running = False
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._send_queue: Optional[asyncio.Queue[Optional[dict]]] = None

        # Reconnect backoff state
        self._current_reconnect_interval = reconnect_interval

        logger.info(
            f"BackendClient initialized for evaluator {evaluator_id}, "
            f"connecting to {backend_url}"
        )

    async def connect_and_run(self) -> None:
        """
        Connect to backend and run the message loop.

        This method will automatically reconnect on connection loss.
        It runs until stop() is called.
        """
        logger.info(f"Starting BackendClient for evaluator {self.evaluator_id}")
        self._running = True
        self._current_reconnect_interval = self.reconnect_interval

        while self._running:
            try:
                await self._connect_to_backend()

                # Connection lost, wait before retry with backoff
                if self._running:
                    logger.warning(
                        f"Connection lost, reconnecting in {self._current_reconnect_interval:.1f}s"
                    )
                    await asyncio.sleep(self._current_reconnect_interval)
                    # Exponential backoff
                    self._current_reconnect_interval = min(
                        self._current_reconnect_interval * 1.5,
                        self.max_reconnect_interval,
                    )
            except Exception as e:
                logger.error(f"Failed to connect to backend: {e}")
                if self._running:
                    await asyncio.sleep(self._current_reconnect_interval)
                    self._current_reconnect_interval = min(
                        self._current_reconnect_interval * 1.5,
                        self.max_reconnect_interval,
                    )

    async def stop(self) -> None:
        """Stop the client and close the connection."""
        logger.info(f"Stopping BackendClient for evaluator {self.evaluator_id}")
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass
            self._heartbeat_task = None

        # Signal sender to stop
        if self._send_queue:
            try:
                self._send_queue.put_nowait(None)  # Sentinel to stop sender
            except asyncio.QueueFull:
                pass

        # Cancel sender
        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None

        self._send_queue = None

        # Close WebSocket
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception as e:
                logger.debug(f"Error closing WebSocket: {e}")
            self.websocket = None
            self.connected = False

        logger.info(f"BackendClient stopped for evaluator {self.evaluator_id}")

    async def send_result(self, result: EvalResultMessage) -> bool:
        """
        Send an evaluation result to the backend.

        Args:
            result: The evaluation result message

        Returns:
            True if queued successfully
        """
        # Use mode='json' to properly serialize enums to their values
        return await self._queue_message(result.model_dump(mode="json"))

    async def send_status_update(self, status: JobStatusUpdateMessage) -> bool:
        """
        Send a job status update to the backend.

        Args:
            status: The status update message

        Returns:
            True if queued successfully
        """
        # Use mode='json' to properly serialize enums to their values
        return await self._queue_message(status.model_dump(mode="json"))

    async def send_episode_data(self, data: EpisodeDataMessage) -> bool:
        """
        Send episode telemetry data to the backend.

        Args:
            data: The episode data message

        Returns:
            True if queued successfully
        """
        # Use mode='json' to properly serialize enums to their values
        return await self._queue_message(data.model_dump(mode="json"))

    async def send_episode_step_data(self, data: EpisodeStepDataMessage) -> bool:
        """
        Send episode step data to the backend.

        Args:
            data: The episode step data message

        Returns:
            True if queued successfully
        """
        # Use mode='json' to properly serialize enums to their values
        return await self._queue_message(data.model_dump(mode="json"))

    async def send_job_ack(
        self, job_id: int, accepted: bool, reason: Optional[str] = None
    ) -> bool:
        """
        Send a job acknowledgment to the backend.

        Args:
            job_id: The job ID being acknowledged
            accepted: Whether the job was accepted
            reason: Optional reason if not accepted

        Returns:
            True if queued successfully
        """
        ack = JobAckMessage(
            job_id=job_id,
            accepted=accepted,
            reason=reason,
        )
        return await self._queue_message(ack.model_dump())

    async def _connect_to_backend(self) -> None:
        """Connect to backend and handle messages."""
        try:
            logger.info(f"Connecting to backend: {self.backend_url}")

            async with websockets.connect(
                self.backend_url,
                ping_interval=None,  # We use application-level heartbeat
                ping_timeout=None,
                close_timeout=10,
            ) as websocket:
                self.websocket = websocket
                self._send_queue = asyncio.Queue(maxsize=SEND_QUEUE_MAXSIZE)
                self._sender_task = asyncio.create_task(self._sender_loop())

                # Register with backend
                await self._register()

                # Start heartbeat
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

                # Reset reconnect interval on successful connection
                self._current_reconnect_interval = self.reconnect_interval

                # Handle messages
                await self._message_loop()

        except ConnectionClosed:
            logger.warning("Backend connection closed")
        except WebSocketException as e:
            logger.error(f"WebSocket error: {e}")
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            self.connected = False
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None

    async def _register(self) -> None:
        """Register with the backend."""
        if not self.websocket:
            raise RuntimeError("Not connected to backend")

        register_msg = EvaluatorRegisterMessage(
            evaluator_id=self.evaluator_id,
            api_key=self.api_key,
            supported_task_types=self.supported_task_types,
            max_concurrent_jobs=self.max_concurrent_jobs,
            capabilities=self.capabilities,
        )

        await self.websocket.send(register_msg.model_dump_json())

        # Wait for acknowledgment
        response = await self.websocket.recv()
        data = json.loads(response)

        if data.get("message_type") == MessageType.EVALUATOR_REGISTRATION_ACK:
            if data.get("success"):
                self.connected = True
                logger.info(
                    f"Evaluator {self.evaluator_id} registered successfully with backend"
                )
            else:
                error = data.get("error", "Unknown error")
                raise RuntimeError(f"Registration failed: {error}")
        else:
            raise RuntimeError(f"Unexpected registration response: {data}")

    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeat messages."""
        try:
            while self._running and self.connected:
                await asyncio.sleep(self.heartbeat_interval)

                if self.websocket and self.connected:
                    heartbeat = HeartbeatMessage()
                    await self._queue_message(heartbeat.model_dump())
                    logger.debug(f"Sent heartbeat from evaluator {self.evaluator_id}")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in heartbeat loop: {e}")

    async def _sender_loop(self) -> None:
        """Background task to send queued messages."""
        try:
            while self._running:
                if not self._send_queue:
                    break

                message = await self._send_queue.get()

                # Sentinel value to stop
                if message is None:
                    break

                if self.websocket and self.connected:
                    try:
                        await self.websocket.send(json.dumps(message, default=str))
                    except Exception as e:
                        logger.error(f"Error sending message: {e}")
                        # Re-queue the message if still running
                        if self._running and self._send_queue:
                            try:
                                self._send_queue.put_nowait(message)
                            except asyncio.QueueFull:
                                logger.warning("Send queue full, dropping message")

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in sender loop: {e}")

    async def _message_loop(self) -> None:
        """Handle incoming messages from backend."""
        if not self.websocket:
            return

        try:
            async for message in self.websocket:
                data = json.loads(message)
                message_type = data.get("message_type")

                if message_type == MessageType.HEARTBEAT_ACK:
                    logger.debug("Received heartbeat ack from backend")

                elif message_type == MessageType.EVAL_JOB:
                    # Received a new job
                    job_msg = EvalJobMessage(**data)
                    logger.info(
                        f"Received job {job_msg.job_id} from backend for "
                        f"competition {job_msg.competition_id}"
                    )

                    # Call the job handler if set
                    if self.on_job_received:
                        try:
                            await self.on_job_received(job_msg)
                            # Acknowledge job acceptance
                            await self.send_job_ack(job_msg.job_id, accepted=True)
                        except Exception as e:
                            logger.error(f"Error handling job {job_msg.job_id}: {e}")
                            await self.send_job_ack(
                                job_msg.job_id,
                                accepted=False,
                                reason=str(e),
                            )
                    else:
                        logger.warning(
                            f"Received job {job_msg.job_id} but no handler registered"
                        )
                        await self.send_job_ack(
                            job_msg.job_id,
                            accepted=False,
                            reason="No job handler registered",
                        )

                elif message_type == MessageType.RESULT_ACK:
                    job_id = data.get("job_id")
                    logger.debug(f"Backend acknowledged result for job {job_id}")

                elif message_type == MessageType.ERROR:
                    error = data.get("error", "Unknown error")
                    details = data.get("details")
                    logger.error(f"Received error from backend: {error} - {details}")

                else:
                    logger.debug(f"Received unknown message type: {message_type}")

        except ConnectionClosed:
            logger.warning("Connection closed during message loop")
            raise
        except Exception as e:
            logger.error(f"Error in message loop: {e}")
            raise

    async def _queue_message(self, message: dict) -> bool:
        """
        Queue a message for sending.

        Args:
            message: The message to queue

        Returns:
            True if queued successfully
        """
        if not self._send_queue:
            logger.warning("Send queue not initialized, dropping message")
            return False

        try:
            # Check queue health
            queue_size = self._send_queue.qsize()
            if queue_size > SEND_QUEUE_MAXSIZE * SEND_QUEUE_WARN_FRACTION:
                logger.warning(f"Send queue is {queue_size}/{SEND_QUEUE_MAXSIZE} full")

            self._send_queue.put_nowait(message)
            return True
        except asyncio.QueueFull:
            logger.error("Send queue full, dropping message")
            return False

    @property
    def is_connected(self) -> bool:
        """Check if the client is connected to the backend."""
        return self.connected and self.websocket is not None
