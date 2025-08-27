"""Child validator job receiver for connecting to parent and receiving jobs."""

import asyncio
import json
import uuid
from typing import Awaitable, Callable, Optional

import websockets
from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from core.log import get_logger
from core.schemas import ChainCommitmentResponse

from .job_distribution import (
    ChildRegisterMessage,
    HeartbeatMessage,
    JobAssignmentMessage,
    MessageType,
    StatusUpdateMessage,
    ValidatorMessage,
)

logger = get_logger(__name__)


class ChildValidatorReceiver:
    """Manages WebSocket connection to parent validator and receives job assignments."""

    def __init__(
        self,
        parent_host: str,
        parent_port: int,
        validator_hotkey: str,
        job_handler: Callable[[ChainCommitmentResponse], Awaitable[bool]],
    ):
        self.parent_host = parent_host
        self.parent_port = parent_port
        self.validator_hotkey = validator_hotkey
        self.job_handler = job_handler
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self._running = False
        self._reconnect_task = None
        self._heartbeat_task = None
        self.queue_size = 0

    async def start(self):
        """Start the child validator receiver and connect to parent."""
        logger.info(
            f"Starting child validator receiver, connecting to {self.parent_host}:{self.parent_port}"
        )
        self._running = True
        self._reconnect_task = asyncio.create_task(self._connection_manager())

    async def stop(self):
        """Stop the child validator receiver and disconnect from parent."""
        logger.info("Stopping child validator receiver")
        self._running = False

        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        if self._reconnect_task:
            self._reconnect_task.cancel()
            try:
                await self._reconnect_task
            except asyncio.CancelledError:
                pass

        if self.websocket:
            await self.websocket.close()

        logger.info("Child validator receiver stopped")

    async def _connection_manager(self):
        """Manage connection to parent validator with automatic reconnection."""
        while self._running:
            try:
                await self._connect_to_parent()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection error: {e}")
                if self._running:
                    logger.info("Reconnecting in 10 seconds...")
                    await asyncio.sleep(10)

    async def _connect_to_parent(self):
        """Establish connection to parent validator."""
        uri = f"ws://{self.parent_host}:{self.parent_port}"
        logger.info(f"Connecting to parent validator at {uri}")

        try:
            async with websockets.connect(
                uri,
                ping_interval=30,
                ping_timeout=10,
            ) as websocket:
                self.websocket = websocket
                logger.info("Connected to parent validator")

                # Register with parent
                await self._register_with_parent()

                # Start heartbeat
                self._heartbeat_task = asyncio.create_task(self._send_heartbeats())

                # Listen for messages
                async for message in websocket:
                    await self._process_message(message)

        except (ConnectionClosedError, ConnectionClosedOK):
            logger.info("Disconnected from parent validator")
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            raise
        finally:
            if self._heartbeat_task:
                self._heartbeat_task.cancel()
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
            self.websocket = None

    async def _register_with_parent(self):
        """Send registration message to parent validator."""
        register_msg = ChildRegisterMessage(
            sender_id=self.validator_hotkey,
            message_id=str(uuid.uuid4()),
            validator_hotkey=self.validator_hotkey,
        )

        await self._send_message(register_msg)
        logger.info(f"Sent registration message for validator {self.validator_hotkey}")

    async def _send_heartbeats(self):
        """Send periodic heartbeat messages to parent validator."""
        while self._running and self.websocket:
            try:
                heartbeat_msg = HeartbeatMessage(
                    sender_id=self.validator_hotkey,
                    message_id=str(uuid.uuid4()),
                    queue_size=self.queue_size,
                )

                await self._send_message(heartbeat_msg)
                logger.debug("Sent heartbeat to parent validator")
                await asyncio.sleep(30)  # Send heartbeat every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error sending heartbeat: {e}")
                break

    async def _process_message(self, raw_message: str):
        """Process incoming messages from parent validator."""
        try:
            message_data = json.loads(raw_message)
            message_type = MessageType(message_data.get("message_type"))

            if message_type == MessageType.JOB_ASSIGNMENT:
                await self._handle_job_assignment(message_data)
            elif message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat(message_data)
            else:
                logger.debug(f"Received message type: {message_type}")

        except Exception as e:
            logger.error(f"Error processing message: {e}")

    async def _handle_job_assignment(self, message_data: dict):
        """Handle job assignment from parent validator."""
        try:
            job_msg = JobAssignmentMessage.model_validate(message_data)
            logger.info(f"Received job assignment: {job_msg.message_id}")

            # Send status update that we received the job
            await self._send_status_update(
                job_msg.message_id, "received", "Job assignment received and queued"
            )

            # Handle the job using the provided handler
            if self.job_handler:
                try:
                    success = await self.job_handler(job_msg.commitment)

                    if success:
                        self.queue_size += 1  # Increment queue size
                        logger.info(f"Job queued successfully: {job_msg.message_id}")

                        # Send status update that job was queued
                        await self._send_status_update(
                            job_msg.message_id, "queued", "Job added to local queue"
                        )
                    else:
                        logger.warning(
                            f"Job handler returned False for job {job_msg.message_id}"
                        )
                        await self._send_status_update(
                            job_msg.message_id,
                            "failed",
                            "Job handler failed to queue job",
                        )

                except Exception as e:
                    logger.error(f"Error handling job {job_msg.message_id}: {e}")
                    await self._send_status_update(
                        job_msg.message_id, "failed", f"Error queuing job: {str(e)}"
                    )

        except Exception as e:
            logger.error(f"Error handling job assignment: {e}")

    async def _handle_heartbeat(self, message_data: dict):
        """Handle heartbeat messages from parent validator."""
        try:
            heartbeat_msg = HeartbeatMessage.model_validate(message_data)
            logger.debug(f"Received heartbeat from parent: {heartbeat_msg.status}")
        except Exception as e:
            logger.error(f"Error handling heartbeat: {e}")

    async def _send_status_update(self, job_id: str, status: str, details: str):
        """Send status update to parent validator."""
        try:
            status_msg = StatusUpdateMessage(
                sender_id=self.validator_hotkey,
                message_id=str(uuid.uuid4()),
                job_id=job_id,
                status=status,
                details=details,
            )

            await self._send_message(status_msg)
            logger.debug(f"Sent status update for job {job_id}: {status}")

        except Exception as e:
            logger.error(f"Error sending status update: {e}")

    async def _send_message(self, message: ValidatorMessage):
        """Send a message to the parent validator."""
        if not self.websocket:
            raise RuntimeError("No connection to parent validator")

        try:
            message_json = message.model_dump_json()
            await self.websocket.send(message_json)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    def update_queue_size(self, size: int):
        """Update the current queue size."""
        self.queue_size = size

    def is_connected(self) -> bool:
        """Check if connected to parent validator."""
        return self.websocket is not None and not self.websocket.closed
