"""
WebSocket-based validator service for Kinitro.

This new validator architecture connects directly to the Kinitro Backend
via WebSocket and receives evaluation jobs from there
"""

import asyncio
import json
import random
from typing import Dict, Optional

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.log import get_logger
from core.messages import (
    EvalJobMessage,
    EvalResultMessage,
    HeartbeatMessage,
    ValidatorRegisterMessage,
)
from core.neuron import Neuron

from .config import ValidatorConfig

logger = get_logger(__name__)


class WebSocketValidator(Neuron):
    """
    WebSocket-based validator that connects to the Kinitro backend.
    """

    def __init__(self, config: ValidatorConfig):
        super().__init__(config)
        self.hotkey = self.keypair.ss58_address

        # Backend connection settings
        self.backend_url = config.settings.get(
            "backend_url", "ws://localhost:8080/ws/validator"
        )
        self.reconnect_interval = config.settings.get("reconnect_interval", 5)
        self.heartbeat_interval = config.settings.get("heartbeat_interval", 30)

        # Connection state
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connected = False
        self._running = False
        self._heartbeat_task = None

        # Job tracking
        self.active_jobs: Dict[str, EvalJobMessage] = {}

        logger.info(f"WebSocket Validator initialized for hotkey: {self.hotkey}")

    async def start(self):
        """Start the validator service."""
        logger.info("Starting WebSocket Validator")
        self._running = True

        # Connect to backend with auto-reconnect
        while self._running:
            try:
                await self._connect_to_backend()
                # Connection lost, wait before retry
                if self._running:
                    logger.warning(
                        f"Connection lost, reconnecting in {self.reconnect_interval}s"
                    )
                    await asyncio.sleep(self.reconnect_interval)
            except Exception as e:
                logger.error(f"Failed to connect to backend: {e}")
                if self._running:
                    await asyncio.sleep(self.reconnect_interval)

    async def stop(self):
        """Stop the validator service."""
        logger.info("Stopping WebSocket Validator")
        self._running = False

        # Cancel heartbeat
        if self._heartbeat_task:
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close WebSocket connection
        if self.websocket:
            await self.websocket.close()
            self.connected = False

        logger.info("WebSocket Validator stopped")

    async def _connect_to_backend(self):
        """Connect to backend and handle messages."""
        try:
            logger.info(f"Connecting to backend: {self.backend_url}")

            async with websockets.connect(
                self.backend_url,
                ping_interval=20,
                ping_timeout=10,
                close_timeout=10,
            ) as websocket:
                self.websocket = websocket

                # Register with backend
                await self._register()

                # Start heartbeat
                self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

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

    async def _register(self):
        """Register validator with backend."""
        register_msg = ValidatorRegisterMessage(hotkey=self.hotkey)
        await self._send_message(register_msg.model_dump())

        # Wait for acknowledgment
        response = await self.websocket.recv()
        ack = json.loads(response)

        if (
            ack.get("message_type") == "registration_ack"
            and ack.get("status") == "registered"
        ):
            self.connected = True
            logger.info("Successfully registered with backend")
        else:
            raise Exception(f"Registration failed: {ack}")

    async def _heartbeat_loop(self):
        """Send periodic heartbeats to backend."""
        try:
            while self.connected and not self._heartbeat_task.cancelled():
                # Send heartbeat with queue size
                queue_size = len(self.active_jobs)
                heartbeat = HeartbeatMessage(queue_size=queue_size)
                await self._send_message(heartbeat.model_dump())

                # Wait for heartbeat interval
                await asyncio.sleep(self.heartbeat_interval)

        except asyncio.CancelledError:
            logger.debug("Heartbeat task cancelled")
        except Exception as e:
            logger.error(f"Heartbeat error: {e}")

    async def _message_loop(self):
        """Handle incoming messages from backend."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("message_type")

                    # TODO: create handler for receiving and setting weights
                    if message_type == "eval_job":
                        await self._handle_eval_job(EvalJobMessage(**data))
                    elif message_type == "heartbeat_ack":
                        logger.debug("Received heartbeat ack")
                    elif message_type == "error":
                        logger.error(f"Backend error: {data.get('error')}")
                    else:
                        logger.warning(f"Unknown message type: {message_type}")

                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")

        except ConnectionClosed:
            logger.info("Message loop ended - connection closed")
        except Exception as e:
            logger.error(f"Message loop error: {e}")

    async def _handle_eval_job(self, job: EvalJobMessage):
        """Handle evaluation job from backend."""
        logger.info(
            f"Received evaluation job: {job.job_id} for miner {job.miner_hotkey}"
        )

        # Track active job
        self.active_jobs[job.job_id] = job

        try:
            # Simulate evaluation processing time
            await asyncio.sleep(random.uniform(1, 3))

            # Send dummy results back to backend for each benchmark
            for benchmark in job.benchmarks:
                # Generate dummy evaluation data
                dummy_score = random.uniform(0.1, 1.0)
                dummy_success_rate = random.uniform(0.5, 1.0)
                dummy_avg_reward = random.uniform(10, 100)
                dummy_episodes = random.randint(50, 200)

                result_msg = EvalResultMessage(
                    job_id=job.job_id,
                    validator_hotkey=self.hotkey,
                    miner_hotkey=job.miner_hotkey,
                    competition_id=job.competition_id,
                    benchmark=benchmark,
                    score=dummy_score,
                    success_rate=dummy_success_rate,
                    avg_reward=dummy_avg_reward,
                    total_episodes=dummy_episodes,
                    logs=f"Dummy evaluation completed for {benchmark}",
                    extra_data={
                        "processing_time": random.uniform(1, 3),
                        "validator_version": "1.0.0-dummy",
                    },
                )

                await self._send_message(result_msg.model_dump())
                logger.info(
                    f"Sent dummy result for job {job.job_id}, benchmark {benchmark} (score: {dummy_score:.3f})"
                )

        except Exception as e:
            logger.error(f"Failed to process job {job.job_id}: {e}")

            # Send error result for each benchmark
            for benchmark in job.benchmarks:
                error_msg = EvalResultMessage(
                    job_id=job.job_id,
                    validator_hotkey=self.hotkey,
                    miner_hotkey=job.miner_hotkey,
                    competition_id=job.competition_id,
                    benchmark=benchmark,
                    score=0.0,
                    error=str(e),
                )

                await self._send_message(error_msg.model_dump())

        finally:
            # Remove from active jobs
            if job.job_id in self.active_jobs:
                del self.active_jobs[job.job_id]

    async def _send_message(self, message: dict):
        """Send message to backend."""
        if not self.websocket:
            raise Exception("No WebSocket connection")

        try:
            message_json = json.dumps(message, default=str)
            await self.websocket.send(message_json)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise

    async def get_status(self) -> dict:
        """Get validator status information."""
        return {
            "hotkey": self.hotkey,
            "connected": self.connected,
            "backend_url": self.backend_url,
            "active_jobs": len(self.active_jobs),
        }
