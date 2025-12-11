"""
WebSocket-based validator service for Kinitro.

This new validator architecture connects directly to the Kinitro Backend
via WebSocket and receives evaluation jobs from there
"""

import asyncio
import json
import os
from typing import Dict, Optional

import asyncpg
import dotenv
import websockets
from fiber.chain.fetch_nodes import _get_nodes_for_uid
from fiber.chain.models import Node
from pgqueuer import Job, PgQueuer, Queries
from pgqueuer.db import AsyncpgDriver
from websockets.exceptions import ConnectionClosed, WebSocketException

from backend.models import SS58Address
from core.chain import set_node_weights
from core.log import get_logger
from core.messages import (
    EpisodeDataMessage,
    EpisodeStepDataMessage,
    EvalJobMessage,
    EvalResultMessage,
    HeartbeatMessage,
    JobStatusUpdateMessage,
    MessageType,
    SetWeightsMessage,
    ValidatorRegisterMessage,
)
from core.neuron import Neuron

from .config import ValidatorConfig

dotenv.load_dotenv()

logger = get_logger(__name__)

VALIDATOR_SEND_QUEUE_MAXSIZE = 1000
VALIDATOR_SEND_QUEUE_WARN_FRACTION = 0.8


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

        # Get API key from environment variable only
        self.api_key = os.environ.get("KINITRO_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Backend API key not provided. Set KINITRO_API_KEY environment variable"
            )

        # Connection state
        self.websocket: Optional[websockets.ServerConnection] = None
        self.connected = False
        self._running = False
        self._heartbeat_task = None
        self._result_processor_task = None
        self._send_queue: Optional[asyncio.Queue[dict]] = None
        self._sender_task: Optional[asyncio.Task] = None

        # Database and pgqueue
        self.database_url = config.settings.get(
            "pg_database", "postgresql://myuser:mypassword@localhost/validatordb"
        )

        if self.database_url is None:
            raise Exception("Database URL not provided")

        self.nodes: Optional[Dict[SS58Address, Node]] = None
        self.validator_node_id: int = None  # Our node ID on the chain

        logger.info(f"WebSocket Validator initialized for hotkey: {self.hotkey}")

    async def start(self):
        """Start the validator service."""
        logger.info("Starting WebSocket Validator")
        self._running = True

        # Initialize chain connection
        await self._init_chain()

        # Start the result processor task
        self._result_processor_task = asyncio.create_task(self._process_results())

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
            self._heartbeat_task = None

        if self._send_queue:
            try:
                self._send_queue.put_nowait(None)
            except asyncio.QueueFull:
                pass

        if self._sender_task:
            self._sender_task.cancel()
            try:
                await self._sender_task
            except asyncio.CancelledError:
                pass
            self._sender_task = None

        self._send_queue = None

        # Cancel result processor
        if self._result_processor_task:
            self._result_processor_task.cancel()
            try:
                await self._result_processor_task
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
                # NOTE: Disabled the library keepalive pings so we rely on our
                # application-level heartbeat rather than closing the
                # connection when the event loop is busy.
                ping_interval=None,
                ping_timeout=None,
                close_timeout=10,
            ) as websocket:
                self.websocket = websocket
                self._send_queue = asyncio.Queue(maxsize=VALIDATOR_SEND_QUEUE_MAXSIZE)
                self._sender_task = asyncio.create_task(self._sender_loop())

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
                try:
                    await self._heartbeat_task
                except asyncio.CancelledError:
                    pass
                self._heartbeat_task = None

            if self._send_queue:
                try:
                    self._send_queue.put_nowait(None)
                except asyncio.QueueFull:
                    pass

            if self._sender_task:
                self._sender_task.cancel()
                try:
                    await self._sender_task
                except asyncio.CancelledError:
                    pass
                self._sender_task = None

            self._send_queue = None

    async def _register(self):
        """Register validator with backend."""
        register_msg = ValidatorRegisterMessage(
            hotkey=self.hotkey, api_key=self.api_key
        )
        await self._send_message(register_msg.model_dump())

        # Wait for acknowledgment
        response = await self.websocket.recv()
        ack = json.loads(response)

        if (
            ack.get("message_type") == MessageType.REGISTRATION_ACK
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
                heartbeat = HeartbeatMessage()
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

                    if message_type == MessageType.EVAL_JOB:
                        await self._handle_eval_job(EvalJobMessage(**data))
                    elif message_type == MessageType.SET_WEIGHTS:
                        await self._handle_set_weights(SetWeightsMessage(**data))
                    elif message_type == MessageType.HEARTBEAT_ACK:
                        logger.debug("Received heartbeat ack")
                    elif message_type == MessageType.ERROR:
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
        # self.active_jobs[job.job_id] = job

        # Queue the job with pgqueuer to the database
        job_bytes = job.to_bytes()
        logger.info(f"Queueing job {job.job_id} to database")
        # Connect to the postgres database
        conn = await asyncpg.connect(dsn=self.database_url)
        driver = AsyncpgDriver(conn)
        q = Queries(driver)
        await q.enqueue(["add_job"], [job_bytes], [0])
        logger.info(f"Job {job.job_id} queued successfully")

    async def _process_results(self):
        """Process evaluation results from pgqueue and send them to the backend."""
        logger.info("Starting result processor task")

        try:
            # Connect to the postgres database
            conn = await asyncpg.connect(dsn=self.database_url)
            driver = AsyncpgDriver(conn)
            pgq = PgQueuer(driver)

            @pgq.entrypoint("eval_result")
            async def process_result(job: Job) -> None:
                """Process an evaluation result from the queue."""
                try:
                    # Parse the result from the job payload
                    result_data = json.loads(job.payload.decode("utf-8"))
                    eval_result = EvalResultMessage(**result_data)

                    logger.info(
                        f"Processing evaluation result for job {eval_result.job_id}"
                    )

                    # Send the result to the backend if connected
                    if self.connected and self.websocket:
                        await self._send_eval_result(eval_result)
                        logger.info(
                            f"Sent evaluation result for job {eval_result.job_id} to backend"
                        )
                    else:
                        # If not connected, the job will remain in queue and be retried
                        logger.warning(
                            f"Not connected to backend, result for job {eval_result.job_id} will be retried"
                        )
                        raise Exception("Not connected to backend")

                except Exception as e:
                    logger.error(f"Failed to process evaluation result: {e}")
                    # Re-raise to let pgqueue handle retry
                    raise

            @pgq.entrypoint("episode_data")
            async def process_episode_data(job: Job) -> None:
                """Process episode data from the queue."""
                try:
                    # Parse the episode data from the job payload
                    episode_data = json.loads(job.payload.decode("utf-8"))
                    episode_msg = EpisodeDataMessage(**episode_data)

                    logger.info(
                        f"Processing episode data for submission {episode_msg.submission_id}, episode {episode_msg.episode_id}"
                    )

                    # Send the episode data to the backend if connected
                    if self.connected and self.websocket:
                        await self._send_episode_data(episode_msg)
                        logger.info(
                            f"Sent episode data for episode {episode_msg.episode_id} to backend"
                        )
                    else:
                        # If not connected, the job will remain in queue and be retried
                        logger.warning(
                            f"Not connected to backend, episode data for episode {episode_msg.episode_id} will be retried"
                        )
                        raise Exception("Not connected to backend")

                except Exception as e:
                    logger.error(f"Failed to process episode data: {e}")
                    # Re-raise to let pgqueue handle retry
                    raise

            @pgq.entrypoint("episode_step_data")
            async def process_episode_step_data(job: Job) -> None:
                """Process episode step data from the queue."""
                try:
                    # Parse the step data from the job payload
                    step_data = json.loads(job.payload.decode("utf-8"))
                    step_msg = EpisodeStepDataMessage(**step_data)

                    logger.info(
                        f"Processing step data for submission {step_msg.submission_id}, episode {step_msg.episode_id}, step {step_msg.step}"
                    )

                    # Send the step data to the backend if connected
                    if self.connected and self.websocket:
                        await self._send_episode_step_data(step_msg)
                        logger.info(
                            f"Sent step data for episode {step_msg.episode_id}, step {step_msg.step} to backend"
                        )
                    else:
                        # If not connected, the job will remain in queue and be retried
                        logger.warning(
                            f"Not connected to backend, step data for episode {step_msg.episode_id}, step {step_msg.step} will be retried"
                        )
                        raise Exception("Not connected to backend")

                except Exception as e:
                    logger.error(f"Failed to process episode step data: {e}")
                    # Re-raise to let pgqueue handle retry
                    raise

            @pgq.entrypoint("job_status_update")
            async def process_job_status_update(job: Job) -> None:
                """Process job status updates from the queue."""
                try:
                    status_data = json.loads(job.payload.decode("utf-8"))
                    status_msg = JobStatusUpdateMessage(**status_data)

                    logger.info(
                        "Processing job status update for job %s: %s",
                        status_msg.job_id,
                        status_msg.status,
                    )

                    if self.connected and self.websocket:
                        await self._send_job_status_update(status_msg)
                        logger.info(
                            "Sent job status update for job %s to backend",
                            status_msg.job_id,
                        )
                    else:
                        logger.warning(
                            "Not connected to backend, job status update for job %s will be retried",
                            status_msg.job_id,
                        )
                        raise Exception("Not connected to backend")

                except Exception as e:
                    logger.error(f"Failed to process job status update: {e}")
                    raise

            logger.info(
                "Result processor is now listening for evaluation data and status updates..."
            )
            await pgq.run()

        except asyncio.CancelledError:
            logger.info("Result processor task cancelled")
            raise
        except Exception as e:
            logger.error(f"Result processor error: {e}")
            # Restart the processor after a delay if still running
            if self._running:
                await asyncio.sleep(5)
                self._result_processor_task = asyncio.create_task(
                    self._process_results()
                )

    async def _send_eval_result(self, result: EvalResultMessage):
        """Send evaluation result to the backend."""
        await self._send_message(result.model_dump(mode="json"))

    async def _send_job_status_update(self, status_update: JobStatusUpdateMessage):
        """Send job status update to the backend."""
        await self._send_message(status_update.model_dump(mode="json"))

    async def _send_episode_data(self, episode_data: EpisodeDataMessage):
        """Send episode data to the backend."""
        await self._send_message(episode_data.model_dump())

    async def _send_episode_step_data(self, step_data: EpisodeStepDataMessage):
        """Send episode step data to the backend."""
        await self._send_message(step_data.model_dump())

    async def _sender_loop(self) -> None:
        """Continuously flush the outbound queue to the backend."""

        if not self._send_queue:
            return

        try:
            while True:
                message = await self._send_queue.get()

                if message is None:
                    self._send_queue.task_done()
                    break

                try:
                    if not self.websocket:
                        raise RuntimeError("WebSocket connection unavailable")

                    message_json = json.dumps(message, default=str)
                    await self.websocket.send(message_json)
                except Exception as exc:
                    logger.error(f"Outbound sender error: {exc}")
                    raise
                finally:
                    self._send_queue.task_done()

        except asyncio.CancelledError:
            logger.error("Outbound sender task cancelled")
            raise
        except Exception:
            if self.websocket:
                try:
                    await self.websocket.close()
                except Exception:
                    logger.error("Error closing WebSocket connection")
        finally:
            if self._send_queue:
                while not self._send_queue.empty():
                    try:
                        self._send_queue.get_nowait()
                        self._send_queue.task_done()
                    except asyncio.QueueEmpty:
                        logger.warning("Outbound queue already empty")
                        break

    async def _init_chain(self) -> None:
        """Initialize blockchain info."""
        try:
            logger.info("Getting nodes from chain...")

            # Sync nodes from chain
            await self._sync_nodes()

            # Get our validator node_id from the nodes
            validator_node = self.nodes.get(self.hotkey) if self.nodes else None
            if validator_node:
                self.validator_node_id = validator_node.node_id
                logger.info(f"Validator node_id: {self.validator_node_id}")
            else:
                logger.warning(f"Validator hotkey {self.hotkey} not found in nodes")
                self.validator_node_id = None

            logger.info("Blockchain connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connection: {e}")
            # Continue without chain connection - validator can still process jobs
            logger.warning("Continuing without blockchain connection")

    async def _sync_nodes(self) -> None:
        """Sync nodes from the chain."""
        try:
            # Run in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            node_list = await loop.run_in_executor(
                None,
                _get_nodes_for_uid,
                self.substrate,
                self.config.settings["subtensor"]["netuid"],
            )
            self.nodes = {node.hotkey: node for node in node_list}
            logger.info(f"Synced {len(self.nodes)} nodes")
        except Exception as e:
            logger.error(f"Failed to sync nodes: {e}")
            if not self.nodes:
                self.nodes = {}

    async def _handle_set_weights(self, weights_msg: SetWeightsMessage):
        """Handle weight setting message from backend.

        This function:
        1. Receives weights dict (UID->weight mapping) from the backend
        2. Syncs the nodes to get latest chain state
        3. Sets the weights on chain using the validator's keypair
        """
        try:
            logger.info(
                f"Received weight update: {len(weights_msg.weights)} weights for miners {list(weights_msg.weights.keys())[:5]}..."
            )

            if not self.substrate or not self.nodes:
                logger.error("Chain connection not initialized, cannot set weights")
                return

            # Sync nodes to get latest state
            logger.info("Syncing nodes before setting weights...")
            await self._sync_nodes()

            # Get validator node_id if not already set
            if self.validator_node_id is None:
                validator_node = self.nodes.get(self.hotkey) if self.nodes else None
                if validator_node:
                    self.validator_node_id = validator_node.node_id
                else:
                    logger.error(
                        f"Validator hotkey {self.hotkey} not found in nodes, cannot set weights"
                    )
                    return

            # Extract node_ids and weights as parallel lists for the set_node_weights function
            node_ids = list(weights_msg.weights.keys())
            node_weights = list(weights_msg.weights.values())

            # Set weights on chain
            logger.info(f"Setting weights on chain for {len(node_ids)} miners")
            success = set_node_weights(
                substrate=self.substrate,
                keypair=self.keypair,
                node_ids=node_ids,
                node_weights=node_weights,
                netuid=self.config.settings["subtensor"]["netuid"],
                validator_node_id=self.validator_node_id,
                version_key=0,
                wait_for_inclusion=True,
                wait_for_finalization=False,
            )

            if success:
                logger.info(
                    f"Successfully set weights on chain for {len(node_ids)} miners"
                )
            else:
                logger.error("Failed to set weights on chain")

        except Exception as e:
            logger.error(f"Error handling set weights message: {e}")

    async def _send_message(self, message: dict):
        """Send message to backend."""
        if not self.websocket:
            raise Exception("No WebSocket connection")

        try:
            if not self._send_queue:
                raise Exception("Outbound queue not initialized")

            queue_size = self._send_queue.qsize()
            if queue_size > int(
                VALIDATOR_SEND_QUEUE_MAXSIZE * VALIDATOR_SEND_QUEUE_WARN_FRACTION
            ):
                logger.warning(
                    "Outbound queue size high (%s/%s)",
                    queue_size,
                    VALIDATOR_SEND_QUEUE_MAXSIZE,
                )

            self._send_queue.put_nowait(message)
        except asyncio.QueueFull:
            logger.error("Outbound message queue full; dropping message")
            raise
        except Exception as e:
            logger.error(f"Failed to enqueue message: {e}")
            raise
