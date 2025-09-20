"""
WebSocket-based validator service for Kinitro.

This new validator architecture connects directly to the Kinitro Backend
via WebSocket and receives evaluation jobs from there
"""

import asyncio
import json
import os
from typing import Optional

import asyncpg
import dotenv
import websockets
from fiber import SubstrateInterface
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.chain.weights import set_node_weights
from pgqueuer import Job, PgQueuer, Queries
from pgqueuer.db import AsyncpgDriver
from websockets.exceptions import ConnectionClosed, WebSocketException

from core.log import get_logger
from core.messages import (
    EpisodeDataMessage,
    EpisodeStepDataMessage,
    EvalJobMessage,
    EvalResultMessage,
    HeartbeatMessage,
    MessageType,
    SetWeightsMessage,
    ValidatorRegisterMessage,
)
from core.neuron import Neuron

from .config import ValidatorConfig

dotenv.load_dotenv()

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

        # Database and pgqueue
        self.database_url = config.settings.get(
            "pg_database", "postgresql://myuser:mypassword@localhost/validatordb"
        )

        if self.database_url is None:
            raise Exception("Database URL not provided")

        self.q: Optional[Queries] = None

        # Chain connection objects
        self.substrate: SubstrateInterface = None
        self.metagraph: Metagraph = None
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

            logger.info("Result processor is now listening for evaluation results...")
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
        await self._send_message(result.model_dump())

    async def _send_episode_data(self, episode_data: EpisodeDataMessage):
        """Send episode data to the backend."""
        await self._send_message(episode_data.model_dump())

    async def _send_episode_step_data(self, step_data: EpisodeStepDataMessage):
        """Send episode step data to the backend."""
        await self._send_message(step_data.model_dump())

    async def _init_chain(self) -> None:
        """Initialize blockchain connection."""
        try:
            logger.info("Initializing blockchain connection...")

            self.substrate = get_substrate(
                subtensor_network=self.config.settings["subtensor"]["network"],
                subtensor_address=self.config.settings["subtensor"]["address"],
            )

            self.metagraph = Metagraph(
                netuid=self.config.settings["subtensor"]["netuid"],
                substrate=self.substrate,
            )

            # Sync metagraph to get our validator node_id
            self.metagraph.sync_nodes()

            # Get our validator node_id from the metagraph
            validator_node = self.metagraph.nodes.get(self.hotkey)
            if validator_node:
                self.validator_node_id = validator_node.node_id
                logger.info(f"Validator node_id: {self.validator_node_id}")
            else:
                logger.warning(f"Validator hotkey {self.hotkey} not found in metagraph")
                self.validator_node_id = None

            logger.info("Blockchain connection initialized")
        except Exception as e:
            logger.error(f"Failed to initialize blockchain connection: {e}")
            # Continue without chain connection - validator can still process jobs
            logger.warning("Continuing without blockchain connection")

    async def _handle_set_weights(self, weights_msg: SetWeightsMessage):
        """Handle weight setting message from backend.

        This function:
        1. Receives weights dict (UID->weight mapping) from the backend
        2. Syncs the metagraph to get latest chain state
        3. Sets the weights on chain using the validator's keypair
        """
        try:
            logger.info(
                f"Received weight update: {len(weights_msg.weights)} weights for miners {list(weights_msg.weights.keys())[:5]}..."
            )

            if not self.substrate or not self.metagraph:
                logger.error("Chain connection not initialized, cannot set weights")
                return

            # Sync metagraph to get latest state
            logger.info("Syncing metagraph before setting weights...")
            self.metagraph.sync_nodes()

            # Get validator node_id if not already set
            if self.validator_node_id is None:
                validator_node = self.metagraph.nodes.get(self.hotkey)
                if validator_node:
                    self.validator_node_id = validator_node.node_id
                else:
                    logger.error(
                        f"Validator hotkey {self.hotkey} not found in metagraph, cannot set weights"
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
            message_json = json.dumps(message, default=str)
            await self.websocket.send(message_json)
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            raise
