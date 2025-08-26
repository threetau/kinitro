import asyncio
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

import websockets
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from pydantic import BaseModel, Field
from snowflake import SnowflakeGenerator
from sqlalchemy import and_, select
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from websockets.server import WebSocketServerProtocol

from core.chain import query_commitments_from_substrate
from core.log import get_logger
from core.schemas import ChainCommitmentResponse

from .config import BackendConfig
from .models import (
    BackendEvaluationJob,
    BackendEvaluationResult,
    BackendState,
    Competition,
    MinerSubmission,
    ValidatorConnection,
)

logger = get_logger(__name__)


class EvalJobMessage(BaseModel):
    """Message for broadcasting evaluation jobs to validators."""

    message_type: str = "eval_job"
    job_id: str
    competition_id: str
    miner_hotkey: str
    hf_repo_id: str
    benchmarks: List[str]
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class EvalResultMessage(BaseModel):
    """Message for receiving evaluation results from validators."""

    message_type: str = "eval_result"
    job_id: str
    validator_hotkey: str
    miner_hotkey: str
    competition_id: str
    benchmark: str
    score: float
    success_rate: Optional[float] = None
    avg_reward: Optional[float] = None
    total_episodes: Optional[int] = None
    logs: Optional[str] = None
    error: Optional[str] = None
    extra_data: Optional[dict] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ValidatorRegisterMessage(BaseModel):
    """Message for validator registration."""

    message_type: str = "register"
    hotkey: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class KinitroBackend:
    """
    Central backend service that coordinates competitions, miner submissions,
    and validator evaluations.
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self.db_url = config.settings.get("database_url")
        self.ws_host = config.settings.get("websocket_host", "0.0.0.0")
        self.ws_port = config.settings.get("websocket_port", 8080)

        # WebSocket connections tracking (in-memory)
        self.websocket_connections: Dict[str, WebSocketServerProtocol] = {}

        # Chain monitoring configuration
        self.max_commitment_lookback: int = config.settings.get(
            "max_commitment_lookback", 360
        )
        self.chain_sync_interval: int = config.settings.get("chain_sync_interval", 30)
        self.min_stake_threshold: float = config.settings.get(
            "min_stake_threshold", 0.0
        )

        # Chain connection objects
        self.substrate = None
        self.metagraph = None

        # Background tasks
        self.ws_server = None
        self._running = False
        self._chain_monitor_task = None
        self._heartbeat_monitor_task = None

        # ID generator
        self.id_generator = SnowflakeGenerator(42)

        # Database engine and session
        self.engine = None
        self.async_session = None

    async def start(self):
        """Start the backend service."""
        logger.info("Starting Kinitro Backend Service")

        # Initialize chain connection
        await self._init_chain()

        # Initialize database connection
        await self._init_database()

        # Load or initialize backend state
        await self._load_backend_state()

        # Start WebSocket server for validator connections
        await self._start_websocket_server()

        # Start background tasks
        self._running = True
        self._chain_monitor_task = asyncio.create_task(self._monitor_chain())
        self._heartbeat_monitor_task = asyncio.create_task(
            self._monitor_validator_heartbeats()
        )

        logger.info("Kinitro Backend Service started successfully")

    async def stop(self):
        """Stop the backend service."""
        logger.info("Stopping Kinitro Backend Service")

        self._running = False

        # Cancel background tasks
        if self._chain_monitor_task:
            self._chain_monitor_task.cancel()
            try:
                await self._chain_monitor_task
            except asyncio.CancelledError:
                pass

        if self._heartbeat_monitor_task:
            self._heartbeat_monitor_task.cancel()
            try:
                await self._heartbeat_monitor_task
            except asyncio.CancelledError:
                pass

        # Stop WebSocket server
        if self.ws_server:
            self.ws_server.close()
            await self.ws_server.wait_closed()

        # Close all validator connections
        for ws in self.websocket_connections.values():
            await ws.close()

        # Close database engine
        if self.engine:
            await self.engine.dispose()

        logger.info("Kinitro Backend Service stopped")

    async def _init_chain(self):
        """Initialize blockchain connection and metagraph."""
        try:
            logger.info("Initializing blockchain connection...")

            # Get substrate connection
            self.substrate = get_substrate(
                subtensor_network=self.config.settings["subtensor_network"],
                subtensor_address=self.config.settings["subtensor_address"],
            )

            # Create metagraph
            self.metagraph = Metagraph(
                netuid=self.config.settings["netuid"],
                substrate=self.substrate,
            )

            logger.info("Blockchain connection initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize blockchain connection: {e}")
            raise

    async def _init_database(self):
        """Initialize database connection."""
        try:
            # Create async engine
            self.engine = create_async_engine(
                self.db_url,
                echo=False,
                pool_pre_ping=True,
                pool_size=20,
                max_overflow=0,
            )

            # Create async session factory
            self.async_session = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )

            logger.info("Database connection initialized")

        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            raise

    async def _load_backend_state(self):
        """Load or initialize backend state from database."""
        try:
            async with self.async_session() as session:
                # Get or create backend state (singleton)
                result = await session.execute(
                    select(BackendState).where(BackendState.id == 1)
                )
                state = result.scalar_one_or_none()

                if not state:
                    # Initialize new state
                    state = BackendState(
                        id=1, last_seen_block=0, service_version="1.0.0"
                    )
                    session.add(state)
                    await session.commit()
                    logger.info("Initialized new backend state")
                else:
                    # Update service start time
                    state.service_start_time = datetime.now(timezone.utc)
                    await session.commit()
                    logger.info(
                        f"Loaded backend state: last_seen_block={state.last_seen_block}"
                    )
        except Exception as e:
            logger.error(f"Failed to load backend state: {e}")
            raise

    async def create_competition(
        self,
        name: str,
        benchmarks: List[str],
        points: int,
        description: Optional[str] = None,
    ) -> Competition:
        """Create a new competition."""
        try:
            async with self.async_session() as session:
                competition = Competition(
                    id=str(uuid.uuid4()),
                    name=name,
                    benchmarks=benchmarks,
                    points=points,
                    description=description,
                    active=True,
                )

                session.add(competition)
                await session.commit()
                await session.refresh(competition)

                logger.info(
                    f"Created competition: {competition.name} (ID: {competition.id})"
                )
                return competition

        except Exception as e:
            logger.error(f"Failed to create competition: {e}")
            raise

    async def get_active_competitions(self) -> List[Competition]:
        """Get all active competitions."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(Competition).where(Competition.active)
                )
                return result.scalars().all()

        except Exception as e:
            logger.error(f"Failed to get active competitions: {e}")
            return []

    async def _start_websocket_server(self):
        """Start WebSocket server for validator connections."""
        logger.info(f"Starting WebSocket server on {self.ws_host}:{self.ws_port}")

        self.ws_server = await websockets.serve(
            self._handle_validator_connection,
            self.ws_host,
            self.ws_port,
            ping_interval=30,
            ping_timeout=10,
        )

        logger.info("WebSocket server started")

    async def _handle_validator_connection(self, websocket: WebSocketServerProtocol):
        """Handle incoming validator WebSocket connections."""
        connection_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"New validator connection from {connection_id}")

        validator_hotkey = None

        try:
            async for message in websocket:
                msg_data = json.loads(message)
                message_type = msg_data.get("message_type")

                if message_type == "register" and not validator_hotkey:
                    # Handle registration
                    validator_hotkey = await self._handle_validator_registration(
                        websocket, connection_id, msg_data
                    )
                elif validator_hotkey:
                    # Handle other messages only after registration
                    await self._process_validator_message(
                        websocket, connection_id, validator_hotkey, msg_data
                    )
                else:
                    logger.warning(
                        f"Message from unregistered validator: {connection_id}"
                    )

        except Exception as e:
            logger.error(f"Error handling validator connection {connection_id}: {e}")
        finally:
            if validator_hotkey:
                await self._handle_validator_disconnect(validator_hotkey, connection_id)
            if connection_id in self.websocket_connections:
                del self.websocket_connections[connection_id]

    async def _handle_validator_registration(
        self, websocket: WebSocketServerProtocol, connection_id: str, message: dict
    ) -> Optional[str]:
        """Handle validator registration."""
        try:
            reg_msg = ValidatorRegisterMessage(**message)

            async with self.async_session() as session:
                # Get or create validator connection record
                result = await session.execute(
                    select(ValidatorConnection).where(
                        ValidatorConnection.validator_hotkey == reg_msg.hotkey
                    )
                )
                validator_conn = result.scalar_one_or_none()

                if not validator_conn:
                    validator_conn = ValidatorConnection(
                        id=next(self.id_generator),
                        validator_hotkey=reg_msg.hotkey,
                        connection_id=connection_id,
                        is_connected=True,
                    )
                    session.add(validator_conn)
                else:
                    validator_conn.connection_id = connection_id
                    validator_conn.last_connected_at = datetime.now(timezone.utc)
                    validator_conn.last_heartbeat = datetime.now(timezone.utc)
                    validator_conn.is_connected = True

                await session.commit()

            self.websocket_connections[connection_id] = websocket

            logger.info(f"Validator registered: {reg_msg.hotkey} ({connection_id})")

            # Send acknowledgment
            ack = {
                "message_type": "registration_ack",
                "status": "registered",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            await websocket.send(json.dumps(ack))

            return reg_msg.hotkey

        except Exception as e:
            logger.error(f"Failed to register validator {connection_id}: {e}")
            return None

    async def _process_validator_message(
        self,
        websocket: WebSocketServerProtocol,
        connection_id: str,
        validator_hotkey: str,
        message: dict,
    ):
        """Process messages from registered validators."""
        try:
            message_type = message.get("message_type")

            if message_type == "heartbeat":
                await self._handle_validator_heartbeat(validator_hotkey)
            elif message_type == "eval_result":
                await self._handle_eval_result(validator_hotkey, message)
            else:
                logger.debug(f"Message from {validator_hotkey}: {message_type}")

        except Exception as e:
            logger.error(f"Error processing message from {validator_hotkey}: {e}")

    async def _handle_validator_heartbeat(self, validator_hotkey: str):
        """Handle validator heartbeat."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(ValidatorConnection).where(
                        ValidatorConnection.validator_hotkey == validator_hotkey
                    )
                )
                validator_conn = result.scalar_one_or_none()

                if validator_conn:
                    validator_conn.last_heartbeat = datetime.now(timezone.utc)
                    await session.commit()

        except Exception as e:
            logger.error(f"Failed to update heartbeat for {validator_hotkey}: {e}")

    async def _handle_eval_result(self, validator_hotkey: str, message: dict):
        """Handle evaluation result from validator."""
        try:
            result_msg = EvalResultMessage(**message)

            async with self.async_session() as session:
                # Find the backend job
                job_result = await session.execute(
                    select(BackendEvaluationJob).where(
                        BackendEvaluationJob.job_id == result_msg.job_id
                    )
                )
                backend_job = job_result.scalar_one_or_none()

                if not backend_job:
                    logger.warning(f"Unknown job ID in result: {result_msg.job_id}")
                    return

                # Create evaluation result
                eval_result = BackendEvaluationResult(
                    id=next(self.id_generator),
                    job_id=result_msg.job_id,
                    backend_job_id=backend_job.id,
                    validator_hotkey=validator_hotkey,
                    miner_hotkey=result_msg.miner_hotkey,
                    competition_id=result_msg.competition_id,
                    benchmark=result_msg.benchmark,
                    score=result_msg.score,
                    success_rate=result_msg.success_rate,
                    avg_reward=result_msg.avg_reward,
                    total_episodes=result_msg.total_episodes,
                    logs=result_msg.logs,
                    error=result_msg.error,
                    extra_data=result_msg.extra_data,
                )

                session.add(eval_result)

                # Update job completion count
                backend_job.validators_completed += 1

                # Update validator statistics
                validator_result = await session.execute(
                    select(ValidatorConnection).where(
                        ValidatorConnection.validator_hotkey == validator_hotkey
                    )
                )
                validator_conn = validator_result.scalar_one_or_none()

                if validator_conn:
                    validator_conn.total_results_received += 1
                    if result_msg.error:
                        validator_conn.total_errors += 1

                await session.commit()

                logger.info(
                    f"Stored result for job {result_msg.job_id} from {validator_hotkey} "
                    f"(benchmark: {result_msg.benchmark}, score: {result_msg.score})"
                )

        except Exception as e:
            logger.error(f"Failed to handle eval result: {e}")

    async def _handle_validator_disconnect(
        self, validator_hotkey: str, connection_id: str
    ):
        """Handle validator disconnection."""
        try:
            async with self.async_session() as session:
                result = await session.execute(
                    select(ValidatorConnection).where(
                        ValidatorConnection.validator_hotkey == validator_hotkey
                    )
                )
                validator_conn = result.scalar_one_or_none()

                if validator_conn:
                    validator_conn.is_connected = False
                    await session.commit()

            logger.info(f"Validator disconnected: {validator_hotkey} ({connection_id})")

        except Exception as e:
            logger.error(f"Failed to handle disconnect for {validator_hotkey}: {e}")

    async def _monitor_chain(self):
        """Monitor blockchain for miner commitments."""
        while self._running:
            try:
                await self.sync_metagraph()
                async with self.async_session() as session:
                    # Get backend state
                    state_result = await session.execute(
                        select(BackendState).where(BackendState.id == 1)
                    )
                    state = state_result.scalar_one()

                    # Get latest block (placeholder - actual implementation would query chain)
                    latest_block = await self._get_latest_block()
                    start_block = max(
                        state.last_seen_block + 1,
                        latest_block - self.max_commitment_lookback + 1,
                    )

                    logger.info(
                        f"Checking blocks {start_block} to {latest_block} for commitments"
                    )

                    # Get active competitions for filtering
                    comp_result = await session.execute(
                        select(Competition).where(Competition.active)
                    )
                    active_competitions = {c.id: c for c in comp_result.scalars()}

                    # Query commitments for each block
                    for block_num in range(start_block, latest_block + 1):
                        commitments = await self._query_block_commitments(block_num)

                        for commitment in commitments:
                            await self._process_commitment(
                                commitment, block_num, active_competitions
                            )

                    # Update last seen block
                    state.last_seen_block = latest_block
                    state.last_chain_scan = datetime.now(timezone.utc)
                    await session.commit()

                # Sleep before next check
                await asyncio.sleep(self.chain_sync_interval)

            except Exception as e:
                logger.error(f"Error monitoring chain: {e}")
                await asyncio.sleep(self.chain_sync_interval // 2)

    async def _process_commitment(
        self,
        commitment: ChainCommitmentResponse,
        block_num: int,
        active_competitions: dict,
    ):
        """Process a miner commitment from the chain."""
        try:
            # Extract competition ID from commitment data
            competition_id = getattr(commitment.data, "competition_id", None)

            if not competition_id or competition_id not in active_competitions:
                logger.warning(
                    f"Unknown competition ID in commitment: {competition_id}"
                )
                return

            competition = active_competitions[competition_id]

            async with self.async_session() as session:
                # Check if submission already exists
                existing_result = await session.execute(
                    select(MinerSubmission).where(
                        and_(
                            MinerSubmission.miner_hotkey == commitment.hotkey,
                            MinerSubmission.competition_id == competition_id,
                            MinerSubmission.version == commitment.data.version,
                        )
                    )
                )

                if existing_result.scalar_one_or_none():
                    logger.debug(f"Submission already exists for {commitment.hotkey}")
                    return

                # Create new submission
                submission = MinerSubmission(
                    id=next(self.id_generator),
                    miner_hotkey=commitment.hotkey,
                    competition_id=competition_id,
                    hf_repo_id=commitment.data.repo_id,
                    version=commitment.data.version,
                    commitment_block=block_num,
                )

                session.add(submission)
                await session.flush()

                # Create evaluation job
                job_id = str(uuid.uuid4())
                eval_job = BackendEvaluationJob(
                    id=next(self.id_generator),
                    job_id=job_id,
                    submission_id=submission.id,
                    competition_id=competition_id,
                    miner_hotkey=submission.miner_hotkey,
                    hf_repo_id=submission.hf_repo_id,
                    benchmarks=competition.benchmarks,
                )

                session.add(eval_job)
                await session.commit()

                # Broadcast to validators
                await self._broadcast_eval_job(eval_job, competition)

                logger.info(
                    f"Processed commitment from {commitment.hotkey} for competition {competition_id}"
                )

        except Exception as e:
            logger.error(f"Failed to process commitment: {e}")

    async def _broadcast_eval_job(
        self, job: BackendEvaluationJob, competition: Competition
    ):
        """Broadcast evaluation job to all connected validators."""
        if not self.websocket_connections:
            logger.warning("No validators connected - cannot broadcast job")
            return

        # Create job message
        job_msg = EvalJobMessage(
            job_id=job.job_id,
            competition_id=job.competition_id,
            miner_hotkey=job.miner_hotkey,
            hf_repo_id=job.hf_repo_id,
            benchmarks=job.benchmarks,
        )

        broadcast_count = 0
        failed_connections = []

        for connection_id, websocket in self.websocket_connections.items():
            try:
                await websocket.send(job_msg.model_dump_json())
                broadcast_count += 1
                logger.debug(f"Job {job.job_id} sent to connection {connection_id}")
            except Exception as e:
                logger.error(f"Failed to send job to {connection_id}: {e}")
                failed_connections.append(connection_id)

        # Update job broadcast info
        async with self.async_session() as session:
            job_result = await session.execute(
                select(BackendEvaluationJob).where(BackendEvaluationJob.id == job.id)
            )
            db_job = job_result.scalar_one()
            db_job.broadcast_time = datetime.now(timezone.utc)
            db_job.validators_sent = broadcast_count

            # Update validator stats
            for connection_id in self.websocket_connections:
                if connection_id not in failed_connections:
                    # Find validator by connection_id
                    val_result = await session.execute(
                        select(ValidatorConnection).where(
                            ValidatorConnection.connection_id == connection_id
                        )
                    )
                    validator = val_result.scalar_one_or_none()
                    if validator:
                        validator.total_jobs_sent += 1

            await session.commit()

        # Clean up failed connections
        for connection_id in failed_connections:
            del self.websocket_connections[connection_id]

        logger.info(f"Job {job.job_id} broadcasted to {broadcast_count} validators")

    async def _monitor_validator_heartbeats(self):
        """Monitor validator heartbeats and mark stale connections."""
        while self._running:
            try:
                current_time = datetime.now(timezone.utc)
                timeout_threshold = current_time - timedelta(minutes=2)

                async with self.async_session() as session:
                    # Find stale validators
                    result = await session.execute(
                        select(ValidatorConnection).where(
                            and_(
                                ValidatorConnection.is_connected,
                                ValidatorConnection.last_heartbeat < timeout_threshold,
                            )
                        )
                    )

                    stale_validators = result.scalars().all()

                    for validator in stale_validators:
                        logger.warning(
                            f"Marking validator as disconnected: {validator.validator_hotkey}"
                        )
                        validator.is_connected = False

                        # Close WebSocket if still in connections
                        if validator.connection_id in self.websocket_connections:
                            try:
                                await self.websocket_connections[
                                    validator.connection_id
                                ].close()
                            except Exception:
                                pass
                            del self.websocket_connections[validator.connection_id]

                    await session.commit()

                await asyncio.sleep(30)

            except Exception as e:
                logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(30)

    async def _get_latest_block(self) -> int:
        """Get the latest block number from the chain."""
        try:
            if not self.substrate:
                raise RuntimeError("Substrate connection not initialized")
            return self.substrate.get_block_number()
        except Exception as e:
            logger.error(f"Failed to get latest block number: {e}")
            # Return a fallback value to prevent complete failure
            return 1000

    async def sync_metagraph(self):
        self.metagraph.sync_nodes()

    async def _query_block_commitments(
        self, block_num: int
    ) -> List[ChainCommitmentResponse]:
        """Query commitments for a specific block from all miners."""
        commitments = []

        try:
            # Query commitments from each miner for this block
            metagraph_nodes = self.metagraph.nodes
            for node in metagraph_nodes.values():
                miner_hotkey = node.hotkey
                try:
                    miner_commitments = query_commitments_from_substrate(
                        self.config, miner_hotkey, block=block_num
                    )
                    if miner_commitments:
                        commitments.extend(miner_commitments)
                        logger.debug(
                            f"Found {len(miner_commitments)} commitments from {miner_hotkey} at block {block_num}"
                        )
                except Exception as e:
                    logger.debug(
                        f"Failed to query commitments from {miner_hotkey} at block {block_num}: {e}"
                    )
                    continue

        except Exception as e:
            logger.error(f"Failed to query block {block_num} commitments: {e}")

        return commitments

    async def get_competition_stats(self) -> dict:
        """Get statistics about competitions and submissions."""
        try:
            async with self.async_session() as session:
                # Get competitions
                comp_result = await session.execute(
                    select(Competition).where(Competition.active)
                )
                competitions = comp_result.scalars().all()

                # Get connected validators count
                val_result = await session.execute(
                    select(ValidatorConnection).where(ValidatorConnection.is_connected)
                )
                connected_validators = len(val_result.scalars().all())

                # Calculate point distribution
                total_points = sum(c.points for c in competitions)

                stats = {
                    "competitions": len(competitions),
                    "total_points": total_points,
                    "connected_validators": connected_validators,
                }

                # Calculate percentages
                for comp in competitions:
                    percentage = (
                        (comp.points / total_points * 100) if total_points > 0 else 0
                    )
                    stats[f"competition_{comp.id}_percentage"] = percentage

                return stats

        except Exception as e:
            logger.error(f"Failed to get competition stats: {e}")
            return {}


async def main():
    """Main entry point for the Kinitro Backend."""
    config = BackendConfig()

    backend = KinitroBackend(config)

    try:
        await backend.start()

        # Keep running until interrupted
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Shutting down Kinitro Backend")
        await backend.stop()


if __name__ == "__main__":
    asyncio.run(main())
