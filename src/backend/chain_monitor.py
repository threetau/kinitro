"""
Chain monitor for Kinitro backend.

Monitors Bittensor chain for miner commitments and processes them.
Extracted from BackendService for better separation of concerns.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Coroutine, Dict, List, Optional

from fiber.chain.fetch_nodes import _get_nodes_for_uid
from fiber.chain.models import Node
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from core.chain import query_commitments_from_substrate
from core.log import get_logger
from core.schemas import ChainCommitmentResponse

from .config import BackendConfig
from .models import BackendState, Competition, SS58Address

logger = get_logger(__name__)


class ChainConfig:
    """Configuration for the chain monitor."""

    def __init__(
        self,
        max_commitment_lookback: int = 360,
        chain_sync_interval: float = 30.0,
        chain_scan_yield_interval: int = 2,
    ):
        self.max_commitment_lookback = max_commitment_lookback
        self.chain_sync_interval = chain_sync_interval
        self.chain_scan_yield_interval = chain_scan_yield_interval


CommitmentCallback = Callable[
    [ChainCommitmentResponse, int, Dict[str, Competition]],
    Coroutine[Any, Any, None],
]


class ChainMonitor:
    """
    Monitors Bittensor chain for commitments.

    This class is responsible for:
    - Scanning blocks for miner commitments
    - Syncing metagraph nodes
    - Calling registered callbacks when commitments are found
    """

    def __init__(
        self,
        substrate: Any,
        backend_config: BackendConfig,
        session_factory: async_sessionmaker[AsyncSession],
        config: ChainConfig,
        thread_pool: ThreadPoolExecutor,
        on_commitment: Optional[CommitmentCallback] = None,
    ):
        self.substrate = substrate
        self.backend_config = backend_config
        self.session_factory = session_factory
        self.config = config
        self.thread_pool = thread_pool
        self.on_commitment = on_commitment

        # Chain state
        self.nodes: Optional[Dict[SS58Address, Node]] = None

        # Task handle
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """Start the chain monitor background task."""
        if self._running:
            logger.warning("ChainMonitor already running")
            return

        self._running = True
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info("ChainMonitor started")

    async def stop(self) -> None:
        """Stop the chain monitor background task."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                logger.info("ChainMonitor task cancelled")
        self._task = None
        logger.info("ChainMonitor stopped")

    async def _monitor_loop(self) -> None:
        """Background task to monitor blockchain for commitments."""
        while self._running:
            try:
                await self._scan_once()
                await asyncio.sleep(self.config.chain_sync_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring chain: {e}")
                await asyncio.sleep(self.config.chain_sync_interval)

    async def _scan_once(self) -> None:
        """Perform a single scan of the blockchain."""
        if not self.substrate or not self.session_factory:
            return

        # Sync metagraph first
        await self._sync_metagraph()

        if not self.nodes:
            return

        async with self.session_factory() as session:
            # Get backend state
            state_result = await session.execute(
                select(BackendState).where(BackendState.id == 1)
            )
            state = state_result.scalar_one_or_none()
            if not state:
                logger.warning("Backend state not found")
                return

            # Get latest block
            latest_block = await self._get_latest_block()
            if latest_block < 0:
                return

            start_block = max(
                state.last_seen_block + 1,
                latest_block - self.config.max_commitment_lookback + 1,
            )

            logger.info(f"Checking blocks {start_block} to {latest_block}")

            # Get active competitions
            comp_result = await session.execute(
                select(Competition).where(Competition.active)
            )
            active_competitions = {c.id: c for c in comp_result.scalars()}
            logger.debug(
                f"Preview of active competitions: {list(active_competitions.keys())[:5]}"
            )

            # Scan blocks and process commitments
            await self.scan_blocks(start_block, latest_block, active_competitions)

            # Update state
            state.last_seen_block = latest_block
            state.last_chain_scan = datetime.now(timezone.utc)
            await session.commit()

    async def scan_blocks(
        self,
        start: int,
        end: int,
        active_competitions: Dict[str, Competition],
    ) -> List[ChainCommitmentResponse]:
        """
        Scan a range of blocks for commitments.

        Args:
            start: Starting block number
            end: Ending block number
            active_competitions: Currently active competitions

        Returns:
            List of commitments found
        """
        all_commitments: List[ChainCommitmentResponse] = []

        for i, block_num in enumerate(range(start, end + 1)):
            commitments = await self._query_block_commitments(block_num)
            all_commitments.extend(commitments)

            for commitment in commitments:
                if self.on_commitment:
                    await self.on_commitment(commitment, block_num, active_competitions)

            # Yield control periodically to prevent blocking WebSocket connections
            if i % self.config.chain_scan_yield_interval == 0:
                await asyncio.sleep(0)

        return all_commitments

    async def _get_latest_block(self) -> int:
        """Get latest block from chain.

        Returns:
            int: The latest block number, or -1 if an error occurred.
        """
        try:
            if not self.substrate:
                return -1
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self.thread_pool, self.substrate.get_block_number
            )
        except Exception as e:
            logger.error(f"Failed to get latest block: {e}")
            return -1

    def _sync_nodes_sync(self) -> None:
        """Synchronous version of node syncing for thread pool."""
        node_list = _get_nodes_for_uid(
            self.substrate, self.backend_config.settings["subtensor"]["netuid"]
        )
        self.nodes = {node.hotkey: node for node in node_list}

    async def _sync_metagraph(self) -> None:
        """Sync metagraph nodes with memory leak prevention."""
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(self.thread_pool, self._sync_nodes_sync)
            logger.debug("Nodes synced")
        except Exception as e:
            logger.error(f"Failed to sync metagraph: {e}")

    def _query_commitments_sync(
        self, block_num: int, nodes: list
    ) -> List[ChainCommitmentResponse]:
        """Synchronous version of commitment querying for thread pool."""
        commitments = []

        for node in nodes:
            try:
                miner_commitments = query_commitments_from_substrate(
                    self.backend_config, self.substrate, node.hotkey, block=block_num
                )
                if miner_commitments:
                    commitments.extend(miner_commitments)
            except Exception as e:
                logger.debug(f"Failed to query {node.hotkey}: {e}")
                continue

        return commitments

    async def _query_block_commitments(
        self, block_num: int
    ) -> List[ChainCommitmentResponse]:
        """Query commitments for a block."""
        try:
            if not self.nodes:
                return []

            node_list = list(self.nodes.values())

            loop = asyncio.get_event_loop()
            commitments = await loop.run_in_executor(
                self.thread_pool, self._query_commitments_sync, block_num, node_list
            )

            return commitments

        except Exception as e:
            logger.error(f"Failed to query block {block_num}: {e}")
            return []

    def get_nodes(self) -> Optional[Dict[SS58Address, Node]]:
        """Get the current node mapping."""
        return self.nodes
