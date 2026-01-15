"""
Polling-based validator for Kinitro.

This validator periodically fetches weights from the backend and sets them
on the Bittensor chain. This is a simple polling approach that avoids the
complexity of maintaining WebSocket connections.
"""

import asyncio
import os
from typing import Dict, Optional

import httpx
from fiber.chain.fetch_nodes import _get_nodes_for_uid
from fiber.chain.models import Node

from backend.models import SS58Address
from core.chain import set_node_weights
from core.log import get_logger
from core.neuron import Neuron

from .config import ValidatorConfig

logger = get_logger(__name__)


class WeightSettingValidator(Neuron):
    """
    Polling-based validator that fetches weights and sets them on chain.

    This validator:
    1. Periodically polls the backend /weights endpoint
    2. Sets the received weights on the Bittensor chain

    Much simpler than maintaining a WebSocket connection.
    """

    def __init__(self, config: ValidatorConfig):
        super().__init__(config)
        self.hotkey = self.keypair.ss58_address

        # Backend settings
        self.backend_url = config.settings.get(
            "backend_url", "http://localhost:8080"
        ).rstrip("/")
        self.poll_interval = config.settings.get("weight_poll_interval", 300)  # 5 min default

        # Chain state
        self.nodes: Optional[Dict[SS58Address, Node]] = None
        self.validator_node_id: Optional[int] = None

        # Track last weights to avoid redundant chain calls
        self._last_weights_hash: Optional[int] = None

        self._running = False

        logger.info(
            "WeightSettingValidator initialized for hotkey: %s, polling every %ds",
            self.hotkey,
            self.poll_interval,
        )

    async def start(self):
        """Start the validator service."""
        logger.info("Starting WeightSettingValidator")
        self._running = True

        # Initialize chain connection
        await self._init_chain()

        # Main polling loop
        while self._running:
            try:
                await self._poll_and_set_weights()
            except Exception as e:
                logger.error(f"Error in weight polling cycle: {e}")

            if self._running:
                await asyncio.sleep(self.poll_interval)

    async def stop(self):
        """Stop the validator service."""
        logger.info("Stopping WeightSettingValidator")
        self._running = False

    async def _poll_and_set_weights(self):
        """Fetch weights from backend and set on chain."""
        try:
            # Fetch weights from backend
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.backend_url}/weights")

                if response.status_code == 404:
                    logger.debug("Weights not available yet from backend")
                    return

                response.raise_for_status()
                data = response.json()

            weights = data.get("weights", {})
            if not weights:
                logger.debug("Empty weights received from backend")
                return

            # Convert string keys to int (JSON serializes int keys as strings)
            weights = {int(k): float(v) for k, v in weights.items()}

            # Check if weights changed
            weights_hash = hash(frozenset(weights.items()))
            if weights_hash == self._last_weights_hash:
                logger.debug("Weights unchanged, skipping chain update")
                return

            logger.info(
                "Received new weights from backend: %d UIDs, total=%.4f",
                len(weights),
                sum(weights.values()),
            )

            # Set weights on chain
            await self._set_weights_on_chain(weights)
            self._last_weights_hash = weights_hash

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching weights: {e.response.status_code}")
        except httpx.RequestError as e:
            logger.error(f"Request error fetching weights: {e}")
        except Exception as e:
            logger.error(f"Error polling weights: {e}")

    async def _set_weights_on_chain(self, weights: Dict[int, float]):
        """Set weights on the Bittensor chain."""
        if not self.substrate:
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

        # Extract node_ids and weights as parallel lists
        node_ids = list(weights.keys())
        node_weights = list(weights.values())

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
            logger.info(f"Successfully set weights on chain for {len(node_ids)} miners")
        else:
            logger.error("Failed to set weights on chain")

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
            logger.warning("Continuing without blockchain connection")

    async def _sync_nodes(self) -> None:
        """Sync nodes from the chain."""
        try:
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


# Backwards compatibility alias
WebSocketValidator = WeightSettingValidator
