import sys
from abc import ABC
from typing import Dict, Optional, cast

# import httpx
# from fastapi import FastAPI
from fiber.chain import chain_utils, interface
from fiber.chain.fetch_nodes import _get_nodes_for_uid
from fiber.chain.models import Node

# from storb import __spec_version__, get_spec_version
from .config import Config
from .log import get_logger

logger = get_logger(__name__)

LOG_MAX_SIZE = 5 * 1024 * 1024  # 5 MiB


class Neuron(ABC):
    """Base class for Bittensor neurons (miners and validators)"""

    # Abstract attribute that subclasses must define
    api_port: int

    def __init__(self, config: Config):
        self.config = config
        self.settings = self.config.settings
        # self.spec_version = __spec_version__

        # assert (
        #     get_spec_version(self.settings.version) == self.spec_version
        # ), "The spec versions must match"

        assert self.settings.wallet_name, "Wallet must be defined"
        assert self.settings.hotkey_name, "Hotkey must be defined"

        logger.info(
            f"Attempt to load hotkey keypair with wallet name: {self.settings.wallet_name}"
        )
        self.keypair = chain_utils.load_hotkey_keypair(
            cast(str, self.settings.wallet_name), cast(str, self.settings.hotkey_name)
        )
        assert self.keypair, "Keypair must be defined"

        self.subtensor_network = cast(str, self.settings["subtensor"]["network"])
        self.subtensor_address = cast(str, self.settings["subtensor"]["address"])
        assert self.subtensor_network, "Subtensor network must be defined"
        assert self.subtensor_address, "Subtensor address must be defined"

        self.substrate = interface.get_substrate(
            subtensor_network=self.subtensor_network,
            subtensor_address=self.subtensor_address,
        )
        assert self.substrate, "Substrate must be defined"

        self.netuid = cast(int, self.settings.netuid)

        assert self.netuid, "Netuid must be defined"

        # Initialize nodes dictionary
        self.nodes: Optional[Dict[str, Node]] = None
        self.sync_nodes()

        self.check_registration()
        node = self.nodes.get(self.keypair.ss58_address) if self.nodes else None
        self.uid = getattr(node, "node_id", None) if node else None
        assert self.uid, "UID must be defined"

    # @abstractmethod
    # async def start(self): ...

    # @abstractmethod
    # async def stop(self): ...

    def check_registration(self):
        node = self.nodes.get(self.keypair.ss58_address) if self.nodes else None
        if not node or not getattr(node, "node_id", None):
            logger.error(
                f"Wallet is not registered on netuid {self.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit(1)

    def sync_nodes(self):
        """Synchronize nodes from chain."""
        try:
            node_list = _get_nodes_for_uid(self.substrate, self.netuid)
            self.nodes = {node.hotkey: node for node in node_list}
            logger.info(f"Synced {len(self.nodes)} nodes")
        except Exception as e:
            logger.error(f"Failed to sync nodes: {str(e)}")
            if not self.nodes:
                self.nodes = {}
