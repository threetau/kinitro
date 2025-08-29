import logging as pylog
import sys
import time
from abc import ABC
from typing import cast

# import httpx
# from fastapi import FastAPI
from fiber.chain import chain_utils, interface
from fiber.chain.metagraph import Metagraph

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

        self.metagraph = Metagraph(
            netuid=cast(str, self.netuid),
            substrate=self.substrate,  # type: ignore[arg-type]
        )
        assert self.metagraph, "Metagraph must be initialised"
        self.metagraph.sync_nodes()

        self.check_registration()
        node = self.metagraph.nodes.get(self.keypair.ss58_address)
        self.uid = getattr(node, "node_id", None) if node else None
        assert self.uid, "UID must be defined"

    # @abstractmethod
    # async def start(self): ...

    # @abstractmethod
    # async def stop(self): ...

    def check_registration(self):
        node = self.metagraph.nodes.get(self.keypair.ss58_address)
        if not node or not getattr(node, "node_id", None):
            logger.error(
                f"Wallet is not registered on netuid {self.netuid}."
                f" Please register the hotkey using `btcli subnets register` before trying again"
            )
            sys.exit(1)

    def sync_metagraph(self):
        """Synchronize local metagraph state with chain.

        Creates new metagraph instance if needed and syncs node data.

        Raises
        ------
        Exception
            If metagraph sync fails
        """

        try:
            self.substrate = interface.get_substrate(
                subtensor_address=self.substrate.url
            )
            self.metagraph.sync_nodes()
            self.metagraph.save_nodes()

            logger.info("Metagraph synced successfully")
        except Exception as e:
            logger.error(f"Failed to sync metagraph: {str(e)}")

    def run(self):
        """Background task to sync metagraph"""

        while True:
            try:
                self.sync_metagraph()
                time.sleep(cast(int, self.settings["neuron"]["sync_frequency"]))
            except Exception as e:
                logger.error(f"Error in sync metagraph: {e}")
                time.sleep(cast(int, self.settings["neuron"]["sync_frequency"]) // 2)
