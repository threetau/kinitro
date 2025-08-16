import json

from fiber.chain.chain_utils import load_hotkey_keypair
from fiber.chain.commitments import CommitmentDataFieldType, set_commitment
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph

from core.errors import CommitmentError
from core.log import get_logger
from miner.config import MinerConfig

logger = get_logger(__name__)


def commit_to_substrate(config: MinerConfig, commit_data: dict) -> None:
    """
    Commit information to substrate chain using fiber.

    Args:
        config: Miner configuration
        commit_data: Data to commit to the chain

    Raises:
        CommitmentError: If substrate commitment fails
    """
    logger.info("Connecting to substrate chain...")

    try:
        # Get substrate connection
        substrate = get_substrate(
            subtensor_network=config.settings.subtensor_network,
            subtensor_address=config.settings.subtensor_address,
        )

        logger.info("Successfully connected to substrate")

        # Create metagraph to get neuron information (unused but available for future use)
        Metagraph(
            netuid=config.settings.netuid,
            substrate=substrate,
        )

        # Prepare commitment data
        commitment_data = json.dumps(commit_data)

        keypair = load_hotkey_keypair(
            wallet_name=config.settings.wallet_name,
            hotkey_name=config.settings.hotkey_name,
        )

        # Set commitment on chain
        logger.info("Setting commitment on chain...")
        success = set_commitment(
            substrate=substrate,
            keypair=keypair,
            netuid=config.settings.netuid,
            fields=[
                (CommitmentDataFieldType.RAW, commitment_data.encode("utf-8")),
            ],
            wait_for_inclusion=True,
            wait_for_finalization=True,
        )

        if not success:
            raise CommitmentError("Failed to commit data to substrate chain")

        logger.info("Successfully committed data to substrate chain")

    except Exception as e:
        if isinstance(e, CommitmentError):
            raise
        raise CommitmentError(
            f"Error connecting to substrate or committing data: {e}"
        ) from e
