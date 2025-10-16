"""Chain interaction helpers"""

from fiber import Keypair, SubstrateInterface
from fiber.chain.chain_utils import (
    load_hotkey_keypair,
    query_substrate,
)
from fiber.chain.commitments import (
    CommitmentDataFieldType,
    query_commitment,
    set_commitment,
)
from fiber.chain.interface import get_substrate
from fiber.chain.metagraph import Metagraph
from fiber.chain.weights import (
    _normalize_and_quantize_weights,
    _set_weights_with_commit_reveal,
    _set_weights_without_commit_reveal,
    can_set_weights,
)
from pydantic import ValidationError

from .config import Config
from .errors import CommitmentError
from .log import get_logger
from .schemas import ChainCommitment, ChainCommitmentResponse

logger = get_logger(__name__)


def commit_to_substrate(config: Config, commit_data: ChainCommitment) -> None:
    """
    Commit information to substrate chain using fiber.

    Args:
        config: Neuron configuration
        commit_data: Data to commit to the chain

    Raises:
        CommitmentError: If substrate commitment fails
    """

    logger.info("Connecting to substrate chain...")

    try:
        # Get substrate connection
        substrate = get_substrate(
            subtensor_network=config.settings["subtensor"]["network"],
            subtensor_address=config.settings["subtensor"]["address"],
        )

        logger.info("Successfully connected to substrate")

        # Create metagraph to get neuron information (unused but available for future use)
        Metagraph(
            netuid=config.settings["subtensor"]["netuid"],
            substrate=substrate,  # type: ignore
        )

        # Prepare commitment data
        commitment_data = commit_data.model_dump_json()

        keypair = load_hotkey_keypair(
            wallet_name=config.settings["wallet_name"],
            hotkey_name=config.settings["hotkey_name"],
        )

        # Set commitment on chain
        logger.info("Setting commitment on chain...")
        success = set_commitment(
            substrate=substrate,  # type: ignore
            keypair=keypair,
            netuid=config.settings["subtensor"]["netuid"],
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


# check commitment in substrate
def query_commitments_from_substrate(
    config: Config,
    substrate: SubstrateInterface,
    miner_hotkey: str,
    block: int | None = None,
) -> list[ChainCommitmentResponse]:
    """
    Query commitments from substrate chain.

    Args:
        config: Neuron configuration
        block: Block number to query commitments from. If None, queries the latest block.

    Returns:
        ChainCommitment: The queried commitment data

    Raises:
        CommitmentError: If querying fails
    """

    try:
        commitment_query = query_commitment(
            substrate=substrate,  # type: ignore
            netuid=config.settings["subtensor"]["netuid"],
            hotkey=miner_hotkey,
            block=block,
        )

        if commitment_query is None or commitment_query.fields is None:
            return []

        commitments: list[ChainCommitmentResponse] = []

        for query in commitment_query.fields:
            if query is None:
                continue

            # Handle both tuple and dict formats from fiber library
            if isinstance(query, dict):
                # If query is a dict, extract data_type and data from it
                data_type = query.get("data_type") or query.get("type")
                data = query.get("data") or query.get("value")

                if data_type is None or data is None:
                    logger.warning(f"Invalid commitment query format: {query}")
                    continue
            else:
                # If query is a tuple, unpack it as before
                try:
                    data_type, data = query
                except (ValueError, TypeError) as e:
                    logger.warning(
                        f"Failed to unpack commitment query: {query}, error: {e}"
                    )
                    continue

            match data_type:
                case CommitmentDataFieldType.RAW:
                    commitment = data.decode("utf-8")
                    try:
                        commitment = ChainCommitment.model_validate_json(commitment)
                        commitments.append(
                            ChainCommitmentResponse(
                                hotkey=miner_hotkey, data=commitment
                            )
                        )
                    except ValidationError as e:
                        logger.warning(
                            f"Failed to validate commitment: {commitment}, error: {e}"
                        )
                        continue
                case _:
                    logger.warning(
                        f"Unknown/unsupported commitment data field type: {data_type}"
                    )

        return commitments

    except Exception as e:
        raise CommitmentError(f"Failed to query commitment: {e}") from e


def set_node_weights(
    substrate: SubstrateInterface,
    keypair: Keypair,
    node_ids: list[int],
    node_weights: list[float],
    netuid: int,
    validator_node_id: int,
    version_key: int = 0,
    wait_for_inclusion: bool = False,
    wait_for_finalization: bool = False,
    max_attempts: int | None = None,  # NOTE: DEPRECATED
) -> bool:
    """
    Set node weights on the substrate chain.

    NOTE: This function is copied from fiber and modified to fix memory leak issues
    in the original implementation.
    """
    if max_attempts is not None:
        logger.warning(
            "Parameter 'max_attempts' is deprecated and will be removed in version 2.2.0"
        )
    node_ids_formatted, node_weights_formatted = _normalize_and_quantize_weights(
        node_ids, node_weights
    )

    if not can_set_weights(substrate, netuid, validator_node_id):
        return False

    substrate, commit_reveal_enabled = query_substrate(
        substrate,
        "SubtensorModule",
        "CommitRevealWeightsEnabled",
        [netuid],
        return_value=True,
    )

    logger.info(
        f"Commit reveal enabled hyperparameter is set to {commit_reveal_enabled}"
    )

    if commit_reveal_enabled is False:
        return _set_weights_without_commit_reveal(
            substrate,
            keypair,
            node_ids_formatted,
            node_weights_formatted,
            netuid,
            version_key,
            wait_for_inclusion,
            wait_for_finalization,
        )

    elif commit_reveal_enabled is True:
        return _set_weights_with_commit_reveal(
            substrate,
            keypair,
            node_ids_formatted,
            node_weights_formatted,
            netuid,
            version_key,
            wait_for_inclusion,
            wait_for_finalization,
        )

    else:
        raise ValueError(
            f"Commit reveal enabled hyperparameter is set to {commit_reveal_enabled}, which is not a valid value"
        )
