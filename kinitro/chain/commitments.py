"""Miner commitment handling for on-chain model registration.

Miners commit their HuggingFace repo and revision on-chain. The executor
downloads models from HuggingFace and creates deployments on-demand.
"""

from __future__ import annotations

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


# Chain commitment size limit (bytes)
MAX_COMMITMENT_SIZE = 128


@dataclass
class MinerCommitment:
    """Parsed miner commitment from chain.

    Contains HuggingFace repo and revision for executor-managed deployments.
    """

    uid: int
    hotkey: str
    huggingface_repo: str
    revision_sha: str
    committed_block: int

    @property
    def is_valid(self) -> bool:
        """Check if commitment has all required fields."""
        return bool(self.huggingface_repo and self.revision_sha)


def parse_commitment(raw: str) -> dict:
    """
    Parse raw commitment string from chain.

    New format: "user/repo:rev8char"
    Legacy format (backward compatible): "user/repo:rev8char:deployment_id"
                                         "user/repo:rev8char:e:<base85_blob>"

    Note: revision is truncated to 8 characters (short SHA).

    Args:
        raw: Raw commitment string

    Returns:
        Dict with parsed fields:
            - huggingface_repo: HuggingFace repo (e.g., "user/model")
            - revision_sha: Commit SHA (truncated to 8 chars)
    """
    parts = raw.split(":", 3)

    if len(parts) >= 2:
        hf_repo = parts[0]
        revision = parts[1]

        # Legacy format with deployment_id (3+ parts) - ignore deployment info
        # New format is just repo:revision (2 parts)
        return {
            "huggingface_repo": hf_repo,
            "revision_sha": revision,
        }

    # Invalid format
    logger.warning("invalid_commitment_format", raw=raw)
    return {
        "huggingface_repo": "",
        "revision_sha": "",
    }


def _query_commitment_by_hotkey(
    subtensor, netuid: int, hotkey: str
) -> tuple[str | None, int | None]:
    """
    Query commitment directly from chain storage by hotkey.

    This bypasses the broken get_commitment API in bittensor SDK.

    Args:
        subtensor: Bittensor subtensor connection
        netuid: Subnet UID
        hotkey: Hotkey SS58 address

    Returns:
        Tuple of (commitment_string, block_number) or (None, None)
    """
    try:
        result = subtensor.substrate.query("Commitments", "CommitmentOf", [netuid, hotkey])

        # Handle different result types
        if result is None:
            return None, None

        # Result might be a dict directly (newer substrate interface)
        if isinstance(result, dict):
            data = result
        elif hasattr(result, "value"):
            data = result.value
        else:
            return None, None

        if not data:
            return None, None

        # Handle structured commitment format: {'deposit': ..., 'block': ..., 'info': {'fields': ...}}
        if isinstance(data, dict) and "info" in data:
            # Extract block number from commitment data
            block = data.get("block")
            if block is not None:
                try:
                    block = int(block)
                except (TypeError, ValueError):
                    block = None

            info = data.get("info", {})
            fields = info.get("fields", ())

            if fields and len(fields) > 0:
                # First field contains the raw data
                first_field = fields[0]

                # Handle tuple format: ({'Raw94': ((bytes...),)},)
                if isinstance(first_field, tuple) and len(first_field) > 0:
                    first_field = first_field[0]

                # Extract bytes from various formats
                if isinstance(first_field, dict):
                    # Format: {'RawXX': ((bytes...),)} or {'Data': bytes}
                    for key, value in first_field.items():
                        if key.startswith("Raw") or key == "Data":
                            # Extract the bytes tuple
                            if isinstance(value, tuple) and len(value) > 0:
                                byte_data = value[0]
                                if isinstance(byte_data, (list, tuple)):
                                    return bytes(byte_data).decode("utf-8", errors="ignore"), block
                            elif isinstance(value, (bytes, bytearray)):
                                return value.decode("utf-8", errors="ignore"), block
                            elif isinstance(value, str):
                                return value, block
                elif isinstance(first_field, (bytes, bytearray)):
                    return first_field.decode("utf-8", errors="ignore"), block
                elif isinstance(first_field, str):
                    return first_field, block

        # Handle simple formats (no block info available)
        elif isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8", errors="ignore"), None
        elif isinstance(data, str):
            return data, None

        return None, None
    except Exception as e:
        logger.debug("commitment_query_failed", hotkey=hotkey[:16], error=str(e))
        return None, None


def read_miner_commitments(
    subtensor,  # bt.Subtensor
    netuid: int,
    neurons: list | None = None,  # List of NeuronInfo (optional, will fetch if not provided)
) -> list[MinerCommitment]:
    """
    Read all miner commitments from chain.

    Args:
        subtensor: Bittensor subtensor connection
        netuid: Subnet UID
        neurons: Optional pre-fetched neurons list (from subtensor.neurons())

    Returns:
        List of MinerCommitment for miners with valid commitments
    """
    # Use neurons() instead of metagraph() for compatibility with various substrate versions
    if neurons is None:
        neurons = subtensor.neurons(netuid=netuid)

    if not neurons:
        logger.warning("no_neurons_on_subnet", netuid=netuid)
        return []

    commitments = []
    n_neurons = len(neurons)

    for neuron in neurons:
        uid = neuron.uid
        hotkey = neuron.hotkey

        try:
            # Query commitment directly by hotkey to avoid broken SDK API
            raw, block = _query_commitment_by_hotkey(subtensor, netuid, hotkey)

            if raw:
                parsed = parse_commitment(raw)
                # Use block from commitment data if available, fallback to neuron.last_update
                committed_block = block if block is not None else neuron.last_update

                commitment = MinerCommitment(
                    uid=uid,
                    hotkey=hotkey,
                    huggingface_repo=parsed["huggingface_repo"],
                    revision_sha=parsed["revision_sha"],
                    committed_block=committed_block,
                )
                if commitment.is_valid:
                    commitments.append(commitment)
                    logger.debug(
                        "found_commitment",
                        uid=uid,
                        repo=commitment.huggingface_repo,
                        block=committed_block,
                    )
        except Exception as e:
            logger.warning("commitment_read_failed", uid=uid, error=str(e))

    logger.info("commitments_loaded", count=len(commitments), total_miners=n_neurons)
    return commitments


def commit_model(
    subtensor,  # bt.Subtensor
    wallet,  # bt.Wallet
    netuid: int,
    repo: str,
    revision: str,
) -> bool:
    """
    Commit model info to chain using compact colon-separated format.

    This is called by miners to register their model.

    Format: "user/repo:rev8char"

    The executor will download from HuggingFace and create deployments on-demand.

    Args:
        subtensor: Bittensor subtensor connection
        wallet: Miner's wallet
        netuid: Subnet UID
        repo: HuggingFace repository (user/model)
        revision: Commit SHA (will be truncated to 8 chars)

    Returns:
        True if commitment succeeded
    """
    # Truncate revision to 8 chars to fit within MAX_COMMITMENT_SIZE (128 bytes).
    # HuggingFace supports short SHA resolution like git.
    revision_short = revision[:8]

    # Validate no colons in fields (would break colon-separated format)
    if ":" in repo or ":" in revision_short:
        logger.error(
            "commitment_field_contains_colon",
            repo=repo,
            revision=revision_short,
        )
        return False

    # Build commitment data: repo:revision
    commitment_data = f"{repo}:{revision_short}"
    logger.info("commitment_data", data=commitment_data, length=len(commitment_data))

    # Validate commitment size fits chain limit
    if len(commitment_data) > MAX_COMMITMENT_SIZE:
        logger.error(
            "commitment_too_large",
            size=len(commitment_data),
            max_size=MAX_COMMITMENT_SIZE,
            repo_length=len(repo),
        )
        return False

    try:
        result = subtensor.set_commitment(
            wallet=wallet,
            netuid=netuid,
            data=commitment_data,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

        # Handle both bool and ExtrinsicResponse return types
        success = bool(result) if not hasattr(result, "is_success") else result.is_success
        if success:
            logger.info(
                "commitment_submitted",
                repo=repo,
                revision=revision[:8],
            )
        return success
    except Exception as e:
        logger.error("commitment_failed", error=str(e))
        return False
