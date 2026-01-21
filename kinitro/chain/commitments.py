"""Miner commitment handling for on-chain model registration."""

from dataclasses import dataclass

import structlog

logger = structlog.get_logger()


@dataclass
class MinerCommitment:
    """Parsed miner commitment from chain."""

    uid: int
    hotkey: str
    huggingface_repo: str
    revision_sha: str
    chute_id: str
    docker_image: str
    committed_block: int

    @property
    def is_valid(self) -> bool:
        """Check if commitment has all required fields.

        Miners must deploy to Chutes - docker_image alone is not sufficient
        for production evaluations.
        """
        return bool(self.huggingface_repo and self.revision_sha and self.chute_id)


def parse_commitment(raw: str) -> dict:
    """
    Parse raw commitment string from chain.

    Supports two formats:
    1. JSON (preferred, Affine-compatible): {"model": "user/repo", "revision": "sha", "chute_id": "id"}
    2. Legacy colon-separated: "huggingface_repo:revision_sha:chute_id[:docker_image]"

    Args:
        raw: Raw commitment string

    Returns:
        Dict with parsed fields: huggingface_repo, revision_sha, chute_id, docker_image
    """
    import json

    # Try JSON format first (Affine-compatible)
    if raw.strip().startswith("{"):
        try:
            data = json.loads(raw)
            hf_repo = data.get("model", "")
            revision = data.get("revision", "")
            chute_id = data.get("chute_id", "")
            docker_image = (
                data.get("docker_image", "") or f"{hf_repo}:{revision}" if hf_repo else ""
            )

            if hf_repo and revision:
                return {
                    "huggingface_repo": hf_repo,
                    "revision_sha": revision,
                    "chute_id": chute_id,
                    "docker_image": docker_image,
                }
        except json.JSONDecodeError:
            pass

    # Fallback to legacy colon-separated format
    parts = raw.split(":")

    if len(parts) >= 3:
        hf_repo = parts[0]
        revision = parts[1]
        chute_id = parts[2]
        docker_image = parts[3] if len(parts) > 3 else f"{hf_repo}:{revision}"

        return {
            "huggingface_repo": hf_repo,
            "revision_sha": revision,
            "chute_id": chute_id,
            "docker_image": docker_image,
        }

    # Invalid format
    logger.warning("invalid_commitment_format", raw=raw)
    return {
        "huggingface_repo": "",
        "revision_sha": "",
        "chute_id": "",
        "docker_image": "",
    }


def _query_commitment_by_hotkey(subtensor, netuid: int, hotkey: str) -> str | None:
    """
    Query commitment directly from chain storage by hotkey.

    This bypasses the broken get_commitment API in bittensor SDK.

    Args:
        subtensor: Bittensor subtensor connection
        netuid: Subnet UID
        hotkey: Hotkey SS58 address

    Returns:
        Commitment string or None
    """
    try:
        result = subtensor.substrate.query("Commitments", "CommitmentOf", [netuid, hotkey])

        # Handle different result types
        if result is None:
            return None

        # Result might be a dict directly (newer substrate interface)
        if isinstance(result, dict):
            data = result
        elif hasattr(result, "value"):
            data = result.value
        else:
            return None

        if not data:
            return None

        # Handle structured commitment format: {'deposit': ..., 'block': ..., 'info': {'fields': ...}}
        if isinstance(data, dict) and "info" in data:
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
                                    return bytes(byte_data).decode("utf-8", errors="ignore")
                            elif isinstance(value, (bytes, bytearray)):
                                return value.decode("utf-8", errors="ignore")
                            elif isinstance(value, str):
                                return value
                elif isinstance(first_field, (bytes, bytearray)):
                    return first_field.decode("utf-8", errors="ignore")
                elif isinstance(first_field, str):
                    return first_field

        # Handle simple formats
        elif isinstance(data, (bytes, bytearray)):
            return data.decode("utf-8", errors="ignore")
        elif isinstance(data, str):
            return data

        return None
    except Exception as e:
        logger.debug("commitment_query_failed", hotkey=hotkey[:16], error=str(e))
        return None


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
            raw = _query_commitment_by_hotkey(subtensor, netuid, hotkey)

            if raw:
                parsed = parse_commitment(raw)
                commitment = MinerCommitment(
                    uid=uid,
                    hotkey=hotkey,
                    huggingface_repo=parsed["huggingface_repo"],
                    revision_sha=parsed["revision_sha"],
                    chute_id=parsed["chute_id"],
                    docker_image=parsed["docker_image"],
                    committed_block=neuron.last_update,
                )
                if commitment.is_valid:
                    commitments.append(commitment)
                    logger.debug(
                        "found_commitment",
                        uid=uid,
                        repo=commitment.huggingface_repo,
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
    chute_id: str,
) -> bool:
    """
    Commit model info to chain using JSON format (Affine-compatible).

    This is called by miners to register their model.

    Args:
        subtensor: Bittensor subtensor connection
        wallet: Miner's wallet
        netuid: Subnet UID
        repo: HuggingFace repository (user/model)
        revision: Commit SHA
        chute_id: Chutes deployment ID

    Returns:
        True if commitment succeeded
    """
    import json

    # Use JSON format compatible with Affine
    commitment_data = json.dumps(
        {
            "model": repo,
            "revision": revision,
            "chute_id": chute_id,
        }
    )

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
                revision=revision[:12],
                chute_id=chute_id,
            )
        return success
    except Exception as e:
        logger.error("commitment_failed", error=str(e))
        return False
