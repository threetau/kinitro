"""Miner commitment handling for on-chain model registration.

Supports two commitment modes:
1. Plain: deployment_id is stored directly (public endpoint)
2. Encrypted: deployment_id is encrypted with backend operator's public key

Encrypted mode protects miner endpoints from public disclosure while allowing
the backend operator to decrypt and evaluate miners.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import structlog
from cryptography.hazmat.primitives.asymmetric import x25519

from kinitro.crypto import (
    decrypt_deployment_id,
    encrypt_deployment_id,
    get_encrypted_blob,
)

logger = structlog.get_logger()


# Basilica deployment URL template
BASILICA_URL_TEMPLATE = "https://{deployment_id}.deployments.basilica.ai"


def deployment_id_to_url(deployment_id: str) -> str:
    """Convert a Basilica deployment ID to a full URL."""
    # If already a full URL, return as-is
    if deployment_id.startswith("http://") or deployment_id.startswith("https://"):
        return deployment_id.rstrip("/")
    # Otherwise, construct the Basilica URL
    return BASILICA_URL_TEMPLATE.format(deployment_id=deployment_id)


@dataclass
class MinerCommitment:
    """Parsed miner commitment from chain.

    Supports both plain and encrypted deployment IDs:
    - Plain: deployment_id contains the UUID directly
    - Encrypted: encrypted_deployment contains the encrypted blob,
                 deployment_id is populated after decryption
    """

    uid: int
    hotkey: str
    huggingface_repo: str
    revision_sha: str
    deployment_id: str  # Basilica deployment ID (UUID, not full URL) - decrypted if encrypted
    docker_image: str
    committed_block: int
    encrypted_deployment: str | None = field(default=None)  # Base85 encrypted blob (if encrypted)

    @property
    def endpoint(self) -> str:
        """Get the full endpoint URL from deployment ID."""
        return deployment_id_to_url(self.deployment_id)

    @property
    def is_encrypted(self) -> bool:
        """Check if this commitment uses encrypted endpoint."""
        return self.encrypted_deployment is not None

    @property
    def is_valid(self) -> bool:
        """Check if commitment has all required fields.

        For encrypted commitments, deployment_id may be empty until decrypted.
        """
        has_basic_fields = bool(self.huggingface_repo and self.revision_sha)
        has_endpoint = bool(self.deployment_id) or bool(self.encrypted_deployment)
        return has_basic_fields and has_endpoint

    @property
    def needs_decryption(self) -> bool:
        """Check if this commitment needs decryption before use."""
        return bool(self.encrypted_deployment) and not bool(self.deployment_id)


def parse_commitment(raw: str) -> dict:
    """
    Parse raw commitment string from chain.

    Supports multiple formats:
    1. JSON with encrypted endpoint: {"m": "user/repo", "r": "sha", "e": "<base85>"}
    2. JSON with plain deployment_id: {"m": "user/repo", "r": "sha", "d": "uuid"}
    3. Legacy colon-separated: "huggingface_repo:revision_sha:deployment_id[:docker_image]"

    Args:
        raw: Raw commitment string

    Returns:
        Dict with parsed fields:
            - huggingface_repo, revision_sha, docker_image (always)
            - deployment_id (for plain commitments)
            - encrypted_deployment (for encrypted commitments)
    """
    # Try JSON format first
    # Supports both full keys and short keys for compactness:
    #   Plain: {"m": "...", "r": "...", "d": "..."}
    #   Encrypted: {"m": "...", "r": "...", "e": "..."}
    if raw.strip().startswith("{"):
        try:
            data = json.loads(raw)
            # Support both full and short keys
            hf_repo = data.get("model", "") or data.get("m", "")
            revision = data.get("revision", "") or data.get("r", "")
            docker_image = f"{hf_repo}:{revision}" if hf_repo else ""

            if hf_repo and revision:
                # Check for encrypted endpoint first
                encrypted_blob = get_encrypted_blob(data)
                if encrypted_blob:
                    return {
                        "huggingface_repo": hf_repo,
                        "revision_sha": revision,
                        "deployment_id": "",  # Will be decrypted later
                        "encrypted_deployment": encrypted_blob,
                        "docker_image": docker_image,
                    }

                # Plain deployment_id
                deployment_id = data.get("deployment_id", "") or data.get("d", "")
                return {
                    "huggingface_repo": hf_repo,
                    "revision_sha": revision,
                    "deployment_id": deployment_id,
                    "encrypted_deployment": None,
                    "docker_image": docker_image,
                }
        except json.JSONDecodeError:
            pass

    # Fallback to legacy colon-separated format (always plain)
    parts = raw.split(":")

    if len(parts) >= 3:
        hf_repo = parts[0]
        revision = parts[1]
        deployment_id = parts[2]
        docker_image = parts[3] if len(parts) > 3 else f"{hf_repo}:{revision}"

        return {
            "huggingface_repo": hf_repo,
            "revision_sha": revision,
            "deployment_id": deployment_id,
            "encrypted_deployment": None,
            "docker_image": docker_image,
        }

    # Invalid format
    logger.warning("invalid_commitment_format", raw=raw)
    return {
        "huggingface_repo": "",
        "revision_sha": "",
        "deployment_id": "",
        "encrypted_deployment": None,
        "docker_image": "",
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
    backend_private_key: x25519.X25519PrivateKey | None = None,
) -> list[MinerCommitment]:
    """
    Read all miner commitments from chain.

    If backend_private_key is provided, encrypted commitments will be decrypted.
    Otherwise, encrypted commitments will have empty deployment_id and need
    separate decryption via decrypt_commitments().

    Args:
        subtensor: Bittensor subtensor connection
        netuid: Subnet UID
        neurons: Optional pre-fetched neurons list (from subtensor.neurons())
        backend_private_key: Optional X25519 private key for decrypting endpoints

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

                # Handle encrypted deployment if private key provided
                deployment_id = parsed["deployment_id"]
                encrypted_deployment = parsed.get("encrypted_deployment")

                if encrypted_deployment and backend_private_key:
                    try:
                        deployment_id = decrypt_deployment_id(
                            encrypted_deployment, backend_private_key
                        )
                        logger.debug(
                            "decrypted_endpoint",
                            uid=uid,
                            deployment_id=deployment_id[:12] + "...",
                        )
                    except Exception as e:
                        logger.warning(
                            "decryption_failed",
                            uid=uid,
                            error=str(e),
                        )
                        # Decryption failed - commitment will have needs_decryption=True.
                        # The encrypted_deployment blob is preserved so decrypt_commitments()
                        # can retry later with a different key if needed.

                commitment = MinerCommitment(
                    uid=uid,
                    hotkey=hotkey,
                    huggingface_repo=parsed["huggingface_repo"],
                    revision_sha=parsed["revision_sha"],
                    deployment_id=deployment_id,
                    docker_image=parsed["docker_image"],
                    committed_block=committed_block,
                    encrypted_deployment=encrypted_deployment,
                )
                if commitment.is_valid:
                    commitments.append(commitment)
                    logger.debug(
                        "found_commitment",
                        uid=uid,
                        repo=commitment.huggingface_repo,
                        block=committed_block,
                        encrypted=commitment.is_encrypted,
                    )
        except Exception as e:
            logger.warning("commitment_read_failed", uid=uid, error=str(e))

    logger.info("commitments_loaded", count=len(commitments), total_miners=n_neurons)
    return commitments


def decrypt_commitments(
    commitments: list[MinerCommitment],
    backend_private_key: x25519.X25519PrivateKey,
) -> list[MinerCommitment]:
    """
    Decrypt encrypted commitments in-place.

    Args:
        commitments: List of MinerCommitment (may have encrypted endpoints)
        backend_private_key: X25519 private key for decryption

    Returns:
        Same list with deployment_id populated for encrypted commitments
    """
    for commitment in commitments:
        if commitment.needs_decryption:
            try:
                commitment.deployment_id = decrypt_deployment_id(
                    commitment.encrypted_deployment,  # type: ignore
                    backend_private_key,
                )
                logger.debug(
                    "decrypted_endpoint",
                    uid=commitment.uid,
                    deployment_id=commitment.deployment_id[:12] + "...",
                )
            except Exception as e:
                logger.warning(
                    "decryption_failed",
                    uid=commitment.uid,
                    error=str(e),
                )
    return commitments


def commit_model(
    subtensor,  # bt.Subtensor
    wallet,  # bt.Wallet
    netuid: int,
    repo: str,
    revision: str,
    deployment_id: str,
    backend_public_key: str | None = None,
) -> bool:
    """
    Commit model info to chain using JSON format.

    This is called by miners to register their model.

    Args:
        subtensor: Bittensor subtensor connection
        wallet: Miner's wallet
        netuid: Subnet UID
        repo: HuggingFace repository (user/model)
        revision: Commit SHA
        deployment_id: Basilica deployment ID (UUID only, not full URL)
        backend_public_key: Optional hex-encoded X25519 public key for encrypting endpoint.
                           If provided, the deployment_id will be encrypted so only
                           the backend operator can decrypt it.

    Returns:
        True if commitment succeeded
    """
    # Build commitment data
    if backend_public_key:
        # Encrypt the deployment ID
        try:
            encrypted_blob = encrypt_deployment_id(deployment_id, backend_public_key)
            commitment_data = json.dumps(
                {
                    "m": repo,
                    "r": revision,
                    "e": encrypted_blob,  # 'e' for encrypted endpoint
                },
                separators=(",", ":"),
            )
            logger.info(
                "commitment_encrypted",
                data_length=len(commitment_data),
                encrypted_blob_length=len(encrypted_blob),
            )
        except Exception as e:
            logger.exception("encryption_failed", error=str(e))
            return False
    else:
        # Plain commitment (deployment_id visible on-chain)
        commitment_data = json.dumps(
            {
                "m": repo,
                "r": revision,
                "d": deployment_id,
            },
            separators=(",", ":"),
        )
        logger.info("commitment_data", data=commitment_data, length=len(commitment_data))

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
                deployment_id=deployment_id[:12] + "..." if deployment_id else None,
                encrypted=bool(backend_public_key),
            )
        return success
    except Exception as e:
        logger.error("commitment_failed", error=str(e))
        return False
