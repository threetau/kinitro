"""Miner commitment handling for on-chain model registration.

Supports two commitment modes:
1. Plain: deployment_id is stored directly (public endpoint)
2. Encrypted: deployment_id is encrypted with backend operator's public key

Encrypted mode protects miner endpoints from public disclosure while allowing
the backend operator to decrypt and evaluate miners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, cast

import structlog
from bittensor import AsyncSubtensor, NeuronInfo, Subtensor
from bittensor_wallet import Wallet
from cryptography.hazmat.primitives.asymmetric import x25519

from kinitro.crypto import (
    decrypt_deployment_id,
    encrypt_deployment_id,
)
from kinitro.types import BlockNumber, Hotkey, MinerUID, ParsedCommitment

logger = structlog.get_logger()


# Basilica deployment URL template
BASILICA_URL_TEMPLATE = "https://{deployment_id}.deployments.basilica.ai"

# Chain commitment size limit (bytes)
MAX_COMMITMENT_SIZE = 128


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

    uid: MinerUID
    hotkey: Hotkey
    deployment_id: str  # Basilica deployment ID (UUID, not full URL) - decrypted if encrypted
    committed_block: BlockNumber
    encrypted_deployment: str | None = field(default=None)  # Base85 encrypted blob (if encrypted)

    @property
    def endpoint(self) -> str:
        """Get the full endpoint URL from deployment ID.

        Raises:
            ValueError: If deployment_id is empty (commitment needs decryption first)
        """
        if not self.deployment_id:
            raise ValueError(
                f"Cannot access endpoint: commitment requires decryption first "
                f"(encrypted={self.is_encrypted}, needs_decryption={self.needs_decryption})"
            )
        return deployment_id_to_url(self.deployment_id)

    @property
    def is_encrypted(self) -> bool:
        """Check if this commitment uses encrypted endpoint."""
        return self.encrypted_deployment is not None

    @property
    def is_valid(self) -> bool:
        """Check if commitment has all required fields."""
        return bool(self.deployment_id) or bool(self.encrypted_deployment)

    @property
    def needs_decryption(self) -> bool:
        """Check if this commitment needs decryption before use."""
        return bool(self.encrypted_deployment) and not bool(self.deployment_id)


def parse_commitment(raw: str) -> ParsedCommitment:
    """
    Parse raw commitment string from chain.

    New format:
        "deployment_id" (plain)
        "e:<base85_blob>" (encrypted)

    Legacy format (backward compat):
        "user/repo:rev8char:deployment_id" (plain)
        "user/repo:rev8char:e:<base85_blob>" (encrypted)

    Args:
        raw: Raw commitment string

    Returns:
        Dict with parsed fields:
            - deployment_id (for plain commitments)
            - encrypted_deployment (for encrypted commitments)
    """
    parts = raw.split(":", 3)

    # New encrypted format: "e:<base85_blob>"
    if len(parts) >= 2 and parts[0] == "e":
        encrypted_blob = raw[2:]  # Everything after "e:"
        return {
            "deployment_id": "",
            "encrypted_deployment": encrypted_blob,
        }

    # Legacy format: "repo:rev:deployment_id" or "repo:rev:e:<base85_blob>"
    if len(parts) >= 3:
        third_part = parts[2]

        # Legacy encrypted: repo:rev:e:<base85_blob>
        if third_part == "e" and len(parts) >= 4:
            encrypted_blob = parts[3]
            return {
                "deployment_id": "",
                "encrypted_deployment": encrypted_blob,
            }

        # Legacy plain: repo:rev:deployment_id
        return {
            "deployment_id": third_part,
            "encrypted_deployment": None,
        }

    # New plain format: just the deployment_id (no colons)
    if len(parts) == 1 and raw:
        return {
            "deployment_id": raw,
            "encrypted_deployment": None,
        }

    # Invalid format
    logger.warning("invalid_commitment_format", raw=raw)
    return {
        "deployment_id": "",
        "encrypted_deployment": None,
    }


def _parse_commitment_result(result: Any) -> tuple[str | None, int | None]:
    """Parse a raw commitment query result into (commitment_string, block_number).

    ``result`` comes from ``subtensor.query_module("Commitments", ...)`` which
    returns an untyped substrate response.  In practice this is either:
    - ``None`` when no commitment exists
    - A ``dict`` with keys ``"deposit"``, ``"block"``, ``"info"``
    - A SCALE-decoded object exposing a ``.value`` attribute (dict payload)
    We accept ``Any`` because the bittensor SDK does not export a concrete type
    for substrate query results.
    """
    if result is None:
        return None, None

    if isinstance(result, dict):
        data = result
    elif hasattr(result, "value"):
        data = result.value
    else:
        return None, None

    if not data:
        return None, None

    # Handle structured commitment format:
    # {'deposit': ..., 'block': ..., 'info': {'fields': ...}}
    if isinstance(data, dict) and "info" in data:
        data_dict = cast(dict[str, Any], data)
        block = data_dict.get("block")
        if block is not None:
            try:
                block = int(block)
            except (TypeError, ValueError):
                block = None

        info = data_dict.get("info", {})
        fields = info.get("fields", ()) if isinstance(info, dict) else ()

        if fields and len(fields) > 0:
            first_field = fields[0]

            if isinstance(first_field, tuple) and len(first_field) > 0:
                first_field = first_field[0]

            if isinstance(first_field, dict):
                for key, value in first_field.items():
                    if key.startswith("Raw") or key == "Data":
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

    elif isinstance(data, (bytes, bytearray)):
        return data.decode("utf-8", errors="ignore"), None
    elif isinstance(data, str):
        return data, None

    return None, None


def _query_commitment_by_hotkey(
    subtensor: Subtensor,
    netuid: int,
    hotkey: str,
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
        return _parse_commitment_result(result)
    except Exception as e:
        logger.debug("commitment_query_failed", hotkey=hotkey[:16], error=str(e))
        return None, None


async def _query_commitment_by_hotkey_async(
    subtensor: AsyncSubtensor,
    netuid: int,
    hotkey: str,
) -> tuple[str | None, int | None]:
    """Async version of :func:`_query_commitment_by_hotkey`."""
    try:
        result = await subtensor.query_module("Commitments", "CommitmentOf", [netuid, hotkey])
        return _parse_commitment_result(result)
    except Exception as e:
        logger.debug("commitment_query_failed", hotkey=hotkey[:16], error=str(e))
        return None, None


def read_miner_commitments(
    subtensor: Subtensor,
    netuid: int,
    neurons: list[NeuronInfo] | None = None,
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
                    except ValueError as e:
                        logger.warning(
                            "decryption_failed",
                            uid=uid,
                            error=str(e),
                        )
                        # Decryption failed - commitment will have needs_decryption=True.
                        # The encrypted_deployment blob is preserved so decrypt_commitments()
                        # can retry later with a different key if needed.

                commitment = MinerCommitment(
                    uid=MinerUID(uid),
                    hotkey=Hotkey(hotkey),
                    deployment_id=deployment_id,
                    committed_block=BlockNumber(committed_block),
                    encrypted_deployment=encrypted_deployment,
                )
                if commitment.is_valid:
                    commitments.append(commitment)
                    logger.debug(
                        "found_commitment",
                        uid=uid,
                        block=committed_block,
                        encrypted=commitment.is_encrypted,
                    )
        except Exception as e:
            logger.warning("commitment_read_failed", uid=uid, error=str(e))

    logger.info("commitments_loaded", count=len(commitments), total_miners=n_neurons)
    return commitments


async def read_miner_commitments_async(
    subtensor: AsyncSubtensor,
    netuid: int,
    neurons: list[NeuronInfo] | None = None,
    backend_private_key: x25519.X25519PrivateKey | None = None,
) -> list[MinerCommitment]:
    """Async version of :func:`read_miner_commitments`.

    Uses :class:`AsyncSubtensor` for non-blocking chain I/O.
    """
    if neurons is None:
        neurons = await subtensor.neurons(netuid=netuid)

    if not neurons:
        logger.warning("no_neurons_on_subnet", netuid=netuid)
        return []

    commitments = []
    n_neurons = len(neurons)

    for neuron in neurons:
        uid = neuron.uid
        hotkey = neuron.hotkey

        try:
            raw, block = await _query_commitment_by_hotkey_async(subtensor, netuid, hotkey)

            if raw:
                parsed = parse_commitment(raw)
                committed_block = block if block is not None else neuron.last_update

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
                    except ValueError as e:
                        logger.warning(
                            "decryption_failed",
                            uid=uid,
                            error=str(e),
                        )

                commitment = MinerCommitment(
                    uid=MinerUID(uid),
                    hotkey=Hotkey(hotkey),
                    deployment_id=deployment_id,
                    committed_block=BlockNumber(committed_block),
                    encrypted_deployment=encrypted_deployment,
                )
                if commitment.is_valid:
                    commitments.append(commitment)
                    logger.debug(
                        "found_commitment",
                        uid=uid,
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
            except ValueError as e:
                logger.warning(
                    "decryption_failed",
                    uid=commitment.uid,
                    error=str(e),
                )
    return commitments


def _build_commitment_data(
    deployment_id: str,
    backend_public_key: str | None = None,
) -> str | None:
    """Build the commitment string.

    Plain format: ``deployment_id``
    Encrypted format: ``e:<base85_blob>``

    Returns the commitment data string, or None if validation/encryption fails.
    """
    if ":" in deployment_id:
        logger.error(
            "commitment_field_contains_colon",
            deployment_id=deployment_id,
        )
        return None

    if backend_public_key:
        try:
            encrypted_blob = encrypt_deployment_id(deployment_id, backend_public_key)
            commitment_data = f"e:{encrypted_blob}"
            logger.info(
                "commitment_encrypted",
                data_length=len(commitment_data),
                encrypted_blob_length=len(encrypted_blob),
            )
        except Exception as e:
            logger.exception("encryption_failed", error=str(e))
            return None
    else:
        commitment_data = deployment_id
        logger.info("commitment_data", data=commitment_data, length=len(commitment_data))

    if len(commitment_data) > MAX_COMMITMENT_SIZE:
        logger.error(
            "commitment_too_large",
            size=len(commitment_data),
            max_size=MAX_COMMITMENT_SIZE,
        )
        return None

    return commitment_data


def commit_model(
    subtensor: Subtensor,
    wallet: Wallet,
    netuid: int,
    deployment_id: str,
    backend_public_key: str | None = None,
) -> bool:
    """
    Commit deployment info to chain.

    This is called by miners to register their deployment.

    Format:
        - Plain: "deployment_id"
        - Encrypted: "e:<base85_blob>"

    Args:
        subtensor: Bittensor subtensor connection
        wallet: Miner's wallet
        netuid: Subnet UID
        deployment_id: Basilica deployment ID (UUID only, not full URL)
        backend_public_key: Optional hex-encoded X25519 public key for encrypting endpoint.
                           If provided, the deployment_id will be encrypted so only
                           the backend operator can decrypt it.

    Returns:
        True if commitment succeeded
    """
    commitment_data = _build_commitment_data(deployment_id, backend_public_key)
    if commitment_data is None:
        return False

    try:
        result = subtensor.set_commitment(
            wallet=wallet,
            netuid=netuid,
            data=commitment_data,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

        success = result.success
        if success:
            logger.info(
                "commitment_submitted",
                deployment_id=deployment_id[:8] + "..." if deployment_id else None,
                encrypted=bool(backend_public_key),
            )
        return success
    except Exception as e:
        logger.error("commitment_failed", error=str(e))
        return False


async def commit_model_async(
    subtensor: AsyncSubtensor,
    wallet: Wallet,
    netuid: int,
    deployment_id: str,
    backend_public_key: str | None = None,
) -> bool:
    """Async version of :func:`commit_model`.

    Uses :class:`AsyncSubtensor` for non-blocking chain I/O.
    """
    commitment_data = _build_commitment_data(deployment_id, backend_public_key)
    if commitment_data is None:
        return False

    try:
        result = await subtensor.set_commitment(
            wallet=wallet,
            netuid=netuid,
            data=commitment_data,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

        success = result.success
        if success:
            logger.info(
                "commitment_submitted",
                deployment_id=deployment_id[:8] + "..." if deployment_id else None,
                encrypted=bool(backend_public_key),
            )
        return success
    except Exception as e:
        logger.error("commitment_failed", error=str(e))
        return False
