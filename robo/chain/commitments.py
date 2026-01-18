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
        """Check if commitment has all required fields."""
        return bool(
            self.huggingface_repo and self.revision_sha and (self.chute_id or self.docker_image)
        )


def parse_commitment(raw: str) -> dict:
    """
    Parse raw commitment string from chain.

    Expected format: "huggingface_repo:revision_sha:chute_id"
    Or extended format with docker: "hf_repo:rev:chute:docker_image"

    Args:
        raw: Raw commitment string

    Returns:
        Dict with parsed fields
    """
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


def read_miner_commitments(
    subtensor,  # bt.Subtensor
    netuid: int,
    metagraph=None,  # bt.Metagraph (optional, will fetch if not provided)
) -> list[MinerCommitment]:
    """
    Read all miner commitments from chain.

    Args:
        subtensor: Bittensor subtensor connection
        netuid: Subnet UID
        metagraph: Optional pre-fetched metagraph

    Returns:
        List of MinerCommitment for miners with valid commitments
    """
    if metagraph is None:
        metagraph = subtensor.metagraph(netuid)

    commitments = []

    for uid in range(metagraph.n):
        try:
            raw = subtensor.get_commitment(netuid, uid)
            if raw:
                parsed = parse_commitment(raw)
                commitment = MinerCommitment(
                    uid=uid,
                    hotkey=metagraph.hotkeys[uid],
                    huggingface_repo=parsed["huggingface_repo"],
                    revision_sha=parsed["revision_sha"],
                    chute_id=parsed["chute_id"],
                    docker_image=parsed["docker_image"],
                    committed_block=metagraph.last_update[uid],
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

    logger.info("commitments_loaded", count=len(commitments), total_miners=metagraph.n)
    return commitments


def commit_model(
    subtensor,  # bt.Subtensor
    wallet,  # bt.Wallet
    netuid: int,
    repo: str,
    revision: str,
    chute_id: str,
    docker_image: str | None = None,
) -> bool:
    """
    Commit model info to chain.

    This is called by miners to register their model.
    Rate limited to ~1 per 100 blocks (~20 minutes).

    Args:
        subtensor: Bittensor subtensor connection
        wallet: Miner's wallet
        netuid: Subnet UID
        repo: HuggingFace repository (user/model)
        revision: Commit SHA
        chute_id: Chutes deployment ID
        docker_image: Optional Docker image URL

    Returns:
        True if commitment succeeded
    """
    if docker_image:
        commitment = f"{repo}:{revision}:{chute_id}:{docker_image}"
    else:
        commitment = f"{repo}:{revision}:{chute_id}"

    try:
        success = subtensor.set_commitment(
            wallet=wallet,
            netuid=netuid,
            commitment=commitment,
        )
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
