"""Task generator for creating evaluation tasks."""

import hashlib
import uuid

import structlog

from kinitro.chain.commitments import MinerCommitment

logger = structlog.get_logger()


def generate_seed(task_uuid: str) -> int:
    """
    Generate a deterministic seed from task UUID.

    The UUID is random (unpredictable by miners), but the seed derived from it
    is deterministic - given the same task_uuid, you always get the same seed.
    This provides:
    - Unpredictability: miners can't pre-compute seeds for future tasks
    - Reproducibility: same task_uuid always produces same environment

    Args:
        task_uuid: Random UUID assigned to the task

    Returns:
        Positive 32-bit signed integer seed (0 to 2^31-1)
    """
    hash_bytes = hashlib.sha256(task_uuid.encode()).digest()[:4]
    # Mask to 31 bits to fit PostgreSQL signed int4 (max 2,147,483,647)
    return int.from_bytes(hash_bytes, byteorder="big") & 0x7FFFFFFF


def generate_tasks(
    miners: list[MinerCommitment],
    env_ids: list[str],
    episodes_per_env: int,
    block_number: int,
    cycle_id: int,
) -> list[dict]:
    """
    Generate evaluation tasks for all miners and environments.

    Each task gets:
    - task_uuid: Random UUID (unpredictable by miners)
    - seed: Derived from task_uuid via SHA256 (deterministic given UUID)

    This design ensures:
    - Miners cannot predict future seeds (UUID is random)
    - Evaluations are reproducible (same UUID = same seed = same environment)

    Note: miner_endpoint is set to None. The executor will create deployments
    on-demand using miner_repo and miner_revision.

    Args:
        miners: List of miner commitments
        env_ids: List of environment IDs to evaluate
        episodes_per_env: Number of episodes per environment
        block_number: Current block number (logged for reference)
        cycle_id: ID of the evaluation cycle

    Returns:
        List of task dicts ready for bulk insert
    """
    tasks = []

    for miner in miners:
        if not miner.is_valid:
            logger.warning("miner_invalid_commitment", uid=miner.uid)
            continue

        for env_id in env_ids:
            for i in range(episodes_per_env):
                # Generate random UUID - this is the source of unpredictability
                task_uuid = str(uuid.uuid4())

                # Derive seed from UUID - deterministic, but unpredictable
                # since it depends on the random UUID
                seed = generate_seed(task_uuid)

                tasks.append(
                    {
                        "task_uuid": task_uuid,
                        "cycle_id": cycle_id,
                        "miner_uid": miner.uid,
                        "miner_hotkey": miner.hotkey,
                        "miner_endpoint": None,  # Resolved at execution time by executor
                        "miner_repo": miner.huggingface_repo,
                        "miner_revision": miner.revision_sha,
                        "env_id": env_id,
                        "seed": seed,
                    }
                )

    logger.info(
        "tasks_generated",
        n_miners=len(miners),
        n_envs=len(env_ids),
        episodes_per_env=episodes_per_env,
        total_tasks=len(tasks),
        block_number=block_number,
    )

    return tasks
