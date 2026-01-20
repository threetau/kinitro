"""Task generator for creating evaluation tasks."""

import structlog

from kinitro.chain.commitments import MinerCommitment

logger = structlog.get_logger()


def get_miner_endpoint(miner: MinerCommitment) -> str:
    """
    Build miner's base URL from commitment.

    Miners can deploy to:
    1. Chutes: chute_id -> https://{slug}.chutes.ai
    2. Self-hosted: chute_id can be a full URL (http://... or https://...)
    """
    if miner.chute_id:
        # If chute_id is already a URL, use it directly (for testing/self-hosted)
        if miner.chute_id.startswith("http://") or miner.chute_id.startswith("https://"):
            return miner.chute_id.rstrip("/")

        # Otherwise, build Chutes URL from slug
        slug = miner.chute_id.replace("_", "-").lower()
        return f"https://{slug}.chutes.ai"

    raise ValueError(
        f"Miner {miner.uid} has no chute_id. Miners must deploy their policy to Chutes."
    )


def generate_tasks(
    miners: list[MinerCommitment],
    env_ids: list[str],
    episodes_per_env: int,
    block_number: int,
    cycle_id: int,
) -> list[dict]:
    """
    Generate evaluation tasks for all miners and environments.

    Args:
        miners: List of miner commitments
        env_ids: List of environment IDs to evaluate
        episodes_per_env: Number of episodes per environment
        block_number: Current block number (used for seed generation)
        cycle_id: ID of the evaluation cycle

    Returns:
        List of task dicts ready for bulk insert
    """
    tasks = []

    for miner in miners:
        try:
            endpoint = get_miner_endpoint(miner)
        except ValueError as e:
            logger.warning("miner_no_endpoint", uid=miner.uid, error=str(e))
            continue

        for env_id in env_ids:
            for i in range(episodes_per_env):
                # Generate deterministic task ID from block number
                task_id = hash(f"{block_number}:{env_id}:{miner.uid}:{i}") % (2**31)

                tasks.append(
                    {
                        "cycle_id": cycle_id,
                        "miner_uid": miner.uid,
                        "miner_hotkey": miner.hotkey,
                        "miner_endpoint": endpoint,
                        "env_id": env_id,
                        "task_id": task_id,
                    }
                )

    logger.info(
        "tasks_generated",
        n_miners=len(miners),
        n_envs=len(env_ids),
        episodes_per_env=episodes_per_env,
        total_tasks=len(tasks),
    )

    return tasks
