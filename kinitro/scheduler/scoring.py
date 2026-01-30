"""Scoring utilities for computing weights from task results."""

import structlog

from kinitro.backend.models import TaskPoolORM
from kinitro.chain.commitments import MinerCommitment
from kinitro.chain.weights import weights_to_u16
from kinitro.scoring.pareto import compute_pareto_frontier
from kinitro.scoring.threshold import compute_miner_thresholds
from kinitro.scoring.winners_take_all import (
    compute_subset_scores_with_priority,
    scores_to_weights,
)

logger = structlog.get_logger()
PLACEHOLDER_BLOCK_NUM = 2**32  # Large block number for missing data


def aggregate_task_results(
    tasks: list[TaskPoolORM],
) -> dict[int, dict[str, float]]:
    """
    Aggregate task results into miner scores.

    Args:
        tasks: List of completed tasks

    Returns:
        Dict mapping uid -> env_id -> success_rate
    """
    # Group by (miner_uid, env_id)
    scores: dict[int, dict[str, list[float]]] = {}

    for task in tasks:
        if task.result is None:
            continue

        uid = task.miner_uid
        env_id = task.env_id

        if uid not in scores:
            scores[uid] = {}
        if env_id not in scores[uid]:
            scores[uid][env_id] = []

        # Score is 1.0 for success, 0.0 for failure
        success = task.result.get("success", False)
        scores[uid][env_id].append(1.0 if success else 0.0)

    # Average to get success rates
    result: dict[int, dict[str, float]] = {}
    for uid, env_scores in scores.items():
        result[uid] = {}
        for env_id, task_scores in env_scores.items():
            if task_scores:
                result[uid][env_id] = sum(task_scores) / len(task_scores)
            else:
                result[uid][env_id] = 0.0

    logger.info(
        "scores_aggregated",
        n_miners=len(result),
        n_tasks=len(tasks),
    )

    return result


def compute_weights(
    miner_scores: dict[int, dict[str, float]],
    env_ids: list[str],
    episodes_per_env: int,
    miners: dict[int, MinerCommitment],
    pareto_temperature: float = 1.0,
    threshold_z_score: float = 1.5,
    threshold_min_gap: float = 0.02,
    threshold_max_gap: float = 0.10,
) -> tuple[dict[int, float], dict[str, list[int]]]:
    """
    Compute weights from miner scores using Pareto frontier with first-commit advantage.

    Args:
        miner_scores: Dict mapping uid -> env_id -> success_rate
        env_ids: List of environment IDs
        episodes_per_env: Number of episodes per environment
        miners: Dict mapping uid -> MinerCommitment (for first_block info)
        pareto_temperature: Softmax temperature for weight conversion
        threshold_z_score: Z-score for threshold calculation
        threshold_min_gap: Minimum gap for threshold (default 2%)
        threshold_max_gap: Maximum gap for threshold (default 10%)

    Returns:
        Tuple of (weights_dict, weights_u16_dict)

    Raises:
        ValueError: If pareto_temperature is not positive
    """
    if pareto_temperature <= 0:
        raise ValueError("pareto_temperature must be > 0")

    if not miner_scores:
        logger.warning("no_miner_scores", msg="No miner scores to compute weights from")
        return {}, {"uids": [], "values": []}

    # Compute Pareto frontier (for logging/analysis)
    pareto_result = compute_pareto_frontier(
        miner_scores=miner_scores,
        env_ids=env_ids,
        n_samples_per_env=episodes_per_env,
    )

    logger.info(
        "pareto_frontier_computed",
        frontier_size=len(pareto_result.frontier_uids),
    )

    # Compute thresholds for each miner
    miner_thresholds = compute_miner_thresholds(
        miner_scores=miner_scores,
        episodes_per_env=episodes_per_env,
        z_score=threshold_z_score,
        min_gap=threshold_min_gap,
        max_gap=threshold_max_gap,
    )

    # Extract first_block for each miner
    miner_first_blocks = {
        uid: miners[uid].committed_block for uid in miner_scores.keys() if uid in miners
    }

    # Fill in missing first_blocks with a large value (disadvantaged)
    for uid in miner_scores.keys():
        if uid not in miner_first_blocks:
            miner_first_blocks[uid] = PLACEHOLDER_BLOCK_NUM

    logger.info(
        "first_commit_advantage",
        n_miners=len(miner_first_blocks),
        earliest_block=min(miner_first_blocks.values()) if miner_first_blocks else None,
    )

    # Compute winners-take-all scores with first-commit advantage
    subset_scores = compute_subset_scores_with_priority(
        miner_scores=miner_scores,
        miner_thresholds=miner_thresholds,
        miner_first_blocks=miner_first_blocks,
        env_ids=env_ids,
    )

    # Convert to weights
    weights = scores_to_weights(
        subset_scores,
        temperature=pareto_temperature,
    )

    # Convert to u16 for chain submission
    uids, values = weights_to_u16(weights)
    weights_u16 = {"uids": uids, "values": values}

    return weights, weights_u16


def convert_to_scores_data(
    miner_scores: dict[int, dict[str, float]],
    miners_by_uid: dict[int, str],  # uid -> hotkey
    episodes_per_env: int,
) -> list[dict]:
    """
    Convert miner scores to format for storage.

    Args:
        miner_scores: Dict mapping uid -> env_id -> success_rate
        miners_by_uid: Dict mapping uid -> hotkey
        episodes_per_env: Number of episodes per environment

    Returns:
        List of score dicts for bulk insert
    """
    scores_data = []

    for uid, env_scores in miner_scores.items():
        hotkey = miners_by_uid.get(uid, "unknown")

        for env_id, success_rate in env_scores.items():
            scores_data.append(
                {
                    "uid": uid,
                    "hotkey": hotkey,
                    "env_id": env_id,
                    "success_rate": success_rate,
                    # TODO: compute actual mean_reward from task results
                    "mean_reward": 0.0,
                    # TODO: compute actual episodes_completed/failed from task results
                    "episodes_completed": episodes_per_env,
                    "episodes_failed": 0,
                }
            )

    return scores_data
