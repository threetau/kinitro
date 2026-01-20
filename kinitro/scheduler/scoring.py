"""Scoring utilities for computing weights from task results."""

import structlog

from kinitro.backend.models import TaskPoolORM
from kinitro.scoring.pareto import compute_pareto_frontier
from kinitro.scoring.winners_take_all import compute_subset_scores, scores_to_weights
from kinitro.chain.weights import weights_to_u16

logger = structlog.get_logger()


def aggregate_task_results(
    tasks: list[TaskPoolORM],
    env_ids: list[str],
) -> dict[int, dict[str, float]]:
    """
    Aggregate task results into miner scores.

    Args:
        tasks: List of completed tasks
        env_ids: List of environment IDs

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
    pareto_temperature: float = 1.0,
) -> tuple[dict[int, float], dict[str, list[int]]]:
    """
    Compute weights from miner scores using Pareto frontier.

    Args:
        miner_scores: Dict mapping uid -> env_id -> success_rate
        env_ids: List of environment IDs
        episodes_per_env: Number of episodes per environment
        pareto_temperature: Softmax temperature for weight conversion

    Returns:
        Tuple of (weights_dict, weights_u16_dict)

    Raises:
        ValueError: If pareto_temperature is not positive
    """
    if pareto_temperature <= 0:
        raise ValueError("pareto_temperature must be > 0")

    # Compute Pareto frontier
    pareto_result = compute_pareto_frontier(
        miner_scores=miner_scores,
        env_ids=env_ids,
        n_samples_per_env=episodes_per_env,
    )

    logger.info(
        "pareto_frontier_computed",
        frontier_size=len(pareto_result.frontier_uids),
    )

    # Compute winners-take-all scores
    epsilons = {env_id: float(pareto_result.epsilons[i]) for i, env_id in enumerate(env_ids)}

    subset_scores = compute_subset_scores(
        miner_scores=miner_scores,
        env_ids=env_ids,
        epsilons=epsilons,
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
                    "mean_reward": 0.0,  # Not tracked in task pool model
                    "episodes_completed": episodes_per_env,
                    "episodes_failed": 0,
                }
            )

    return scores_data
