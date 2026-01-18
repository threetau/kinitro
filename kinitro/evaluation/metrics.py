"""Metrics computation for evaluation results."""

from dataclasses import dataclass

import numpy as np

from kinitro.environments.base import EpisodeResult
from kinitro.evaluation.parallel import EvaluationResult


@dataclass
class AggregatedMetrics:
    """Aggregated metrics for a miner across all environments."""

    uid: int
    overall_success_rate: float
    overall_mean_reward: float
    per_env_success_rates: dict[str, float]
    per_env_mean_rewards: dict[str, float]
    total_episodes: int
    total_failures: int


def compute_success_rate(results: list[EpisodeResult]) -> float:
    """Compute success rate from episode results."""
    if not results:
        return 0.0
    return sum(1 for r in results if r.success) / len(results)


def compute_mean_reward(results: list[EpisodeResult]) -> float:
    """Compute mean episode reward."""
    if not results:
        return 0.0
    return sum(r.total_reward for r in results) / len(results)


def compute_reward_std(results: list[EpisodeResult]) -> float:
    """Compute standard deviation of episode rewards."""
    if len(results) < 2:
        return 0.0
    rewards = [r.total_reward for r in results]
    return float(np.std(rewards))


def compute_standard_error(success_rate: float, n_samples: int) -> float:
    """
    Compute standard error for success rate (binomial proportion).

    SE = sqrt(p * (1-p) / n)

    Args:
        success_rate: Proportion of successes
        n_samples: Number of samples

    Returns:
        Standard error of the success rate estimate
    """
    if n_samples <= 0:
        return 1.0  # Maximum uncertainty
    p = success_rate
    return float(np.sqrt(p * (1 - p) / n_samples))


def aggregate_results(
    evaluation_result: EvaluationResult,
) -> dict[int, AggregatedMetrics]:
    """
    Aggregate evaluation results into per-miner metrics.

    Args:
        evaluation_result: Complete evaluation results

    Returns:
        Dict mapping uid to AggregatedMetrics
    """
    aggregated: dict[int, AggregatedMetrics] = {}

    for uid, env_results in evaluation_result.miner_results.items():
        per_env_success = {}
        per_env_reward = {}
        total_success = 0
        total_episodes = 0
        total_reward = 0.0
        total_failures = 0

        for env_id, result in env_results.items():
            per_env_success[env_id] = result.success_rate
            per_env_reward[env_id] = result.mean_reward

            # Count successes from episode results
            if result.episode_results:
                success_count = sum(1 for r in result.episode_results if r.success)
                total_success += success_count
                total_episodes += len(result.episode_results)
                total_reward += sum(r.total_reward for r in result.episode_results)

            total_failures += result.episodes_failed

        # Compute overall metrics
        overall_success = total_success / total_episodes if total_episodes > 0 else 0.0
        overall_reward = total_reward / total_episodes if total_episodes > 0 else 0.0

        aggregated[uid] = AggregatedMetrics(
            uid=uid,
            overall_success_rate=overall_success,
            overall_mean_reward=overall_reward,
            per_env_success_rates=per_env_success,
            per_env_mean_rewards=per_env_reward,
            total_episodes=total_episodes,
            total_failures=total_failures,
        )

    return aggregated


def extract_score_matrix(
    evaluation_result: EvaluationResult,
    env_ids: list[str],
) -> tuple[list[int], np.ndarray]:
    """
    Extract score matrix from evaluation results.

    Args:
        evaluation_result: Complete evaluation results
        env_ids: Ordered list of environment IDs

    Returns:
        Tuple of (uids, score_matrix) where score_matrix[i, j] is
        miner i's success rate on environment j
    """
    uids = list(evaluation_result.miner_results.keys())
    n_miners = len(uids)
    n_envs = len(env_ids)

    score_matrix = np.zeros((n_miners, n_envs), dtype=np.float32)

    for i, uid in enumerate(uids):
        env_results = evaluation_result.miner_results.get(uid, {})
        for j, env_id in enumerate(env_ids):
            if env_id in env_results:
                score_matrix[i, j] = env_results[env_id].success_rate

    return uids, score_matrix


def compute_sample_counts(
    evaluation_result: EvaluationResult,
    env_ids: list[str],
) -> np.ndarray:
    """
    Get sample counts per environment for epsilon computation.

    Args:
        evaluation_result: Complete evaluation results
        env_ids: Ordered list of environment IDs

    Returns:
        Array of sample counts per environment
    """
    uids = list(evaluation_result.miner_results.keys())
    n_envs = len(env_ids)

    # Use minimum samples across miners for each environment
    sample_counts = np.zeros(n_envs, dtype=np.int32)

    for j, env_id in enumerate(env_ids):
        counts = []
        for uid in uids:
            env_results = evaluation_result.miner_results.get(uid, {})
            if env_id in env_results:
                counts.append(env_results[env_id].episodes_completed)
        if counts:
            sample_counts[j] = min(counts)

    return sample_counts
