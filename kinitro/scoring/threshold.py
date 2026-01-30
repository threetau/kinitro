"""Threshold calculation for first-commit advantage scoring."""

import math


def calculate_threshold(
    prior_score: float,
    sample_count: int,
    z_score: float = 1.5,
    min_gap: float = 0.02,
    max_gap: float = 0.10,
) -> float:
    """
    Calculate the score threshold a later miner must beat.

    Gap scales with standard error: more samples = smaller gap = easier to beat.
    """
    if sample_count <= 0:
        # No samples = maximum uncertainty = maximum gap
        gap = max_gap
    else:
        # Calculate standard error for a proportion
        p = prior_score
        # Clamp p away from 0 and 1 to avoid SE = 0
        p = max(0.01, min(0.99, p))
        se = math.sqrt(p * (1.0 - p) / sample_count)

        # Calculate gap from standard error
        gap = z_score * se

        # Apply bounds
        gap = max(gap, min_gap)
        gap = min(gap, max_gap)

    # Return threshold, capped at 1.0
    return min(prior_score + gap, 1.0)


def compute_miner_thresholds(
    miner_scores: dict[int, dict[str, float]],
    episodes_per_env: int | dict[str, int],
    z_score: float = 1.5,
    min_gap: float = 0.02,
    max_gap: float = 0.10,
) -> dict[int, dict[str, float]]:
    """
    Compute thresholds for all miners across all environments.

    Args:
        miner_scores: Dict mapping uid -> env_id -> success_rate
        episodes_per_env: Number of episodes per environment (int or dict per env)
        z_score: Z-score for confidence level
        min_gap: Minimum gap to require
        max_gap: Maximum gap cap

    Returns:
        Dict mapping uid -> env_id -> threshold
    """
    thresholds: dict[int, dict[str, float]] = {}

    for uid, env_scores in miner_scores.items():
        thresholds[uid] = {}
        for env_id, score in env_scores.items():
            # Get sample count for this environment
            if isinstance(episodes_per_env, int):
                n_samples = episodes_per_env
            else:
                n_samples = episodes_per_env.get(env_id, 50)

            thresholds[uid][env_id] = calculate_threshold(
                prior_score=score,
                sample_count=n_samples,
                z_score=z_score,
                min_gap=min_gap,
                max_gap=max_gap,
            )

    return thresholds
