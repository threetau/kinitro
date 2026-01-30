"""Winners-take-all scoring over environment subsets."""

from itertools import combinations

import numpy as np


def dominates_on_subset(
    a_scores: dict[str, float],
    b_scores: dict[str, float],
    subset: tuple[str, ...],
    epsilons: dict[str, float],
) -> bool:
    """
    Check if miner A dominates miner B on a specific environment subset.

    Args:
        a_scores: Miner A's scores per environment
        b_scores: Miner B's scores per environment
        subset: Tuple of environment IDs to check
        epsilons: Epsilon tolerance per environment

    Returns:
        True if A dominates B on this subset
    """
    a_vals = np.array([a_scores.get(env, 0.0) for env in subset])
    b_vals = np.array([b_scores.get(env, 0.0) for env in subset])
    eps = np.array([epsilons.get(env, 0.05) for env in subset])

    # A must be >= B - ε on all envs in subset
    if not np.all(a_vals >= b_vals - eps):
        return False

    # A must be > B + ε on at least one env in subset
    if not np.any(a_vals > b_vals + eps):
        return False

    return True


def find_subset_winner(
    miner_scores: dict[int, dict[str, float]],
    subset: tuple[str, ...],
    epsilons: dict[str, float],
) -> int | None:
    """
    Find the miner that ε-dominates all others on a specific subset.

    Args:
        miner_scores: Dict mapping uid -> env_id -> score
        subset: Tuple of environment IDs
        epsilons: Epsilon tolerance per environment

    Returns:
        UID of the winner, or None if no clear winner (ties)
    """
    uids = list(miner_scores.keys())

    for candidate in uids:
        dominates_all = True
        for other in uids:
            if candidate == other:
                continue
            if not dominates_on_subset(
                miner_scores[candidate],
                miner_scores[other],
                subset,
                epsilons,
            ):
                dominates_all = False
                break

        if dominates_all:
            return candidate

    return None  # No clear winner (ties exist)


def compute_subset_scores(
    miner_scores: dict[int, dict[str, float]],
    env_ids: list[str],
    epsilons: dict[str, float],
    subset_weight_scheme: str = "linear",
) -> dict[int, float]:
    """
    Compute winners-take-all scores over all environment subsets.

    For each non-empty subset S of environments:
    1. Find the miner that dominates on S
    2. Award them a score K_|S| based on subset size

    This rewards generalists who dominate on larger subsets more
    than specialists who only dominate on single environments.

    Args:
        miner_scores: Dict mapping uid -> env_id -> score
        env_ids: List of environment IDs
        epsilons: Epsilon tolerance per environment
        subset_weight_scheme: How to weight subsets:
            - "linear": K_s = s (subset size)
            - "exponential": K_s = 2^(s-1)
            - "equal": K_s = 1 (all subsets equal)

    Returns:
        Dict mapping uid -> total score
    """
    uids = list(miner_scores.keys())
    final_scores = {uid: 0.0 for uid in uids}

    if not uids or not env_ids:
        return final_scores

    # Iterate over all non-empty subsets
    for subset_size in range(1, len(env_ids) + 1):
        # Compute weight for this subset size
        if subset_weight_scheme == "linear":
            subset_weight = float(subset_size)
        elif subset_weight_scheme == "exponential":
            subset_weight = float(2 ** (subset_size - 1))
        elif subset_weight_scheme == "equal":
            subset_weight = 1.0
        else:
            subset_weight = float(subset_size)

        # Check each subset of this size
        for subset in combinations(env_ids, subset_size):
            winner = find_subset_winner(miner_scores, subset, epsilons)
            if winner is not None:
                final_scores[winner] += subset_weight

    return final_scores


def scores_to_weights(
    scores: dict[int, float],
    temperature: float = 1.0,
    min_weight: float = 0.0,
) -> dict[int, float]:
    """
    Convert scores to normalized weights via softmax.

    Args:
        scores: Dict mapping uid -> score
        temperature: Softmax temperature (lower = more winner-take-all)
        min_weight: Minimum weight for any miner (0 = pure softmax)

    Returns:
        Dict mapping uid -> normalized weight (sums to 1)
    """
    if not scores:
        return {}

    uids = list(scores.keys())
    values = np.array([scores[uid] for uid in uids])

    if np.all(values == 0):
        # All zeros - uniform distribution
        n = len(uids)
        return {uid: 1.0 / n for uid in uids}

    # Softmax with temperature
    # Shift values for numerical stability
    values_shifted = values - np.max(values)
    exp_values = np.exp(values_shifted / temperature)
    weights = exp_values / exp_values.sum()

    # Apply minimum weight if specified
    if min_weight > 0:
        n = len(uids)
        min_total = min_weight * n
        if min_total < 1.0:
            # Scale down current weights and add minimum
            scale = 1.0 - min_total
            weights = weights * scale + min_weight
        # else: min_weight too high, just normalize

    return {uid: float(w) for uid, w in zip(uids, weights)}


def compute_full_scoring(
    miner_scores: dict[int, dict[str, float]],
    env_ids: list[str],
    epsilons: dict[str, float] | None = None,
    temperature: float = 1.0,
) -> dict[int, float]:
    """
    Complete scoring pipeline: subset scores -> weights.

    Convenience function that combines compute_subset_scores
    and scores_to_weights.

    Args:
        miner_scores: Dict mapping uid -> env_id -> score
        env_ids: List of environment IDs
        epsilons: Epsilon per environment (default: 0.05 for all)
        temperature: Softmax temperature

    Returns:
        Dict mapping uid -> weight (normalized, sums to 1)
    """
    if epsilons is None:
        epsilons = {env: 0.05 for env in env_ids}

    subset_scores = compute_subset_scores(miner_scores, env_ids, epsilons)
    return scores_to_weights(subset_scores, temperature=temperature)


def find_subset_winner_with_priority(
    miner_scores: dict[int, dict[str, float]],
    miner_thresholds: dict[int, dict[str, float]],
    miner_first_blocks: dict[int, int],
    subset: tuple[str, ...],
) -> int | None:
    """
    Find the miner that dominates all others on a subset, with first-commit advantage.

    The earlier miner (lower first_block) wins ties. A later miner must beat the
    earlier miner's threshold on ALL environments in the subset to win.

    Args:
        miner_scores: Dict mapping uid -> env_id -> score
        miner_thresholds: Dict mapping uid -> env_id -> threshold (score + gap)
        miner_first_blocks: Dict mapping uid -> first committed block
        subset: Tuple of environment IDs to check

    Returns:
        UID of the winner, or None if no clear winner
    """
    uids = list(miner_scores.keys())

    if len(uids) < 2:
        return uids[0] if uids else None

    # Sort by first_block (earlier first), then by uid for determinism
    sorted_uids = sorted(uids, key=lambda u: (miner_first_blocks.get(u, float("inf")), u))

    for candidate in sorted_uids:
        dominates_all = True
        candidate_block = miner_first_blocks.get(candidate, float("inf"))

        for other in sorted_uids:
            if candidate == other:
                continue

            other_block = miner_first_blocks.get(other, float("inf"))

            # Determine who came first
            if candidate_block <= other_block:
                earlier_uid = candidate
                later_uid = other
            else:
                earlier_uid = other
                later_uid = candidate

            # Count wins for the later miner (must beat threshold to win)
            later_wins = 0
            for env in subset:
                earlier_score = miner_scores[earlier_uid].get(env, 0.0)
                later_score = miner_scores[later_uid].get(env, 0.0)
                threshold = miner_thresholds[earlier_uid].get(env, earlier_score + 0.02)

                if later_score > threshold:
                    later_wins += 1

            # Determine who dominates this pair on the subset
            n_envs = len(subset)
            if candidate_block <= other_block:
                # Candidate is earlier: candidate dominates if other doesn't win all
                if later_wins == n_envs:
                    # Other beat all thresholds - candidate doesn't dominate
                    dominates_all = False
                    break
            else:
                # Other is earlier: candidate dominates only if candidate wins all
                if later_wins < n_envs:
                    # Candidate didn't beat all thresholds - candidate doesn't dominate
                    dominates_all = False
                    break

        if dominates_all:
            return candidate

    return None  # No clear winner


def compute_subset_scores_with_priority(
    miner_scores: dict[int, dict[str, float]],
    miner_thresholds: dict[int, dict[str, float]],
    miner_first_blocks: dict[int, int],
    env_ids: list[str],
    subset_weight_scheme: str = "linear",
) -> dict[int, float]:
    """
    Compute winners-take-all scores with first-commit advantage.

    For each non-empty subset S of environments:
    1. Find the miner that dominates on S (with time priority)
    2. Award them a score K_|S| based on subset size

    Args:
        miner_scores: Dict mapping uid -> env_id -> score
        miner_thresholds: Dict mapping uid -> env_id -> threshold
        miner_first_blocks: Dict mapping uid -> first committed block
        env_ids: List of environment IDs
        subset_weight_scheme: How to weight subsets ("linear", "exponential", "equal")

    Returns:
        Dict mapping uid -> total score
    """
    uids = list(miner_scores.keys())
    final_scores = {uid: 0.0 for uid in uids}

    if not uids or not env_ids:
        return final_scores

    # Iterate over all non-empty subsets
    for subset_size in range(1, len(env_ids) + 1):
        # Compute weight for this subset size
        if subset_weight_scheme == "linear":
            subset_weight = float(subset_size)
        elif subset_weight_scheme == "exponential":
            subset_weight = float(2 ** (subset_size - 1))
        elif subset_weight_scheme == "equal":
            subset_weight = 1.0
        else:
            subset_weight = float(subset_size)

        # Check each subset of this size
        for subset in combinations(env_ids, subset_size):
            winner = find_subset_winner_with_priority(
                miner_scores,
                miner_thresholds,
                miner_first_blocks,
                subset,
            )
            if winner is not None:
                final_scores[winner] += subset_weight

    return final_scores
