"""ε-Pareto dominance computation for multi-environment scoring."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ParetoResult:
    """Result of Pareto frontier computation."""

    # UIDs of miners on the Pareto frontier
    frontier_uids: list[int]

    # Dominance matrix: dominance[i, j] = True if miner i dominates miner j
    dominance_matrix: np.ndarray

    # Epsilon values used for each environment
    epsilons: np.ndarray

    # Score matrix used (for reference)
    score_matrix: np.ndarray

    # Mapping from matrix index to UID
    uid_mapping: list[int] = field(default_factory=list)


def compute_epsilon(
    values: np.ndarray,
    n_samples: int,
    min_epsilon: float = 0.01,
    max_epsilon: float = 0.2,
) -> float:
    """
    Compute epsilon tolerance based on standard error.

    Epsilon provides statistical robustness against sampling noise.
    We use 2 standard errors as the tolerance.

    Args:
        values: Array of scores for one environment
        n_samples: Number of samples per score
        min_epsilon: Minimum epsilon value
        max_epsilon: Maximum epsilon value

    Returns:
        Epsilon value for dominance comparison
    """
    if n_samples <= 1:
        return max_epsilon

    # Standard deviation of the scores
    std = np.std(values)

    # Standard error of the mean
    se = std / np.sqrt(n_samples)

    # Use 2 standard errors for 95% confidence
    epsilon = 2 * se

    # Clamp to reasonable range
    return float(np.clip(epsilon, min_epsilon, max_epsilon))


def epsilon_dominates(
    a_scores: np.ndarray,
    b_scores: np.ndarray,
    epsilons: np.ndarray,
) -> bool:
    """
    Check if miner A ε-dominates miner B.

    A ε-dominates B if:
    1. A is not worse than B on ANY environment (within ε tolerance)
    2. A is strictly better than B on at least ONE environment

    This is the core mechanism that makes copying useless:
    - If you copy, you tie (no dominance either way)
    - You must IMPROVE to dominate and earn emissions

    Args:
        a_scores: Scores for miner A across environments
        b_scores: Scores for miner B across environments
        epsilons: Epsilon tolerance for each environment

    Returns:
        True if A ε-dominates B
    """
    # A must be >= B - ε on all environments (not worse within tolerance)
    if not np.all(a_scores >= b_scores - epsilons):
        return False

    # A must be > B + ε on at least one environment (strictly better)
    if not np.any(a_scores > b_scores + epsilons):
        return False

    return True


def compute_pareto_frontier(
    miner_scores: dict[int, dict[str, float]],
    env_ids: list[str],
    n_samples_per_env: int | dict[str, int],
) -> ParetoResult:
    """
    Compute ε-Pareto frontier across all miners.

    The Pareto frontier is the set of non-dominated miners.
    A miner is on the frontier if no other miner dominates them.

    This naturally handles:
    - Sybil attacks: Copies tie, don't dominate each other
    - Specialization: Must be good on ALL envs, not just one
    - Gaming: Can't game without actually improving

    Args:
        miner_scores: Dict mapping uid -> env_id -> success_rate
        env_ids: List of environment IDs (defines ordering)
        n_samples_per_env: Sample count per environment (for epsilon)

    Returns:
        ParetoResult with frontier UIDs and dominance matrix
    """
    uids = list(miner_scores.keys())
    n_miners = len(uids)
    n_envs = len(env_ids)

    if n_miners == 0:
        return ParetoResult(
            frontier_uids=[],
            dominance_matrix=np.zeros((0, 0), dtype=bool),
            epsilons=np.zeros(0),
            score_matrix=np.zeros((0, 0)),
            uid_mapping=[],
        )

    # Build score matrix: (n_miners, n_environments)
    score_matrix = np.zeros((n_miners, n_envs), dtype=np.float32)
    for i, uid in enumerate(uids):
        for j, env_id in enumerate(env_ids):
            score_matrix[i, j] = miner_scores.get(uid, {}).get(env_id, 0.0)

    # Compute epsilon for each environment
    if isinstance(n_samples_per_env, int):
        samples = {env_id: n_samples_per_env for env_id in env_ids}
    else:
        samples = n_samples_per_env

    epsilons = np.array(
        [compute_epsilon(score_matrix[:, j], samples.get(env_ids[j], 50)) for j in range(n_envs)]
    )

    # Compute dominance matrix
    dominance = np.zeros((n_miners, n_miners), dtype=bool)
    for i in range(n_miners):
        for j in range(n_miners):
            if i != j:
                dominance[i, j] = epsilon_dominates(score_matrix[i], score_matrix[j], epsilons)

    # Frontier = miners not dominated by anyone
    is_dominated = np.any(dominance, axis=0)
    frontier_indices = np.where(~is_dominated)[0]
    frontier_uids = [uids[i] for i in frontier_indices]

    return ParetoResult(
        frontier_uids=frontier_uids,
        dominance_matrix=dominance,
        epsilons=epsilons,
        score_matrix=score_matrix,
        uid_mapping=uids,
    )


def get_dominating_miners(pareto_result: ParetoResult, uid: int) -> list[int]:
    """
    Get list of miners that dominate the given miner.

    Args:
        pareto_result: Result from compute_pareto_frontier
        uid: UID of miner to check

    Returns:
        List of UIDs that dominate this miner
    """
    if uid not in pareto_result.uid_mapping:
        return []

    idx = pareto_result.uid_mapping.index(uid)
    dominating_indices = np.where(pareto_result.dominance_matrix[:, idx])[0]
    return [pareto_result.uid_mapping[i] for i in dominating_indices]


def get_dominated_miners(pareto_result: ParetoResult, uid: int) -> list[int]:
    """
    Get list of miners that are dominated by the given miner.

    Args:
        pareto_result: Result from compute_pareto_frontier
        uid: UID of miner to check

    Returns:
        List of UIDs that this miner dominates
    """
    if uid not in pareto_result.uid_mapping:
        return []

    idx = pareto_result.uid_mapping.index(uid)
    dominated_indices = np.where(pareto_result.dominance_matrix[idx, :])[0]
    return [pareto_result.uid_mapping[i] for i in dominated_indices]


def later_beats_earlier(
    later_scores: dict[str, float],
    earlier_thresholds: dict[str, float],
    env_ids: list[str],
) -> bool:
    """
    Check if a later miner beats an earlier miner's thresholds on all environments.

    Args:
        later_scores: The later miner's scores
        earlier_thresholds: The earlier miner's thresholds (score + gap)
        env_ids: Environments to check

    Returns:
        True if later miner beats threshold on ALL environments
    """
    return all(later_scores.get(env, 0.0) > earlier_thresholds.get(env, 1.0) for env in env_ids)
