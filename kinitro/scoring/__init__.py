"""
Scoring module for ε-Pareto dominance and weight computation.

Implements the multi-environment scoring mechanism where only
policies on the Pareto frontier earn rewards.
"""

from kinitro.scoring.pareto import (
    ParetoResult,
    compute_epsilon,
    compute_pareto_frontier,
    epsilon_dominates,
    later_beats_earlier,
)
from kinitro.scoring.threshold import (
    calculate_threshold,
    compute_miner_thresholds,
)
from kinitro.scoring.winners_take_all import (
    compute_subset_scores,
    compute_subset_scores_with_priority,
    find_subset_winner,
    find_subset_winner_with_priority,
    scores_to_weights,
)

__all__ = [
    "ParetoResult",
    "calculate_threshold",
    "compute_epsilon",
    "compute_miner_thresholds",
    "compute_pareto_frontier",
    "compute_subset_scores",
    "compute_subset_scores_with_priority",
    "epsilon_dominates",
    "find_subset_winner",
    "find_subset_winner_with_priority",
    "later_beats_earlier",
    "scores_to_weights",
]
