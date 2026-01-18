"""
Scoring module for ε-Pareto dominance and weight computation.

Implements the multi-environment scoring mechanism where only
policies on the Pareto frontier earn rewards.
"""

from robo.scoring.pareto import (
    ParetoResult,
    compute_epsilon,
    compute_pareto_frontier,
    epsilon_dominates,
)
from robo.scoring.winners_take_all import (
    compute_subset_scores,
    find_subset_winner,
    scores_to_weights,
)

__all__ = [
    "ParetoResult",
    "compute_epsilon",
    "epsilon_dominates",
    "compute_pareto_frontier",
    "compute_subset_scores",
    "find_subset_winner",
    "scores_to_weights",
]
