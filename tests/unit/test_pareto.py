"""Tests for Pareto scoring."""

import numpy as np
import pytest

from robo.scoring.pareto import (
    compute_epsilon,
    compute_pareto_frontier,
    epsilon_dominates,
)
from robo.scoring.winners_take_all import (
    compute_full_scoring,
    compute_subset_scores,
    find_subset_winner,
    scores_to_weights,
)


class TestEpsilonDominates:
    """Tests for epsilon dominance."""

    def test_clear_dominance(self):
        """A clearly dominates B."""
        a = np.array([0.9, 0.8, 0.85])
        b = np.array([0.7, 0.6, 0.65])
        eps = np.array([0.05, 0.05, 0.05])

        assert epsilon_dominates(a, b, eps) is True
        assert epsilon_dominates(b, a, eps) is False

    def test_tie_no_dominance(self):
        """Equal scores mean no dominance (anti-copy mechanism)."""
        a = np.array([0.8, 0.7, 0.75])
        b = np.array([0.8, 0.7, 0.75])
        eps = np.array([0.05, 0.05, 0.05])

        assert epsilon_dominates(a, b, eps) is False
        assert epsilon_dominates(b, a, eps) is False

    def test_pareto_incomparable(self):
        """Neither dominates when each is better on different dims."""
        a = np.array([0.9, 0.5])  # Better on env 0
        b = np.array([0.5, 0.9])  # Better on env 1
        eps = np.array([0.05, 0.05])

        assert epsilon_dominates(a, b, eps) is False
        assert epsilon_dominates(b, a, eps) is False

    def test_within_epsilon_no_dominance(self):
        """Small differences within epsilon don't count."""
        a = np.array([0.82, 0.72])
        b = np.array([0.80, 0.70])
        eps = np.array([0.05, 0.05])  # Differences are < 2*eps

        assert epsilon_dominates(a, b, eps) is False


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_single_miner(self):
        """Single miner is always on frontier."""
        scores = {0: {"env_a": 0.8, "env_b": 0.7}}
        result = compute_pareto_frontier(scores, ["env_a", "env_b"], 50)

        assert result.frontier_uids == [0]

    def test_dominant_miner(self):
        """Clearly dominant miner is sole frontier member."""
        scores = {
            0: {"env_a": 0.9, "env_b": 0.9},
            1: {"env_a": 0.5, "env_b": 0.5},
            2: {"env_a": 0.6, "env_b": 0.6},
        }
        result = compute_pareto_frontier(scores, ["env_a", "env_b"], 50)

        assert 0 in result.frontier_uids
        assert 1 not in result.frontier_uids
        assert 2 not in result.frontier_uids

    def test_pareto_incomparable_both_on_frontier(self):
        """Pareto-incomparable miners both on frontier."""
        scores = {
            0: {"env_a": 0.9, "env_b": 0.5},  # Specialist in A
            1: {"env_a": 0.5, "env_b": 0.9},  # Specialist in B
        }
        result = compute_pareto_frontier(scores, ["env_a", "env_b"], 50)

        assert set(result.frontier_uids) == {0, 1}

    def test_copy_attack_both_on_frontier(self):
        """Copying another miner results in tie (both on frontier)."""
        scores = {
            0: {"env_a": 0.8, "env_b": 0.8},
            1: {"env_a": 0.8, "env_b": 0.8},  # Copy of miner 0
        }
        result = compute_pareto_frontier(scores, ["env_a", "env_b"], 50)

        # Both are on frontier (tie)
        assert set(result.frontier_uids) == {0, 1}


class TestWinnersTakeAll:
    """Tests for winners-take-all scoring."""

    def test_clear_winner_gets_all_points(self):
        """Clear winner gets points from all subsets."""
        scores = {
            0: {"a": 0.9, "b": 0.9},
            1: {"a": 0.5, "b": 0.5},
        }
        epsilons = {"a": 0.05, "b": 0.05}

        subset_scores = compute_subset_scores(scores, ["a", "b"], epsilons)

        assert subset_scores[0] > subset_scores[1]

    def test_specialists_split_points(self):
        """Specialists win their respective subsets."""
        scores = {
            0: {"a": 0.9, "b": 0.3},  # Specialist in A
            1: {"a": 0.3, "b": 0.9},  # Specialist in B
        }
        epsilons = {"a": 0.05, "b": 0.05}

        subset_scores = compute_subset_scores(scores, ["a", "b"], epsilons)

        # Both should have some points (from single-env subsets)
        assert subset_scores[0] > 0
        assert subset_scores[1] > 0

    def test_scores_to_weights_normalized(self):
        """Weights should sum to 1."""
        scores = {0: 10.0, 1: 5.0, 2: 3.0}
        weights = scores_to_weights(scores)

        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_full_scoring_sybil_resistance(self):
        """Sybil attack (copies) doesn't increase total reward."""
        env_ids = ["a", "b"]

        # Single honest miner
        single = {0: {"a": 0.8, "b": 0.8}}
        single_weights = compute_full_scoring(single, env_ids)

        # Same policy across 5 sybil identities
        sybil = {i: {"a": 0.8, "b": 0.8} for i in range(5)}
        sybil_weights = compute_full_scoring(sybil, env_ids)

        # Total reward is same (just split across identities)
        single_total = sum(single_weights.values())
        sybil_total = sum(sybil_weights.values())

        assert abs(single_total - sybil_total) < 0.01


class TestComputeEpsilon:
    """Tests for epsilon computation."""

    def test_high_variance_high_epsilon(self):
        """High variance should give higher epsilon."""
        low_var = np.array([0.5, 0.5, 0.5, 0.5])
        high_var = np.array([0.1, 0.9, 0.2, 0.8])

        eps_low = compute_epsilon(low_var, 50)
        eps_high = compute_epsilon(high_var, 50)

        assert eps_high > eps_low

    def test_more_samples_lower_epsilon(self):
        """More samples should give lower epsilon."""
        values = np.array([0.6, 0.7, 0.65, 0.75])

        eps_few = compute_epsilon(values, 10)
        eps_many = compute_epsilon(values, 100)

        assert eps_many < eps_few
