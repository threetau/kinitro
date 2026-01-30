"""Tests for Pareto scoring."""

import numpy as np

from kinitro.scoring.pareto import (
    compute_epsilon,
    compute_pareto_frontier,
    epsilon_dominates,
)
from kinitro.scoring.threshold import calculate_threshold, compute_miner_thresholds
from kinitro.scoring.winners_take_all import (
    compute_full_scoring,
    compute_subset_scores,
    compute_subset_scores_with_priority,
    find_subset_winner_with_priority,
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


class TestThreshold:
    """Tests for threshold calculation."""

    def test_threshold_basic(self):
        """Threshold should be score + gap."""
        threshold = calculate_threshold(0.5, 100)
        assert threshold > 0.5
        assert threshold <= 0.6  # Should be <= score + max_gap

    def test_threshold_more_samples_smaller_gap(self):
        """More samples should result in smaller gap."""
        t_few = calculate_threshold(0.5, 50)
        t_many = calculate_threshold(0.5, 500)

        # More samples = smaller gap = lower threshold
        assert t_many < t_few

    def test_threshold_min_gap_enforced(self):
        """Minimum gap should be enforced."""
        # With many samples, gap would be tiny, but min_gap enforces 2%
        threshold = calculate_threshold(0.5, 10000, min_gap=0.02)
        assert threshold >= 0.52  # At least score + min_gap

    def test_threshold_max_gap_enforced(self):
        """Maximum gap should be enforced."""
        # With few samples, gap would be huge, but max_gap caps at 10%
        threshold = calculate_threshold(0.5, 5, max_gap=0.10)
        assert threshold <= 0.60  # At most score + max_gap

    def test_compute_miner_thresholds(self):
        """Should compute thresholds for all miners and environments."""
        scores = {
            0: {"a": 0.8, "b": 0.7},
            1: {"a": 0.6, "b": 0.9},
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)

        assert 0 in thresholds
        assert 1 in thresholds
        assert "a" in thresholds[0]
        assert "b" in thresholds[0]
        # Threshold should be higher than score
        assert thresholds[0]["a"] > 0.8
        assert thresholds[1]["b"] > 0.9


class TestFirstCommitAdvantage:
    """Tests for first-commit advantage scoring."""

    def test_earlier_miner_wins_tie(self):
        """Earlier miner should win when scores are identical."""
        scores = {
            0: {"a": 0.8, "b": 0.8},
            1: {"a": 0.8, "b": 0.8},  # Identical to miner 0
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {0: 1000, 1: 2000}  # Miner 0 came first

        winner = find_subset_winner_with_priority(scores, thresholds, first_blocks, ("a", "b"))

        # Miner 0 should win because they came first
        assert winner == 0

    def test_later_miner_wins_if_beats_threshold(self):
        """Later miner should win if they beat the threshold on all envs."""
        scores = {
            0: {"a": 0.7, "b": 0.7},
            1: {"a": 0.9, "b": 0.9},  # Much better than miner 0
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {0: 1000, 1: 2000}  # Miner 0 came first

        winner = find_subset_winner_with_priority(scores, thresholds, first_blocks, ("a", "b"))

        # Miner 1 should win because they beat the threshold
        assert winner == 1

    def test_tradeoff_earlier_wins_by_default(self):
        """Earlier miner wins tradeoff because later can't beat threshold on all envs."""
        scores = {
            0: {"a": 0.9, "b": 0.5},  # Specialist in A, came first
            1: {"a": 0.5, "b": 0.9},  # Specialist in B, came later
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {0: 1000, 1: 2000}

        winner = find_subset_winner_with_priority(scores, thresholds, first_blocks, ("a", "b"))

        # Miner 0 wins because miner 1 can't beat threshold on env "a"
        # (1 has 0.5, threshold for 0's 0.9 is ~0.92-1.0)
        assert winner == 0

    def test_tradeoff_neither_wins_same_block(self):
        """Neither wins tradeoff when both registered at same block."""
        scores = {
            0: {"a": 0.9, "b": 0.5},  # Specialist in A
            1: {"a": 0.5, "b": 0.9},  # Specialist in B
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        # Same block - neither has time priority, but 0 wins by UID tiebreaker
        first_blocks = {0: 1000, 1: 1000}

        winner = find_subset_winner_with_priority(scores, thresholds, first_blocks, ("a", "b"))

        # Miner 0 wins by UID tiebreaker (lower UID when same block)
        assert winner == 0

    def test_copy_attack_fails(self):
        """Copying the leader should not help the copier."""
        scores = {
            0: {"a": 0.8, "b": 0.8},  # Leader
            1: {"a": 0.8, "b": 0.8},  # Copier
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {0: 1000, 1: 2000}

        subset_scores = compute_subset_scores_with_priority(
            scores, thresholds, first_blocks, ["a", "b"]
        )

        # Leader should get all points, copier gets nothing
        assert subset_scores[0] > 0
        assert subset_scores[1] == 0

    def test_genuine_improvement_rewarded(self):
        """Genuine improvement over the leader should be rewarded."""
        scores = {
            0: {"a": 0.7, "b": 0.7},  # Leader
            1: {"a": 0.85, "b": 0.85},  # Genuinely better
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {0: 1000, 1: 2000}

        subset_scores = compute_subset_scores_with_priority(
            scores, thresholds, first_blocks, ["a", "b"]
        )

        # The genuinely better miner should get more points
        assert subset_scores[1] > subset_scores[0]
