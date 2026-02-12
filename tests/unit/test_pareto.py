"""Tests for Pareto scoring."""

import numpy as np

from kinitro.scoring.pareto import (
    compute_epsilon,
    compute_pareto_frontier,
    epsilon_dominates,
)
from kinitro.scoring.threshold import calculate_threshold, compute_miner_thresholds
from kinitro.scoring.winners_take_all import (
    compute_subset_scores_with_priority,
    find_subset_winner_with_priority,
    scores_to_weights,
)
from kinitro.types import BlockNumber, EnvironmentId, MinerUID

ENV_A = EnvironmentId("a")
ENV_B = EnvironmentId("b")
ENV_AB = [ENV_A, ENV_B]
ENV_AB_TUPLE = (ENV_A, ENV_B)


class TestEpsilonDominates:
    """Tests for epsilon dominance."""

    def test_clear_dominance(self) -> None:
        """A clearly dominates B."""
        a = np.array([0.9, 0.8, 0.85])
        b = np.array([0.7, 0.6, 0.65])
        eps = np.array([0.05, 0.05, 0.05])

        assert epsilon_dominates(a, b, eps) is True
        assert epsilon_dominates(b, a, eps) is False

    def test_tie_no_dominance(self) -> None:
        """Equal scores mean no dominance (anti-copy mechanism)."""
        a = np.array([0.8, 0.7, 0.75])
        b = np.array([0.8, 0.7, 0.75])
        eps = np.array([0.05, 0.05, 0.05])

        assert epsilon_dominates(a, b, eps) is False
        assert epsilon_dominates(b, a, eps) is False

    def test_pareto_incomparable(self) -> None:
        """Neither dominates when each is better on different dims."""
        a = np.array([0.9, 0.5])  # Better on env 0
        b = np.array([0.5, 0.9])  # Better on env 1
        eps = np.array([0.05, 0.05])

        assert epsilon_dominates(a, b, eps) is False
        assert epsilon_dominates(b, a, eps) is False

    def test_within_epsilon_no_dominance(self) -> None:
        """Small differences within epsilon don't count."""
        a = np.array([0.82, 0.72])
        b = np.array([0.80, 0.70])
        eps = np.array([0.05, 0.05])  # Differences are < 2*eps

        assert epsilon_dominates(a, b, eps) is False


class TestParetoFrontier:
    """Tests for Pareto frontier computation."""

    def test_single_miner(self) -> None:
        """Single miner is always on frontier."""
        scores = {MinerUID(0): {EnvironmentId("env_a"): 0.8, EnvironmentId("env_b"): 0.7}}
        result = compute_pareto_frontier(
            scores, [EnvironmentId("env_a"), EnvironmentId("env_b")], 50
        )

        assert result.frontier_uids == [MinerUID(0)]

    def test_dominant_miner(self) -> None:
        """Clearly dominant miner is sole frontier member."""
        scores = {
            MinerUID(0): {EnvironmentId("env_a"): 0.9, EnvironmentId("env_b"): 0.9},
            MinerUID(1): {EnvironmentId("env_a"): 0.5, EnvironmentId("env_b"): 0.5},
            MinerUID(2): {EnvironmentId("env_a"): 0.6, EnvironmentId("env_b"): 0.6},
        }
        result = compute_pareto_frontier(
            scores, [EnvironmentId("env_a"), EnvironmentId("env_b")], 50
        )

        assert MinerUID(0) in result.frontier_uids
        assert MinerUID(1) not in result.frontier_uids
        assert MinerUID(2) not in result.frontier_uids

    def test_pareto_incomparable_both_on_frontier(self) -> None:
        """Pareto-incomparable miners both on frontier."""
        scores = {
            MinerUID(0): {
                EnvironmentId("env_a"): 0.9,
                EnvironmentId("env_b"): 0.5,
            },  # Specialist in A
            MinerUID(1): {
                EnvironmentId("env_a"): 0.5,
                EnvironmentId("env_b"): 0.9,
            },  # Specialist in B
        }
        result = compute_pareto_frontier(
            scores, [EnvironmentId("env_a"), EnvironmentId("env_b")], 50
        )

        assert set(result.frontier_uids) == {MinerUID(0), MinerUID(1)}

    def test_copy_attack_both_on_frontier(self) -> None:
        """Copying another miner results in tie (both on frontier)."""
        scores = {
            MinerUID(0): {EnvironmentId("env_a"): 0.8, EnvironmentId("env_b"): 0.8},
            MinerUID(1): {
                EnvironmentId("env_a"): 0.8,
                EnvironmentId("env_b"): 0.8,
            },  # Copy of miner 0
        }
        result = compute_pareto_frontier(
            scores, [EnvironmentId("env_a"), EnvironmentId("env_b")], 50
        )

        # Both are on frontier (tie)
        assert set(result.frontier_uids) == {MinerUID(0), MinerUID(1)}


class TestWinnersTakeAll:
    """Tests for winners-take-all scoring."""

    def test_clear_winner_gets_all_points(self) -> None:
        """Clear winner gets points from all subsets."""
        scores = {
            MinerUID(0): {ENV_A: 0.9, ENV_B: 0.9},
            MinerUID(1): {ENV_A: 0.5, ENV_B: 0.5},
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {
            MinerUID(0): BlockNumber(1000),
            MinerUID(1): BlockNumber(1000),
        }  # Same block

        subset_scores = compute_subset_scores_with_priority(
            scores, thresholds, first_blocks, ENV_AB
        )

        assert subset_scores[MinerUID(0)] > subset_scores[MinerUID(1)]

    def test_specialists_split_points(self) -> None:
        """Specialists win their respective single-env subsets."""
        scores = {
            MinerUID(0): {ENV_A: 0.9, ENV_B: 0.3},  # Specialist in A
            MinerUID(1): {ENV_A: 0.3, ENV_B: 0.9},  # Specialist in B
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {
            MinerUID(0): BlockNumber(1000),
            MinerUID(1): BlockNumber(1000),
        }  # Same block

        subset_scores = compute_subset_scores_with_priority(
            scores, thresholds, first_blocks, ENV_AB
        )

        # Both should have some points (from single-env subsets)
        assert subset_scores[MinerUID(0)] > 0
        assert subset_scores[MinerUID(1)] > 0

    def test_scores_to_weights_normalized(self) -> None:
        """Weights should sum to 1."""
        scores = {MinerUID(0): 10.0, MinerUID(1): 5.0, MinerUID(2): 3.0}
        weights = scores_to_weights(scores)

        assert abs(sum(weights.values()) - 1.0) < 1e-6

    def test_sybil_resistance(self) -> None:
        """Sybil attack (copies) doesn't increase total reward."""
        env_ids = ENV_AB

        # Single honest miner
        single_scores = {MinerUID(0): {ENV_A: 0.8, ENV_B: 0.8}}
        single_thresholds = compute_miner_thresholds(single_scores, episodes_per_env=50)
        single_blocks = {MinerUID(0): BlockNumber(1000)}
        single_subset = compute_subset_scores_with_priority(
            single_scores, single_thresholds, single_blocks, env_ids
        )
        single_weights = scores_to_weights(single_subset)

        # Same policy across 5 sybil identities (all same block)
        sybil_scores = {MinerUID(i): {ENV_A: 0.8, ENV_B: 0.8} for i in range(5)}
        sybil_thresholds = compute_miner_thresholds(sybil_scores, episodes_per_env=50)
        sybil_blocks = {MinerUID(i): BlockNumber(1000) for i in range(5)}  # All same block
        sybil_subset = compute_subset_scores_with_priority(
            sybil_scores, sybil_thresholds, sybil_blocks, env_ids
        )
        sybil_weights = scores_to_weights(sybil_subset)

        # Total reward is same (just split across identities)
        single_total = sum(single_weights.values())
        sybil_total = sum(sybil_weights.values())

        assert abs(single_total - sybil_total) < 0.01


class TestComputeEpsilon:
    """Tests for epsilon computation."""

    def test_high_variance_high_epsilon(self) -> None:
        """High variance should give higher epsilon."""
        low_var = np.array([0.5, 0.5, 0.5, 0.5])
        high_var = np.array([0.1, 0.9, 0.2, 0.8])

        eps_low = compute_epsilon(low_var, 50)
        eps_high = compute_epsilon(high_var, 50)

        assert eps_high > eps_low

    def test_more_samples_lower_epsilon(self) -> None:
        """More samples should give lower epsilon."""
        values = np.array([0.6, 0.7, 0.65, 0.75])

        eps_few = compute_epsilon(values, 10)
        eps_many = compute_epsilon(values, 100)

        assert eps_many < eps_few


class TestThreshold:
    """Tests for threshold calculation."""

    def test_threshold_basic(self) -> None:
        """Threshold should be score + gap."""
        threshold = calculate_threshold(0.5, 100)
        assert threshold > 0.5
        assert threshold <= 0.6  # Should be <= score + max_gap

    def test_threshold_more_samples_smaller_gap(self) -> None:
        """More samples should result in smaller gap."""
        t_few = calculate_threshold(0.5, 50)
        t_many = calculate_threshold(0.5, 500)

        # More samples = smaller gap = lower threshold
        assert t_many < t_few

    def test_threshold_min_gap_enforced(self) -> None:
        """Minimum gap should be enforced."""
        # With many samples, gap would be tiny, but min_gap enforces 2%
        threshold = calculate_threshold(0.5, 10000, min_gap=0.02)
        assert threshold >= 0.52  # At least score + min_gap

    def test_threshold_max_gap_enforced(self) -> None:
        """Maximum gap should be enforced."""
        # With few samples, gap would be huge, but max_gap caps at 10%
        threshold = calculate_threshold(0.5, 5, max_gap=0.10)
        assert threshold <= 0.60  # At most score + max_gap

    def test_compute_miner_thresholds(self) -> None:
        """Should compute thresholds for all miners and environments."""
        scores = {
            MinerUID(0): {ENV_A: 0.8, ENV_B: 0.7},
            MinerUID(1): {ENV_A: 0.6, ENV_B: 0.9},
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)

        assert MinerUID(0) in thresholds
        assert MinerUID(1) in thresholds
        assert ENV_A in thresholds[MinerUID(0)]
        assert ENV_B in thresholds[MinerUID(0)]
        # Threshold should be higher than score
        assert thresholds[MinerUID(0)][ENV_A] > 0.8
        assert thresholds[MinerUID(1)][ENV_B] > 0.9


class TestFirstCommitAdvantage:
    """Tests for first-commit advantage scoring."""

    def test_earlier_miner_wins_tie(self) -> None:
        """Earlier miner should win when scores are identical."""
        scores = {
            MinerUID(0): {ENV_A: 0.8, ENV_B: 0.8},
            MinerUID(1): {ENV_A: 0.8, ENV_B: 0.8},  # Identical to miner 0
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {
            MinerUID(0): BlockNumber(1000),
            MinerUID(1): BlockNumber(2000),
        }  # Miner 0 came first

        winner = find_subset_winner_with_priority(scores, thresholds, first_blocks, ENV_AB_TUPLE)

        # Miner 0 should win because they came first
        assert winner == MinerUID(0)

    def test_later_miner_wins_if_beats_threshold(self) -> None:
        """Later miner should win if they beat the threshold on all envs."""
        scores = {
            MinerUID(0): {ENV_A: 0.7, ENV_B: 0.7},
            MinerUID(1): {
                ENV_A: 0.9,
                ENV_B: 0.9,
            },  # Much better than miner 0
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {
            MinerUID(0): BlockNumber(1000),
            MinerUID(1): BlockNumber(2000),
        }  # Miner 0 came first

        winner = find_subset_winner_with_priority(scores, thresholds, first_blocks, ENV_AB_TUPLE)

        # Miner 1 should win because they beat the threshold
        assert winner == MinerUID(1)

    def test_tradeoff_earlier_wins_by_default(self) -> None:
        """Earlier miner wins tradeoff because later can't beat threshold on all envs."""
        scores = {
            MinerUID(0): {
                ENV_A: 0.9,
                ENV_B: 0.5,
            },  # Specialist in A, came first
            MinerUID(1): {
                ENV_A: 0.5,
                ENV_B: 0.9,
            },  # Specialist in B, came later
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {MinerUID(0): BlockNumber(1000), MinerUID(1): BlockNumber(2000)}

        winner = find_subset_winner_with_priority(scores, thresholds, first_blocks, ENV_AB_TUPLE)

        # Miner 0 wins because miner 1 can't beat threshold on env "a"
        # (1 has 0.5, threshold for 0's 0.9 is ~0.92-1.0)
        assert winner == MinerUID(0)

    def test_tradeoff_lower_uid_wins_same_block(self) -> None:
        """Lower UID wins when both registered at the same block."""
        scores = {
            MinerUID(0): {ENV_A: 0.9, ENV_B: 0.5},  # Specialist in A
            MinerUID(1): {ENV_A: 0.5, ENV_B: 0.9},  # Specialist in B
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        # Same block - neither has time priority, so lower UID wins
        first_blocks = {MinerUID(0): BlockNumber(1000), MinerUID(1): BlockNumber(1000)}

        winner = find_subset_winner_with_priority(scores, thresholds, first_blocks, ENV_AB_TUPLE)

        # Miner 0 wins by UID tiebreaker (lower UID when same block)
        assert winner == MinerUID(0)

    def test_copy_attack_fails(self) -> None:
        """Copying the leader should not help the copier."""
        scores = {
            MinerUID(0): {ENV_A: 0.8, ENV_B: 0.8},  # Leader
            MinerUID(1): {ENV_A: 0.8, ENV_B: 0.8},  # Copier
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {MinerUID(0): BlockNumber(1000), MinerUID(1): BlockNumber(2000)}

        subset_scores = compute_subset_scores_with_priority(
            scores, thresholds, first_blocks, ENV_AB
        )

        # Leader should get all points, copier gets nothing
        assert subset_scores[MinerUID(0)] > 0
        assert subset_scores[MinerUID(1)] == 0

    def test_genuine_improvement_rewarded(self) -> None:
        """Genuine improvement over the leader should be rewarded."""
        scores = {
            MinerUID(0): {ENV_A: 0.7, ENV_B: 0.7},  # Leader
            MinerUID(1): {ENV_A: 0.85, ENV_B: 0.85},  # Genuinely better
        }
        thresholds = compute_miner_thresholds(scores, episodes_per_env=50)
        first_blocks = {MinerUID(0): BlockNumber(1000), MinerUID(1): BlockNumber(2000)}

        subset_scores = compute_subset_scores_with_priority(
            scores, thresholds, first_blocks, ENV_AB
        )

        # The genuinely better miner should get more points
        assert subset_scores[MinerUID(1)] > subset_scores[MinerUID(0)]
