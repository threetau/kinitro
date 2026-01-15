"""Tests for ScoringEngine component."""

from unittest.mock import MagicMock

import pytest

from backend.models import (
    BackendEvaluationResult,
    Competition,
)
from backend.scoring import ScoringConfig, ScoringEngine


class TestScoringConfig:
    """Tests for ScoringConfig."""

    def test_default_values(self):
        """Test default configuration values."""
        config = ScoringConfig()
        assert config.owner_uid == 4
        assert config.burn_pct == 0.98

    def test_custom_values(self):
        """Test custom configuration values."""
        config = ScoringConfig(owner_uid=10, burn_pct=0.5)
        assert config.owner_uid == 10
        assert config.burn_pct == 0.5

    def test_burn_pct_clamping_low(self):
        """Test that burn_pct is clamped to [0, 1]."""
        config = ScoringConfig(burn_pct=-0.5)
        assert config.burn_pct == 0.0

    def test_burn_pct_clamping_high(self):
        """Test that burn_pct is clamped to [0, 1]."""
        config = ScoringConfig(burn_pct=1.5)
        assert config.burn_pct == 1.0


class TestScoringEngineEligibility:
    """Tests for ScoringEngine eligibility checks."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session_factory = MagicMock()
        self.config = ScoringConfig(owner_uid=4, burn_pct=0.98)
        self.id_generator = iter(range(1000))
        self.engine = ScoringEngine(
            session_factory=self.session_factory,
            config=self.config,
            id_generator=self.id_generator,
        )

    def test_eligible_result(self):
        """Test that a result meeting all criteria is eligible."""
        result = MagicMock(spec=BackendEvaluationResult)
        result.success_rate = 0.9
        result.avg_reward = 100.0
        result.miner_hotkey = "test_hotkey"

        competition = MagicMock(spec=Competition)
        competition.id = "test_comp"
        competition.min_success_rate = 0.8
        competition.min_avg_reward = 50.0

        assert self.engine.is_eligible(result, competition) is True

    def test_ineligible_low_success_rate(self):
        """Test that a result below success rate threshold is ineligible."""
        result = MagicMock(spec=BackendEvaluationResult)
        result.success_rate = 0.5
        result.avg_reward = 100.0
        result.miner_hotkey = "test_hotkey"

        competition = MagicMock(spec=Competition)
        competition.id = "test_comp"
        competition.min_success_rate = 0.8
        competition.min_avg_reward = 50.0

        assert self.engine.is_eligible(result, competition) is False

    def test_ineligible_low_avg_reward(self):
        """Test that a result below avg reward threshold is ineligible."""
        result = MagicMock(spec=BackendEvaluationResult)
        result.success_rate = 0.9
        result.avg_reward = 30.0
        result.miner_hotkey = "test_hotkey"

        competition = MagicMock(spec=Competition)
        competition.id = "test_comp"
        competition.min_success_rate = 0.8
        competition.min_avg_reward = 50.0

        assert self.engine.is_eligible(result, competition) is False

    def test_ineligible_none_success_rate(self):
        """Test that a result with None success rate is ineligible."""
        result = MagicMock(spec=BackendEvaluationResult)
        result.success_rate = None
        result.avg_reward = 100.0
        result.miner_hotkey = "test_hotkey"

        competition = MagicMock(spec=Competition)
        competition.id = "test_comp"
        competition.min_success_rate = 0.8
        competition.min_avg_reward = 50.0

        assert self.engine.is_eligible(result, competition) is False

    def test_ineligible_none_avg_reward(self):
        """Test that a result with None avg reward is ineligible."""
        result = MagicMock(spec=BackendEvaluationResult)
        result.success_rate = 0.9
        result.avg_reward = None
        result.miner_hotkey = "test_hotkey"

        competition = MagicMock(spec=Competition)
        competition.id = "test_comp"
        competition.min_success_rate = 0.8
        competition.min_avg_reward = 50.0

        assert self.engine.is_eligible(result, competition) is False


class TestScoringEngineWeights:
    """Tests for ScoringEngine weight computation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.session_factory = MagicMock()
        self.config = ScoringConfig(
            owner_uid=4, burn_pct=0.0
        )  # No burn for easier testing
        self.id_generator = iter(range(1000))
        self.engine = ScoringEngine(
            session_factory=self.session_factory,
            config=self.config,
            id_generator=self.id_generator,
        )

    def test_compute_weights_single_miner(self):
        """Test weight computation with a single miner."""
        miner_scores = {"hotkey1": 0.5}

        # Create mock node
        node1 = MagicMock()
        node1.node_id = 1

        nodes = {"hotkey1": node1}

        weights = self.engine.compute_weights(miner_scores, nodes)

        assert weights[1] == 0.5  # Miner weight
        assert weights[4] == 0.5  # Owner gets remainder

    def test_compute_weights_multiple_miners(self):
        """Test weight computation with multiple miners."""
        miner_scores = {"hotkey1": 0.3, "hotkey2": 0.4}

        node1 = MagicMock()
        node1.node_id = 1
        node2 = MagicMock()
        node2.node_id = 2

        nodes = {"hotkey1": node1, "hotkey2": node2}

        weights = self.engine.compute_weights(miner_scores, nodes)

        assert weights[1] == 0.3
        assert weights[2] == 0.4
        assert weights[4] == pytest.approx(0.3, rel=1e-6)  # Owner gets remainder

    def test_compute_weights_fills_zeros(self):
        """Test that unscored nodes get 0.0 weight."""
        miner_scores = {"hotkey1": 0.5}

        node1 = MagicMock()
        node1.node_id = 1
        node2 = MagicMock()
        node2.node_id = 2

        nodes = {"hotkey1": node1, "hotkey2": node2}

        weights = self.engine.compute_weights(miner_scores, nodes)

        assert weights[1] == 0.5
        assert weights[2] == 0.0
        assert weights[4] == 0.5  # Owner gets remainder
