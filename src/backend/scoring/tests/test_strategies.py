"""Tests for scoring strategies."""

from unittest.mock import MagicMock

import pytest

from core.tasks import TaskResult, TaskType

from backend.scoring import (
    EligibilityResult,
    RLRolloutScoringStrategy,
    ScoringMetrics,
    ScoringStrategyRegistry,
    StrategyNotFoundError,
)


# =============================================================================
# ScoringMetrics Tests
# =============================================================================


class TestScoringMetrics:
    """Tests for ScoringMetrics dataclass."""

    def test_to_dict_all_fields(self):
        """Test serialization with all fields set."""
        metrics = ScoringMetrics(
            success_rate=0.85,
            avg_reward=100.5,
            total_episodes=50,
            score=0.85,
            extra={"custom_metric": 42.0},
        )
        result = metrics.to_dict()

        assert result["success_rate"] == 0.85
        assert result["avg_reward"] == 100.5
        assert result["total_episodes"] == 50
        assert result["score"] == 0.85
        assert result["custom_metric"] == 42.0

    def test_to_dict_partial_fields(self):
        """Test serialization with only some fields set."""
        metrics = ScoringMetrics(success_rate=0.5)
        result = metrics.to_dict()

        assert result == {"success_rate": 0.5}
        assert "avg_reward" not in result
        assert "total_episodes" not in result

    def test_to_dict_empty(self):
        """Test serialization with no fields set."""
        metrics = ScoringMetrics()
        result = metrics.to_dict()

        assert result == {}


# =============================================================================
# RLRolloutScoringStrategy Tests
# =============================================================================


class TestRLRolloutScoringStrategy:
    """Tests for RLRolloutScoringStrategy."""

    @pytest.fixture
    def strategy(self):
        """Create a strategy instance for testing."""
        return RLRolloutScoringStrategy()

    @pytest.fixture
    def mock_competition(self):
        """Create a mock competition with thresholds."""
        comp = MagicMock()
        comp.id = "test-competition"
        comp.min_success_rate = 0.5
        comp.min_avg_reward = 10.0
        comp.task_type = "rl_rollout"
        return comp

    @pytest.fixture
    def mock_result(self):
        """Create a mock BackendEvaluationResult."""
        result = MagicMock()
        result.success_rate = 0.75
        result.avg_reward = 50.0
        result.total_episodes = 100
        result.score = 0.75
        return result

    def test_task_type(self, strategy):
        """Test that task_type property returns correct value."""
        assert strategy.task_type == TaskType.RL_ROLLOUT

    def test_extract_metrics(self, strategy, mock_result):
        """Test extracting metrics from BackendEvaluationResult."""
        metrics = strategy.extract_metrics(mock_result)

        assert metrics.success_rate == 0.75
        assert metrics.avg_reward == 50.0
        assert metrics.total_episodes == 100
        assert metrics.score == 0.75

    def test_extract_metrics_with_none_values(self, strategy):
        """Test extracting metrics when some values are None."""
        result = MagicMock()
        result.success_rate = None
        result.avg_reward = None
        result.total_episodes = None
        result.score = None

        metrics = strategy.extract_metrics(result)

        assert metrics.success_rate is None
        assert metrics.avg_reward is None
        assert metrics.total_episodes is None
        assert metrics.score is None

    def test_extract_metrics_from_task_result(self, strategy):
        """Test extracting metrics from TaskResult."""
        task_result = TaskResult(
            task_id="test-task",
            success=True,
            metrics={
                "success_rate": 0.9,
                "avg_reward": 75.0,
                "score": 0.9,
            },
            total_episodes=50,
        )

        metrics = strategy.extract_metrics_from_task_result(task_result)

        assert metrics.success_rate == 0.9
        assert metrics.avg_reward == 75.0
        assert metrics.total_episodes == 50
        assert metrics.score == 0.9

    # -------------------------------------------------------------------------
    # Eligibility Tests
    # -------------------------------------------------------------------------

    def test_check_eligibility_eligible(self, strategy, mock_competition):
        """Test eligibility check for eligible result."""
        metrics = ScoringMetrics(success_rate=0.8, avg_reward=50.0)

        result = strategy.check_eligibility(metrics, mock_competition)

        assert result.eligible is True
        assert result.reason is None

    def test_check_eligibility_below_success_rate(self, strategy, mock_competition):
        """Test eligibility when success_rate is below threshold."""
        metrics = ScoringMetrics(success_rate=0.3, avg_reward=50.0)

        result = strategy.check_eligibility(metrics, mock_competition)

        assert result.eligible is False
        assert "success_rate" in result.reason
        assert "0.300" in result.reason

    def test_check_eligibility_below_avg_reward(self, strategy, mock_competition):
        """Test eligibility when avg_reward is below threshold."""
        metrics = ScoringMetrics(success_rate=0.8, avg_reward=5.0)

        result = strategy.check_eligibility(metrics, mock_competition)

        assert result.eligible is False
        assert "avg_reward" in result.reason

    def test_check_eligibility_missing_metrics(self, strategy, mock_competition):
        """Test eligibility when required metrics are missing."""
        metrics = ScoringMetrics(success_rate=None, avg_reward=None)

        result = strategy.check_eligibility(metrics, mock_competition)

        assert result.eligible is False
        assert "Missing required metrics" in result.reason

    def test_check_eligibility_at_threshold(self, strategy, mock_competition):
        """Test eligibility at exact threshold values."""
        metrics = ScoringMetrics(success_rate=0.5, avg_reward=10.0)

        result = strategy.check_eligibility(metrics, mock_competition)

        assert result.eligible is True

    # -------------------------------------------------------------------------
    # Score Computation Tests
    # -------------------------------------------------------------------------

    def test_compute_score_uses_existing_score(self, strategy, mock_competition):
        """Test that compute_score uses existing score when available."""
        metrics = ScoringMetrics(success_rate=0.8, score=0.95)

        score = strategy.compute_score(metrics, mock_competition)

        assert score == 0.95

    def test_compute_score_falls_back_to_success_rate(self, strategy, mock_competition):
        """Test that compute_score falls back to success_rate when score is None."""
        metrics = ScoringMetrics(success_rate=0.8, score=None)

        score = strategy.compute_score(metrics, mock_competition)

        assert score == 0.8

    def test_compute_score_returns_zero_when_no_metrics(
        self, strategy, mock_competition
    ):
        """Test that compute_score returns 0 when no relevant metrics."""
        metrics = ScoringMetrics()

        score = strategy.compute_score(metrics, mock_competition)

        assert score == 0.0

    # -------------------------------------------------------------------------
    # Comparison Tests
    # -------------------------------------------------------------------------

    def test_compare_by_success_rate(self, strategy):
        """Test comparison primarily by success_rate."""
        a = ScoringMetrics(success_rate=0.9, avg_reward=50.0)
        b = ScoringMetrics(success_rate=0.7, avg_reward=100.0)

        assert strategy.compare(a, b) == 1  # a > b
        assert strategy.compare(b, a) == -1  # b < a

    def test_compare_by_avg_reward_when_success_rate_equal(self, strategy):
        """Test comparison by avg_reward when success_rate is equal."""
        a = ScoringMetrics(success_rate=0.8, avg_reward=100.0)
        b = ScoringMetrics(success_rate=0.8, avg_reward=50.0)

        assert strategy.compare(a, b) == 1  # a > b
        assert strategy.compare(b, a) == -1  # b < a

    def test_compare_equal_metrics(self, strategy):
        """Test comparison when both metrics are equal."""
        a = ScoringMetrics(success_rate=0.8, avg_reward=50.0)
        b = ScoringMetrics(success_rate=0.8, avg_reward=50.0)

        assert strategy.compare(a, b) == 0

    def test_compare_handles_none_values(self, strategy):
        """Test comparison handles None values gracefully."""
        a = ScoringMetrics(success_rate=0.5, avg_reward=None)
        b = ScoringMetrics(success_rate=None, avg_reward=100.0)

        assert strategy.compare(a, b) == 1  # 0.5 > -inf
        assert strategy.compare(b, a) == -1


# =============================================================================
# ScoringStrategyRegistry Tests
# =============================================================================


class TestScoringStrategyRegistry:
    """Tests for ScoringStrategyRegistry."""

    def test_default_registry_has_rl_rollout(self):
        """Test that default registry has RL rollout strategy registered."""
        registry = ScoringStrategyRegistry.default()

        strategy = registry.get(TaskType.RL_ROLLOUT)
        assert isinstance(strategy, RLRolloutScoringStrategy)

    def test_get_with_string_task_type(self):
        """Test getting strategy with string task type."""
        registry = ScoringStrategyRegistry.default()

        strategy = registry.get("rl_rollout")
        assert isinstance(strategy, RLRolloutScoringStrategy)

    def test_get_raises_for_unknown_task_type(self):
        """Test that get raises StrategyNotFoundError for unknown types."""
        registry = ScoringStrategyRegistry()

        with pytest.raises(StrategyNotFoundError) as exc_info:
            registry.get(TaskType.RL_ROLLOUT)

        assert exc_info.value.task_type == TaskType.RL_ROLLOUT

    def test_get_raises_for_invalid_string_task_type(self):
        """Test that get raises for invalid string task types."""
        registry = ScoringStrategyRegistry()

        with pytest.raises(StrategyNotFoundError) as exc_info:
            registry.get("invalid_type")

        assert exc_info.value.task_type == "invalid_type"

    def test_has_returns_true_for_registered(self):
        """Test that has returns True for registered task types."""
        registry = ScoringStrategyRegistry.default()

        assert registry.has(TaskType.RL_ROLLOUT) is True
        assert registry.has("rl_rollout") is True

    def test_has_returns_false_for_unregistered(self):
        """Test that has returns False for unregistered task types."""
        registry = ScoringStrategyRegistry()

        assert registry.has(TaskType.RL_ROLLOUT) is False
        assert registry.has("invalid_type") is False

    def test_register_and_get(self):
        """Test registering and retrieving a strategy."""
        registry = ScoringStrategyRegistry()
        strategy = RLRolloutScoringStrategy()

        registry.register(strategy)
        retrieved = registry.get(TaskType.RL_ROLLOUT)

        assert retrieved is strategy

    def test_register_overwrites_existing(self):
        """Test that registering overwrites existing strategy."""
        registry = ScoringStrategyRegistry()
        strategy1 = RLRolloutScoringStrategy()
        strategy2 = RLRolloutScoringStrategy()

        registry.register(strategy1)
        registry.register(strategy2)
        retrieved = registry.get(TaskType.RL_ROLLOUT)

        assert retrieved is strategy2

    def test_register_raises_for_non_protocol(self):
        """Test that register raises for non-ScoringStrategy objects."""
        registry = ScoringStrategyRegistry()

        with pytest.raises(TypeError):
            registry.register("not a strategy")  # type: ignore

    def test_list_task_types(self):
        """Test listing registered task types."""
        registry = ScoringStrategyRegistry.default()

        task_types = registry.list_task_types()

        assert TaskType.RL_ROLLOUT in task_types

    def test_unregister(self):
        """Test unregistering a strategy."""
        registry = ScoringStrategyRegistry.default()

        result = registry.unregister(TaskType.RL_ROLLOUT)

        assert result is True
        assert registry.has(TaskType.RL_ROLLOUT) is False

    def test_unregister_nonexistent(self):
        """Test unregistering a strategy that doesn't exist."""
        registry = ScoringStrategyRegistry()

        result = registry.unregister(TaskType.RL_ROLLOUT)

        assert result is False

    def test_clear(self):
        """Test clearing all strategies."""
        registry = ScoringStrategyRegistry.default()
        assert len(registry.list_task_types()) > 0

        registry.clear()

        assert len(registry.list_task_types()) == 0
