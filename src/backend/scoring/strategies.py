"""
Scoring strategy interfaces and implementations.

Each task type can have its own scoring strategy that defines how to:
- Extract metrics from task results
- Check eligibility against competition thresholds
- Compute final scores
- Compare results for ranking
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from core.tasks import TaskResult, TaskType

if TYPE_CHECKING:
    from backend.models import BackendEvaluationResult, Competition


class StrategyNotFoundError(Exception):
    """Raised when no strategy is registered for a task type."""

    def __init__(self, task_type: TaskType | str):
        self.task_type = task_type
        super().__init__(f"No scoring strategy registered for task type: {task_type}")


@dataclass
class ScoringMetrics:
    """Container for extracted metrics from a task result.
    
    This provides a typed container for the most common metrics while
    allowing task-specific extras via the `extra` dict.
    """

    # Common metrics used for eligibility/ranking
    success_rate: float | None = None
    avg_reward: float | None = None
    total_episodes: int | None = None
    score: float | None = None

    # Task-specific additional metrics
    extra: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage/serialization."""
        result: dict[str, Any] = {}
        if self.success_rate is not None:
            result["success_rate"] = self.success_rate
        if self.avg_reward is not None:
            result["avg_reward"] = self.avg_reward
        if self.total_episodes is not None:
            result["total_episodes"] = self.total_episodes
        if self.score is not None:
            result["score"] = self.score
        if self.extra:
            result.update(self.extra)
        return result


@dataclass
class EligibilityResult:
    """Result of an eligibility check."""

    eligible: bool
    reason: str | None = None


@runtime_checkable
class ScoringStrategy(Protocol):
    """Interface for task-type-specific scoring logic.
    
    Implement this protocol to add scoring support for new task types.
    The strategy handles all task-specific logic for:
    - Extracting scoreable metrics from results
    - Checking if results meet competition thresholds
    - Computing final scores
    - Comparing results for ranking
    """

    @property
    def task_type(self) -> TaskType:
        """The task type this strategy handles."""
        ...

    def extract_metrics(
        self, result: BackendEvaluationResult
    ) -> ScoringMetrics:
        """Extract scoreable metrics from an evaluation result.
        
        Args:
            result: The evaluation result to extract metrics from
            
        Returns:
            ScoringMetrics with extracted values
        """
        ...

    def extract_metrics_from_task_result(
        self, result: TaskResult
    ) -> ScoringMetrics:
        """Extract scoreable metrics from a TaskResult.
        
        This is useful during evaluation when we have TaskResult
        but not yet a BackendEvaluationResult.
        
        Args:
            result: The task result to extract metrics from
            
        Returns:
            ScoringMetrics with extracted values
        """
        ...

    def check_eligibility(
        self,
        metrics: ScoringMetrics,
        competition: Competition,
    ) -> EligibilityResult:
        """Check if metrics meet competition eligibility thresholds.
        
        Args:
            metrics: The extracted metrics to check
            competition: The competition with threshold configuration
            
        Returns:
            EligibilityResult indicating eligibility and reason if not
        """
        ...

    def compute_score(
        self,
        metrics: ScoringMetrics,
        competition: Competition,
    ) -> float:
        """Compute a final score from metrics.
        
        This score is used for ranking and may be a combination of
        multiple metrics depending on the task type.
        
        Args:
            metrics: The extracted metrics
            competition: Competition with scoring configuration
            
        Returns:
            Final computed score
        """
        ...

    def compare(
        self,
        a: ScoringMetrics,
        b: ScoringMetrics,
    ) -> int:
        """Compare two results for ranking.
        
        Args:
            a: First set of metrics
            b: Second set of metrics
            
        Returns:
            -1 if a < b, 0 if equal, 1 if a > b
        """
        ...


class RLRolloutScoringStrategy:
    """Scoring strategy for RL rollout tasks.
    
    Uses success_rate as primary metric and avg_reward as secondary.
    Eligibility is determined by min_success_rate and min_avg_reward thresholds.
    """

    @property
    def task_type(self) -> TaskType:
        return TaskType.RL_ROLLOUT

    def extract_metrics(
        self, result: BackendEvaluationResult
    ) -> ScoringMetrics:
        """Extract RL-specific metrics from evaluation result."""
        return ScoringMetrics(
            success_rate=result.success_rate,
            avg_reward=result.avg_reward,
            total_episodes=result.total_episodes,
            score=result.score,
        )

    def extract_metrics_from_task_result(
        self, result: TaskResult
    ) -> ScoringMetrics:
        """Extract RL-specific metrics from TaskResult."""
        return ScoringMetrics(
            success_rate=result.metrics.get("success_rate"),
            avg_reward=result.metrics.get("avg_reward"),
            total_episodes=result.total_episodes,
            score=result.metrics.get("score"),
        )

    def check_eligibility(
        self,
        metrics: ScoringMetrics,
        competition: Competition,
    ) -> EligibilityResult:
        """Check RL eligibility based on success rate and avg reward thresholds."""
        if metrics.success_rate is None or metrics.avg_reward is None:
            return EligibilityResult(
                eligible=False,
                reason="Missing required metrics (success_rate or avg_reward)",
            )

        if metrics.success_rate < competition.min_success_rate:
            return EligibilityResult(
                eligible=False,
                reason=(
                    f"success_rate {metrics.success_rate:.3f} below "
                    f"threshold {competition.min_success_rate:.3f}"
                ),
            )

        if metrics.avg_reward < competition.min_avg_reward:
            return EligibilityResult(
                eligible=False,
                reason=(
                    f"avg_reward {metrics.avg_reward:.3f} below "
                    f"threshold {competition.min_avg_reward}"
                ),
            )

        return EligibilityResult(eligible=True)

    def compute_score(
        self,
        metrics: ScoringMetrics,
        competition: Competition,
    ) -> float:
        """Compute score for RL tasks.
        
        Currently uses success_rate as the primary score metric.
        Could be extended to use weighted combinations based on
        competition.scoring_config.
        """
        # Use the pre-computed score if available
        if metrics.score is not None:
            return metrics.score
        
        # Otherwise, use success_rate as the score
        if metrics.success_rate is not None:
            return metrics.success_rate
        
        return 0.0

    def compare(
        self,
        a: ScoringMetrics,
        b: ScoringMetrics,
    ) -> int:
        """Compare RL results by success_rate, then avg_reward."""
        # Primary: success_rate (higher is better)
        a_sr = a.success_rate if a.success_rate is not None else float("-inf")
        b_sr = b.success_rate if b.success_rate is not None else float("-inf")
        
        if a_sr != b_sr:
            return 1 if a_sr > b_sr else -1
        
        # Secondary: avg_reward (higher is better)
        a_ar = a.avg_reward if a.avg_reward is not None else float("-inf")
        b_ar = b.avg_reward if b.avg_reward is not None else float("-inf")
        
        if a_ar != b_ar:
            return 1 if a_ar > b_ar else -1
        
        return 0
