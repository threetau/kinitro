"""
Scoring strategy registry for task type dispatch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from core.log import get_logger
from core.tasks import TaskType

from .strategies import (
    RLRolloutScoringStrategy,
    ScoringStrategy,
    StrategyNotFoundError,
)

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)


class ScoringStrategyRegistry:
    """Registry for scoring strategies.
    
    Maps task types to their corresponding scoring strategies.
    Strategies must implement the ScoringStrategy protocol.
    
    Example usage:
        # Get the default registry with all built-in strategies
        registry = ScoringStrategyRegistry.default()
        
        # Get a strategy for a task type
        strategy = registry.get(TaskType.RL_ROLLOUT)
        
        # Register a custom strategy
        registry.register(MyCustomStrategy())
    """

    _strategies: dict[TaskType, ScoringStrategy]

    def __init__(self) -> None:
        """Create an empty registry."""
        self._strategies = {}

    @classmethod
    def default(cls) -> "ScoringStrategyRegistry":
        """Create a registry with all built-in strategies registered."""
        registry = cls()
        registry.register(RLRolloutScoringStrategy())
        return registry

    def register(self, strategy: ScoringStrategy) -> None:
        """Register a scoring strategy for its task type.
        
        Args:
            strategy: The strategy to register. Must implement ScoringStrategy protocol.
            
        Raises:
            TypeError: If strategy doesn't implement ScoringStrategy protocol.
        """
        if not isinstance(strategy, ScoringStrategy):
            raise TypeError(
                f"Strategy must implement ScoringStrategy protocol, got {type(strategy)}"
            )
        
        task_type = strategy.task_type
        if task_type in self._strategies:
            logger.warning(
                "Overwriting existing strategy for task type %s: %s -> %s",
                task_type,
                type(self._strategies[task_type]).__name__,
                type(strategy).__name__,
            )
        
        self._strategies[task_type] = strategy
        logger.info(
            "Registered scoring strategy for task type %s: %s",
            task_type,
            type(strategy).__name__,
        )

    def get(self, task_type: TaskType | str) -> ScoringStrategy:
        """Get the scoring strategy for a task type.
        
        Args:
            task_type: The task type to get a strategy for
            
        Returns:
            The registered ScoringStrategy
            
        Raises:
            StrategyNotFoundError: If no strategy is registered for the task type
        """
        # Convert string to TaskType if needed
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                raise StrategyNotFoundError(task_type)
        
        strategy = self._strategies.get(task_type)
        if strategy is None:
            raise StrategyNotFoundError(task_type)
        
        return strategy

    def has(self, task_type: TaskType | str) -> bool:
        """Check if a strategy is registered for a task type.
        
        Args:
            task_type: The task type to check
            
        Returns:
            True if a strategy is registered, False otherwise
        """
        if isinstance(task_type, str):
            try:
                task_type = TaskType(task_type)
            except ValueError:
                return False
        
        return task_type in self._strategies

    def list_task_types(self) -> list[TaskType]:
        """List all task types that have registered strategies.
        
        Returns:
            List of TaskType values with registered strategies
        """
        return list(self._strategies.keys())

    def unregister(self, task_type: TaskType) -> bool:
        """Unregister a strategy for a task type.
        
        Args:
            task_type: The task type to unregister
            
        Returns:
            True if a strategy was unregistered, False if none was registered
        """
        if task_type in self._strategies:
            del self._strategies[task_type]
            logger.info("Unregistered scoring strategy for task type %s", task_type)
            return True
        return False

    def clear(self) -> None:
        """Remove all registered strategies."""
        self._strategies.clear()
        logger.info("Cleared all scoring strategies")
