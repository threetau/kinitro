"""
Executor registry for task type dispatch.

The registry maps TaskType values to TaskExecutor implementations,
allowing the Orchestrator to dispatch jobs to the appropriate executor.
"""

from typing import Dict, Optional

from core.log import get_logger
from core.tasks import ExecutorNotFoundError, TaskExecutor, TaskType

logger = get_logger(__name__)


class ExecutorRegistry:
    """Registry for TaskExecutor implementations.

    This class provides a central point for registering and retrieving
    executors based on task type. It follows a class-level registry pattern
    for global access.

    Usage:
        # Register an executor
        ExecutorRegistry.register(RLRolloutExecutor())

        # Get an executor for a task type
        executor = ExecutorRegistry.get(TaskType.RL_ROLLOUT)

        # Check if an executor exists
        if ExecutorRegistry.has(TaskType.RL_ROLLOUT):
            ...

        # List all registered task types
        types = ExecutorRegistry.list_types()
    """

    _executors: Dict[TaskType, TaskExecutor] = {}

    @classmethod
    def register(cls, executor: TaskExecutor) -> None:
        """Register an executor for its task type.

        If an executor is already registered for the task type, it will be
        replaced with a warning.

        Args:
            executor: The TaskExecutor to register
        """
        task_type = executor.task_type
        if task_type in cls._executors:
            logger.warning(
                "Overwriting existing executor for task type %s", task_type.value
            )
        cls._executors[task_type] = executor
        logger.info("Registered executor for task type %s", task_type.value)

    @classmethod
    def get(cls, task_type: TaskType) -> TaskExecutor:
        """Get the executor for a task type.

        Args:
            task_type: The task type to get an executor for

        Returns:
            The registered TaskExecutor

        Raises:
            ExecutorNotFoundError: If no executor is registered for the task type
        """
        executor = cls._executors.get(task_type)
        if executor is None:
            raise ExecutorNotFoundError(
                f"No executor registered for task type: {task_type.value}"
            )
        return executor

    @classmethod
    def get_optional(cls, task_type: TaskType) -> Optional[TaskExecutor]:
        """Get the executor for a task type, or None if not registered.

        Args:
            task_type: The task type to get an executor for

        Returns:
            The registered TaskExecutor, or None
        """
        return cls._executors.get(task_type)

    @classmethod
    def has(cls, task_type: TaskType) -> bool:
        """Check if an executor is registered for a task type.

        Args:
            task_type: The task type to check

        Returns:
            True if an executor is registered
        """
        return task_type in cls._executors

    @classmethod
    def unregister(cls, task_type: TaskType) -> bool:
        """Unregister the executor for a task type.

        Args:
            task_type: The task type to unregister

        Returns:
            True if an executor was unregistered, False if none was registered
        """
        if task_type in cls._executors:
            del cls._executors[task_type]
            logger.info("Unregistered executor for task type %s", task_type.value)
            return True
        return False

    @classmethod
    def list_types(cls) -> list[TaskType]:
        """List all registered task types.

        Returns:
            List of registered TaskType values
        """
        return list(cls._executors.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered executors.

        Primarily useful for testing.
        """
        cls._executors.clear()
        logger.info("Cleared all registered executors")
