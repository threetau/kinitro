# Task Executors

This package contains implementations of `TaskExecutor` for different task types in Kinitro.

## Architecture

The executor pattern decouples task execution logic from the orchestrator, allowing new task types to be added without modifying the core evaluation infrastructure.

```
Orchestrator
    │
    ▼
ExecutorRegistry.get(task_type)
    │
    ▼
TaskExecutor (interface)
    │
    ├── RLRolloutExecutor (rl_rollout)
    └── TrainingExecutor (training_run) [future]
    ...
```

## Creating a New Executor

### 1. Define Your Task Type

Add your task type to `src/core/tasks.py`:

```python
class TaskType(StrEnum):
    RL_ROLLOUT = "rl_rollout"
    YOUR_NEW_TYPE = "your_new_type"  # Add this
```

### 2. Implement TaskExecutor

Create a new file in `src/evaluator/executors/`:

```python
# src/evaluator/executors/your_executor.py

from core.tasks import TaskContext, TaskExecutor, TaskResult, TaskSpec, TaskType

class YourExecutor:
    """Executor for your task type."""

    task_type = TaskType.YOUR_NEW_TYPE

    def __init__(self, config):
        self.config = config

    async def validate_spec(self, spec: TaskSpec) -> list[str]:
        """Return validation errors, or empty list if valid."""
        errors = []
        if not spec.config.get("required_field"):
            errors.append("required_field is missing")
        return errors

    async def setup(self, spec: TaskSpec) -> TaskContext:
        """Prepare execution environment."""
        # Create containers, initialize resources
        context = TaskContext(
            spec=spec,
            work_dir="/tmp/your-task",
            state={"your_state": "here"},
        )
        return context

    async def execute(self, context: TaskContext) -> TaskResult:
        """Run the task."""
        try:
            # Your execution logic
            metrics = {"score": 0.95}
            return TaskResult(
                task_id=context.spec.task_id,
                success=True,
                metrics=metrics,
            )
        except Exception as e:
            return TaskResult(
                task_id=context.spec.task_id,
                success=False,
                error=str(e),
            )

    async def teardown(self, context: TaskContext) -> None:
        """Clean up resources."""
        # Release containers, close connections, etc.
        pass
```

### 3. Register Your Executor

Register in the orchestrator's `_register_default_executors()`:

```python
def _register_default_executors(self) -> None:
    ExecutorRegistry.register(RLRolloutExecutor(self.config))
    ExecutorRegistry.register(YourExecutor(self.config))  # Add this
```

Or register dynamically:

```python
from evaluator.executors import ExecutorRegistry
from evaluator.executors.your_executor import YourExecutor

ExecutorRegistry.register(YourExecutor(config))
```

### 4. Update Competition Model

If needed, add competition support for your task type. Competitions use the `task_type` field to determine which executor handles evaluations.

## TaskSpec Configuration

The `TaskSpec.config` dict contains task-type-specific configuration. For RL rollouts, this includes:

- `env_provider`: Environment provider name (e.g., "metaworld", "swarm")
- `benchmark_name`: Benchmark identifier (e.g., "MT1", "MT10")
- `config`: Nested benchmark configuration

Your executor can define its own configuration schema.

## TaskResult Metrics

The `TaskResult.metrics` dict should contain numerical metrics that can be used for scoring:

```python
# RL rollout metrics
{
    "success_rate": 0.85,
    "avg_reward": 1500.0,
    "total_episodes": 100,
}

# Your custom metrics
{
    "accuracy": 0.95,
    "latency_ms": 150.0,
    "custom_score": 42.0,
}
```

## Testing

Add tests in `src/evaluator/executors/tests/`:

```python
# test_your_executor.py

import pytest
from evaluator.executors.your_executor import YourExecutor

class TestYourExecutor:
    def test_validate_spec_valid(self):
        # ...

    async def test_execute_success(self):
        # ...
```

## Files

- `registry.py` - ExecutorRegistry for task type dispatch
- `rl_rollout.py` - RLRolloutExecutor for RL evaluation tasks
- `tests/` - Unit tests
