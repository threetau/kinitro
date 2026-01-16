"""Tests for ExecutorRegistry."""

from datetime import timedelta

import pytest

from core.tasks import (
    ExecutorNotFoundError,
    ResourceSpec,
    TaskContext,
    TaskResult,
    TaskSpec,
    TaskType,
)
from evaluator.executors.registry import ExecutorRegistry


class MockExecutor:
    """Mock executor for testing."""

    task_type = TaskType.RL_ROLLOUT

    async def validate_spec(self, spec: TaskSpec) -> list[str]:
        return []

    async def setup(self, spec: TaskSpec) -> TaskContext:
        return TaskContext(spec=spec, work_dir="/tmp")

    async def execute(self, context: TaskContext) -> TaskResult:
        return TaskResult(task_id=context.spec.task_id, success=True)

    async def teardown(self, context: TaskContext) -> None:
        pass


class TestExecutorRegistry:
    """Tests for ExecutorRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ExecutorRegistry.clear()

    def teardown_method(self):
        """Clear registry after each test."""
        ExecutorRegistry.clear()

    def test_register_executor(self):
        """Test registering an executor."""
        executor = MockExecutor()
        ExecutorRegistry.register(executor)

        assert ExecutorRegistry.has(TaskType.RL_ROLLOUT)
        assert ExecutorRegistry.get(TaskType.RL_ROLLOUT) is executor

    def test_register_overwrites_existing(self):
        """Test that registering overwrites existing executor."""
        executor1 = MockExecutor()
        executor2 = MockExecutor()

        ExecutorRegistry.register(executor1)
        ExecutorRegistry.register(executor2)

        assert ExecutorRegistry.get(TaskType.RL_ROLLOUT) is executor2

    def test_get_nonexistent_raises(self):
        """Test that getting nonexistent executor raises error."""
        with pytest.raises(ExecutorNotFoundError):
            ExecutorRegistry.get(TaskType.RL_ROLLOUT)

    def test_get_optional_returns_none(self):
        """Test that get_optional returns None for nonexistent."""
        assert ExecutorRegistry.get_optional(TaskType.RL_ROLLOUT) is None

    def test_get_optional_returns_executor(self):
        """Test that get_optional returns executor when exists."""
        executor = MockExecutor()
        ExecutorRegistry.register(executor)

        assert ExecutorRegistry.get_optional(TaskType.RL_ROLLOUT) is executor

    def test_has_returns_false_when_not_registered(self):
        """Test has returns False when not registered."""
        assert ExecutorRegistry.has(TaskType.RL_ROLLOUT) is False

    def test_has_returns_true_when_registered(self):
        """Test has returns True when registered."""
        ExecutorRegistry.register(MockExecutor())
        assert ExecutorRegistry.has(TaskType.RL_ROLLOUT) is True

    def test_unregister_existing(self):
        """Test unregistering an existing executor."""
        ExecutorRegistry.register(MockExecutor())
        assert ExecutorRegistry.unregister(TaskType.RL_ROLLOUT) is True
        assert ExecutorRegistry.has(TaskType.RL_ROLLOUT) is False

    def test_unregister_nonexistent(self):
        """Test unregistering a nonexistent executor."""
        assert ExecutorRegistry.unregister(TaskType.RL_ROLLOUT) is False

    def test_list_types_empty(self):
        """Test listing types when registry is empty."""
        assert ExecutorRegistry.list_types() == []

    def test_list_types_with_executors(self):
        """Test listing types with registered executors."""
        ExecutorRegistry.register(MockExecutor())
        types = ExecutorRegistry.list_types()
        assert TaskType.RL_ROLLOUT in types

    def test_clear(self):
        """Test clearing the registry."""
        ExecutorRegistry.register(MockExecutor())
        ExecutorRegistry.clear()
        assert ExecutorRegistry.list_types() == []


class TestTaskSpec:
    """Tests for TaskSpec serialization."""

    def test_to_dict(self):
        """Test TaskSpec serialization to dict."""
        spec = TaskSpec(
            task_type=TaskType.RL_ROLLOUT,
            task_id="test-task-123",
            config={"env_name": "test-env"},
            timeout=timedelta(hours=1),
            resources=ResourceSpec(cpu_cores=2.0, memory_mb=4096),
            submission_id=12345,
            competition_id="comp-1",
            miner_hotkey="5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY",
            artifact_url="https://example.com/artifact.tar.gz",
        )

        data = spec.to_dict()

        assert data["task_type"] == "rl_rollout"
        assert data["task_id"] == "test-task-123"
        assert data["config"] == {"env_name": "test-env"}
        assert data["timeout_seconds"] == 3600.0
        assert data["resources"]["cpu_cores"] == 2.0
        assert data["resources"]["memory_mb"] == 4096

    def test_from_dict(self):
        """Test TaskSpec deserialization from dict."""
        data = {
            "task_type": "rl_rollout",
            "task_id": "test-task-456",
            "config": {"benchmark": "MT1"},
            "timeout_seconds": 7200,
            "resources": {"cpu_cores": 4.0, "gpu_count": 1},
            "submission_id": 67890,
            "competition_id": "comp-2",
            "miner_hotkey": "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            "artifact_url": "https://example.com/model.tar.gz",
        }

        spec = TaskSpec.from_dict(data)

        assert spec.task_type == TaskType.RL_ROLLOUT
        assert spec.task_id == "test-task-456"
        assert spec.config == {"benchmark": "MT1"}
        assert spec.timeout.total_seconds() == 7200
        assert spec.resources.cpu_cores == 4.0
        assert spec.resources.gpu_count == 1

    def test_roundtrip_serialization(self):
        """Test that to_dict/from_dict roundtrip preserves data."""
        original = TaskSpec(
            task_type=TaskType.RL_ROLLOUT,
            task_id="roundtrip-test",
            config={"key": "value"},
            timeout=timedelta(minutes=30),
            resources=ResourceSpec(),
            submission_id=11111,
            competition_id="comp-rt",
            miner_hotkey="5FLSigC9HGRKVhB9FiEo4Y3koPsNmBmLJbpXg2mp1hXcS59Y",
            artifact_url="https://example.com/test.tar.gz",
            job_id=99999,
            env_provider="metaworld",
            benchmark_name="MT10",
        )

        data = original.to_dict()
        restored = TaskSpec.from_dict(data)

        assert restored.task_type == original.task_type
        assert restored.task_id == original.task_id
        assert restored.config == original.config
        assert restored.timeout == original.timeout
        assert restored.resources.cpu_cores == original.resources.cpu_cores
        assert restored.submission_id == original.submission_id
        assert restored.job_id == original.job_id
        assert restored.env_provider == original.env_provider


class TestResourceSpec:
    """Tests for ResourceSpec."""

    def test_default_values(self):
        """Test default resource values."""
        spec = ResourceSpec()

        assert spec.cpu_cores == 1.0
        assert spec.memory_mb == 2048
        assert spec.gpu_count == 0
        assert spec.gpu_memory_mb == 0
        assert spec.storage_mb == 1024

    def test_custom_values(self):
        """Test custom resource values."""
        spec = ResourceSpec(
            cpu_cores=8.0,
            memory_mb=16384,
            gpu_count=2,
            gpu_memory_mb=8192,
            storage_mb=10240,
        )

        assert spec.cpu_cores == 8.0
        assert spec.memory_mb == 16384
        assert spec.gpu_count == 2
        assert spec.gpu_memory_mb == 8192
        assert spec.storage_mb == 10240

    def test_negative_cpu_raises(self):
        """Test that negative CPU raises error."""
        with pytest.raises(ValueError, match="cpu_cores must be non-negative"):
            ResourceSpec(cpu_cores=-1.0)

    def test_negative_memory_raises(self):
        """Test that negative memory raises error."""
        with pytest.raises(ValueError, match="memory_mb must be non-negative"):
            ResourceSpec(memory_mb=-1)

    def test_negative_gpu_raises(self):
        """Test that negative GPU count raises error."""
        with pytest.raises(ValueError, match="gpu_count must be non-negative"):
            ResourceSpec(gpu_count=-1)


class TestTaskResult:
    """Tests for TaskResult."""

    def test_success_result(self):
        """Test creating a success result."""
        result = TaskResult(
            task_id="task-1",
            success=True,
            metrics={"accuracy": 0.95},
            duration_seconds=120.5,
        )

        assert result.success is True
        assert result.metrics["accuracy"] == 0.95
        assert result.error is None

    def test_failure_result(self):
        """Test creating a failure result."""
        result = TaskResult(
            task_id="task-2",
            success=False,
            error="Container failed to start",
            duration_seconds=5.0,
        )

        assert result.success is False
        assert result.error == "Container failed to start"

    def test_to_dict(self):
        """Test TaskResult serialization."""
        result = TaskResult(
            task_id="task-3",
            success=True,
            metrics={"success_rate": 0.8, "avg_reward": 1500.0},
            total_episodes=100,
            duration_seconds=300.0,
        )

        data = result.to_dict()

        assert data["task_id"] == "task-3"
        assert data["success"] is True
        assert data["metrics"]["success_rate"] == 0.8
        assert data["total_episodes"] == 100
