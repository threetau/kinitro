#!/usr/bin/env python3
"""
Unit tests for the rollout package components.
Tests EnvManager, BenchmarkSpec, EnvSpec functionality without agent dependencies.
"""

import logging

import pytest
from kinitro_eval.roullout.envs import (
    BenchmarkSpec,
    EnvManager,
    EnvSpec,
)

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRolloutPackage:
    """Test rollout package components without agent dependencies."""

    @pytest.fixture
    def env_manager(self):
        """Create an EnvManager instance."""
        return EnvManager()

    @pytest.fixture
    def mt1_benchmark_spec(self):
        """Create an MT1 benchmark specification."""
        return BenchmarkSpec(
            provider="metaworld", benchmark_name="MT1", config={"env_name": "reach-v3"}
        )

    @pytest.fixture
    def mt10_benchmark_spec(self):
        """Create an MT10 benchmark specification."""
        return BenchmarkSpec(provider="metaworld", benchmark_name="MT10")

    def test_env_manager_task_enumeration(self, env_manager, mt1_benchmark_spec):
        """Test that EnvManager can enumerate tasks correctly."""
        task_specs = env_manager.get_benchmark_envs(mt1_benchmark_spec)

        assert len(task_specs) > 0, "Should find tasks for MT1"
        assert len(task_specs) == 50, f"MT1 should have 50 tasks, got {len(task_specs)}"

        # Check that all tasks are for the specified environment
        for task_spec in task_specs:
            assert "reach-v3" in str(task_spec), (
                f"Task should be for reach-v3, got {task_spec}"
            )

    def test_benchmark_task_counting(self, env_manager):
        """Test that benchmark task counting is correct."""
        # Test MT1
        mt1_spec = BenchmarkSpec(
            provider="metaworld", benchmark_name="MT1", config={"env_name": "reach-v3"}
        )
        mt1_tasks = env_manager.get_benchmark_envs(mt1_spec)
        assert len(mt1_tasks) == 50, f"MT1 should have 50 tasks, got {len(mt1_tasks)}"

        # Test MT10
        mt10_spec = BenchmarkSpec(provider="metaworld", benchmark_name="MT10")
        mt10_tasks = env_manager.get_benchmark_envs(mt10_spec)
        assert len(mt10_tasks) == 500, (
            f"MT10 should have 500 tasks, got {len(mt10_tasks)}"
        )

    def test_env_spec_creation(self, env_manager, mt1_benchmark_spec):
        """Test that EnvSpec objects are created correctly."""
        task_specs = env_manager.get_benchmark_envs(mt1_benchmark_spec)

        # Check first task spec
        env_spec = task_specs[0]
        assert isinstance(env_spec, EnvSpec), "Should return EnvSpec objects"
        assert env_spec.provider == "metaworld", "Provider should be metaworld"
        assert "reach" in env_spec.env_name, "Environment name should contain reach"
        assert env_spec.benchmark_name == "MT1", "Benchmark should be MT1"

    def test_environment_creation(self, env_manager, mt1_benchmark_spec):
        """Test that environments can be created from specs."""
        task_specs = env_manager.get_benchmark_envs(mt1_benchmark_spec)
        env_spec = task_specs[0]

        # Create environment
        env = env_manager.make_env(env_spec)

        assert env is not None, "Environment should be created"

        # Test basic environment functionality
        observation, info = env.reset()
        assert observation is not None, "Should receive observation from reset"
        assert isinstance(observation, dict), "MetaWorld observation should be dict"
        assert "base" in observation, "Should have base observation"

        # Test that action space exists
        assert hasattr(env, "action_space"), "Environment should have action_space"
        assert hasattr(env, "observation_space"), (
            "Environment should have observation_space"
        )

        # Clean up
        env.close()

    def test_benchmark_spec_validation(self):
        """Test BenchmarkSpec validation and creation."""
        # Valid benchmark spec
        valid_spec = BenchmarkSpec(
            provider="metaworld", benchmark_name="MT1", config={"env_name": "reach-v3"}
        )
        assert valid_spec.provider == "metaworld"
        assert valid_spec.benchmark_name == "MT1"
        assert valid_spec.config["env_name"] == "reach-v3"

    def test_multiple_benchmark_handling(self, env_manager):
        """Test handling multiple benchmark specifications."""
        # Create multiple benchmark specs
        mt1_spec = BenchmarkSpec(
            provider="metaworld", benchmark_name="MT1", config={"env_name": "reach-v3"}
        )

        mt10_spec = BenchmarkSpec(provider="metaworld", benchmark_name="MT10")

        # Get tasks for each benchmark
        mt1_tasks = env_manager.get_benchmark_envs(mt1_spec)
        mt10_tasks = env_manager.get_benchmark_envs(mt10_spec)

        # Verify they are different
        assert len(mt1_tasks) != len(mt10_tasks), (
            "Different benchmarks should have different task counts"
        )
        assert len(mt1_tasks) == 50, "MT1 should have 50 tasks"
        assert len(mt10_tasks) == 500, "MT10 should have 500 tasks"

    def test_error_handling_invalid_provider(self, env_manager):
        """Test error handling with invalid provider."""
        invalid_spec = EnvSpec(
            provider="invalid_provider",
            env_name="invalid_env",
            benchmark_name="invalid_benchmark",
        )

        # This should raise an error or handle gracefully
        try:
            env = env_manager.make_env(invalid_spec)
            # If it doesn't crash, clean up
            if env is not None:
                env.close()
        except Exception as e:
            # Expected - invalid providers should fail
            assert "invalid_provider" in str(e) or "not supported" in str(e).lower()

    def test_env_spec_string_representation(self, env_manager, mt1_benchmark_spec):
        """Test that EnvSpec has proper string representation."""
        task_specs = env_manager.get_benchmark_envs(mt1_benchmark_spec)
        env_spec = task_specs[0]

        spec_str = str(env_spec)
        assert "metaworld" in spec_str, "String should contain provider"
        assert "reach" in spec_str, "String should contain environment name"
        assert "MT1" in spec_str, "String should contain benchmark name"


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
