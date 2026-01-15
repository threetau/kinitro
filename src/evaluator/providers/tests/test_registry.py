"""Tests for ProviderRegistry component."""

from unittest.mock import MagicMock

import pytest

from evaluator.providers.registry import (
    BenchmarkSpec,
    EnvSpec,
    ProviderRegistry,
)


class MockProvider:
    """Mock provider for testing."""

    def __init__(self, name: str = "mock"):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def get_benchmark_specs(self, config):
        return [BenchmarkSpec(provider=self._name, benchmark_name="test")]

    def get_env_specs(self, benchmark_spec):
        return [
            EnvSpec(env_name="test_env", benchmark_name="test", provider=self._name)
        ]

    def make_env(self, spec, submission_id=None, save_images=False):
        return MagicMock()

    def get_observation_wrapper(self):
        return None


class TestProviderRegistry:
    """Tests for ProviderRegistry."""

    def setup_method(self):
        """Clear registry before each test."""
        ProviderRegistry.clear()

    def teardown_method(self):
        """Clean up after each test."""
        ProviderRegistry.clear()

    def test_register_provider(self):
        """Test registering a provider."""
        provider = MockProvider("test_provider")
        ProviderRegistry.register(provider)

        assert ProviderRegistry.has_provider("test_provider")
        assert "test_provider" in ProviderRegistry.list_providers()

    def test_register_duplicate_overwrites(self):
        """Test that registering a duplicate provider overwrites."""
        provider1 = MockProvider("test")
        provider2 = MockProvider("test")

        ProviderRegistry.register(provider1)
        ProviderRegistry.register(provider2)

        assert ProviderRegistry.get("test") == provider2

    def test_get_provider(self):
        """Test getting a provider by name."""
        provider = MockProvider("test")
        ProviderRegistry.register(provider)

        result = ProviderRegistry.get("test")
        assert result == provider

    def test_get_nonexistent_provider(self):
        """Test that getting a nonexistent provider raises KeyError."""
        with pytest.raises(KeyError) as excinfo:
            ProviderRegistry.get("nonexistent")

        assert "nonexistent" in str(excinfo.value)

    def test_get_optional_provider(self):
        """Test getting a provider optionally."""
        provider = MockProvider("test")
        ProviderRegistry.register(provider)

        assert ProviderRegistry.get_optional("test") == provider
        assert ProviderRegistry.get_optional("nonexistent") is None

    def test_unregister_provider(self):
        """Test unregistering a provider."""
        provider = MockProvider("test")
        ProviderRegistry.register(provider)

        result = ProviderRegistry.unregister("test")

        assert result is True
        assert not ProviderRegistry.has_provider("test")

    def test_unregister_nonexistent_provider(self):
        """Test unregistering a provider that doesn't exist."""
        result = ProviderRegistry.unregister("nonexistent")
        assert result is False

    def test_list_providers(self):
        """Test listing all registered providers."""
        provider1 = MockProvider("provider1")
        provider2 = MockProvider("provider2")

        ProviderRegistry.register(provider1)
        ProviderRegistry.register(provider2)

        providers = ProviderRegistry.list_providers()
        assert len(providers) == 2
        assert "provider1" in providers
        assert "provider2" in providers

    def test_has_provider(self):
        """Test checking if a provider exists."""
        provider = MockProvider("test")
        ProviderRegistry.register(provider)

        assert ProviderRegistry.has_provider("test") is True
        assert ProviderRegistry.has_provider("nonexistent") is False

    def test_clear_providers(self):
        """Test clearing all providers."""
        provider = MockProvider("test")
        ProviderRegistry.register(provider)

        ProviderRegistry.clear()

        assert len(ProviderRegistry.list_providers()) == 0
        assert not ProviderRegistry.has_provider("test")


class TestBenchmarkSpec:
    """Tests for BenchmarkSpec dataclass."""

    def test_default_values(self):
        """Test default values for BenchmarkSpec."""
        spec = BenchmarkSpec(provider="test", benchmark_name="bench1")

        assert spec.provider == "test"
        assert spec.benchmark_name == "bench1"
        assert spec.config == {}
        assert spec.render_mode == "rgb_array"
        assert spec.camera_names == ("corner",)
        assert spec.camera_attribute == "camera_name"

    def test_str_representation(self):
        """Test string representation."""
        spec = BenchmarkSpec(provider="metaworld", benchmark_name="MT10")
        assert str(spec) == "metaworld/MT10"


class TestEnvSpec:
    """Tests for EnvSpec dataclass."""

    def test_default_values(self):
        """Test default values for EnvSpec."""
        spec = EnvSpec(env_name="test_env", benchmark_name="bench1", provider="test")

        assert spec.env_name == "test_env"
        assert spec.benchmark_name == "bench1"
        assert spec.provider == "test"
        assert spec.config == {}
        assert spec.episodes_per_task == 3
        assert spec.max_episode_steps == 10
        assert spec.render_mode == "rgb_array"

    def test_str_representation(self):
        """Test string representation."""
        spec = EnvSpec(
            env_name="door-open-v2", benchmark_name="MT10", provider="metaworld"
        )
        assert str(spec) == "metaworld/MT10/door-open-v2"
