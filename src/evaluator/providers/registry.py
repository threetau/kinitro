"""
Provider registry for Kinitro evaluator.

Provides a plugin architecture for environment providers (metaworld, swarm, etc.).
Extracted from EnvManager for better separation of concerns.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Type, runtime_checkable

import gymnasium as gym
from gymnasium import ObservationWrapper

from core.log import get_logger

logger = get_logger(__name__)


@dataclass
class BenchmarkSpec:
    """Specification for a benchmark and its environments."""

    provider: str  # "metaworld", "swarm", etc.
    benchmark_name: str  # "MT1", "MT10", etc.
    config: Dict[str, Any] = field(default_factory=dict)
    render_mode: str | None = "rgb_array"
    camera_names: tuple[str, ...] = ("corner",)
    camera_attribute: str | None = "camera_name"

    def __str__(self) -> str:
        return f"{self.provider}/{self.benchmark_name}"


@dataclass
class EnvSpec:
    """Specification for a single environment instance."""

    env_name: str
    benchmark_name: str
    provider: str
    config: Dict[str, Any] = field(default_factory=dict)

    # Runtime controls
    episodes_per_task: int = 3
    max_episode_steps: int = 10
    render_mode: str | None = "rgb_array"

    # Observation capture options
    camera_attribute: str | None = "camera_name"
    camera_names: tuple[str, ...] = ("corner",)

    def __str__(self) -> str:
        return f"{self.provider}/{self.benchmark_name}/{self.env_name}"


@runtime_checkable
class EnvironmentProvider(Protocol):
    """
    Interface for environment providers.

    Environment providers are responsible for:
    - Providing benchmark specifications for a given config
    - Creating gymnasium environments from EnvSpec
    - Optionally providing observation wrappers
    """

    @property
    def name(self) -> str:
        """Return the unique name of this provider (e.g., 'metaworld', 'swarm')."""
        ...

    def get_benchmark_specs(self, config: Dict[str, Any]) -> List[BenchmarkSpec]:
        """
        Get benchmark specifications from a configuration.

        Args:
            config: Provider-specific configuration

        Returns:
            List of BenchmarkSpec objects
        """
        ...

    def get_env_specs(self, benchmark_spec: BenchmarkSpec) -> List[EnvSpec]:
        """
        Get environment specifications for a benchmark.

        Args:
            benchmark_spec: The benchmark specification

        Returns:
            List of EnvSpec objects for environments in this benchmark
        """
        ...

    def make_env(
        self,
        spec: EnvSpec,
        submission_id: str | None = None,
        save_images: bool = False,
    ) -> gym.Env:
        """
        Create a gymnasium environment from an EnvSpec.

        Args:
            spec: The environment specification
            submission_id: Optional submission ID for logging/debugging
            save_images: Whether to save rendered images

        Returns:
            A gymnasium environment
        """
        ...

    def get_observation_wrapper(self) -> Type[ObservationWrapper] | None:
        """
        Get the observation wrapper class for this provider.

        Returns:
            The observation wrapper class, or None if no wrapper is needed
        """
        ...


class ProviderRegistry:
    """
    Registry for environment providers.

    This class provides:
    - Registration of provider implementations
    - Lookup of providers by name
    - Listing of available providers
    """

    _providers: Dict[str, EnvironmentProvider] = {}
    _initialized: bool = False

    @classmethod
    def register(cls, provider: EnvironmentProvider) -> None:
        """
        Register an environment provider.

        Args:
            provider: The provider implementation to register

        Raises:
            ValueError: If a provider with this name is already registered
        """
        name = provider.name
        if name in cls._providers:
            logger.warning(f"Provider '{name}' is already registered, overwriting")
        cls._providers[name] = provider
        logger.info(f"Registered environment provider: {name}")

    @classmethod
    def unregister(cls, name: str) -> bool:
        """
        Unregister an environment provider.

        Args:
            name: The provider name to unregister

        Returns:
            True if the provider was unregistered, False if not found
        """
        if name in cls._providers:
            del cls._providers[name]
            logger.info(f"Unregistered environment provider: {name}")
            return True
        return False

    @classmethod
    def get(cls, name: str) -> EnvironmentProvider:
        """
        Get a provider by name.

        Args:
            name: The provider name

        Returns:
            The provider implementation

        Raises:
            KeyError: If the provider is not registered
        """
        if name not in cls._providers:
            available = ", ".join(cls._providers.keys()) or "none"
            raise KeyError(
                f"Provider '{name}' not found. Available providers: {available}"
            )
        return cls._providers[name]

    @classmethod
    def get_optional(cls, name: str) -> Optional[EnvironmentProvider]:
        """
        Get a provider by name, returning None if not found.

        Args:
            name: The provider name

        Returns:
            The provider implementation, or None if not found
        """
        return cls._providers.get(name)

    @classmethod
    def list_providers(cls) -> List[str]:
        """
        List all registered provider names.

        Returns:
            List of provider names
        """
        return list(cls._providers.keys())

    @classmethod
    def has_provider(cls, name: str) -> bool:
        """
        Check if a provider is registered.

        Args:
            name: The provider name

        Returns:
            True if the provider is registered
        """
        return name in cls._providers

    @classmethod
    def clear(cls) -> None:
        """Clear all registered providers (useful for testing)."""
        cls._providers.clear()
        cls._initialized = False
        logger.info("Cleared all environment providers")

    @classmethod
    def initialize_default_providers(cls) -> None:
        """
        Initialize and register the default providers (metaworld, swarm).

        This should be called once at application startup.
        """
        if cls._initialized:
            return

        # Import and register default providers
        try:
            from .metaworld_provider import MetaworldProvider  # noqa: PLC0415

            cls.register(MetaworldProvider())
        except ImportError as e:
            logger.warning(f"Failed to load metaworld provider: {e}")

        try:
            from .swarm_provider import SwarmProvider  # noqa: PLC0415

            cls.register(SwarmProvider())
        except ImportError as e:
            logger.warning(f"Failed to load swarm provider: {e}")

        cls._initialized = True
        logger.info(
            f"Initialized {len(cls._providers)} default providers: "
            f"{', '.join(cls.list_providers())}"
        )
