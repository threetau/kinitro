"""Base classes for robotics environments."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class TaskConfig:
    """
    Procedurally generated task specification.

    Each evaluation uses a unique TaskConfig generated from a seed,
    ensuring miners cannot memorize specific scenarios.
    """

    env_name: str
    task_name: str
    seed: int

    # Procedural parameters - vary per task instance
    object_positions: np.ndarray = field(default_factory=lambda: np.zeros(3))
    target_positions: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Physics randomization
    physics_params: dict[str, float] = field(default_factory=dict)

    # Domain randomization (visual, etc.)
    domain_randomization: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization to miner."""
        return {
            "env_name": self.env_name,
            "task_name": self.task_name,
            "seed": self.seed,
            "object_positions": self.object_positions.tolist(),
            "target_positions": self.target_positions.tolist(),
            "physics_params": self.physics_params,
            "domain_randomization": self.domain_randomization,
        }


@dataclass
class EpisodeResult:
    """Result of a single episode evaluation."""

    success: bool
    total_reward: float
    timesteps: int
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        """Reward per timestep (higher is better)."""
        if self.timesteps == 0:
            return 0.0
        return self.total_reward / self.timesteps


class RoboticsEnvironment(ABC):
    """
    Abstract base class for all robotics environments.

    Subclasses must implement:
    - generate_task: Create procedural task from seed
    - reset: Initialize environment with task config
    - step: Execute action and return results
    - get_success: Check if task was completed successfully
    """

    @property
    @abstractmethod
    def env_name(self) -> str:
        """Environment family name (e.g., 'metaworld', 'dm_control')."""
        pass

    @property
    @abstractmethod
    def task_name(self) -> str:
        """Specific task name (e.g., 'pick-place-v2', 'walker-walk')."""
        pass

    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of observation array."""
        pass

    @property
    @abstractmethod
    def action_shape(self) -> tuple[int, ...]:
        """Shape of action array."""
        pass

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Returns (low, high) bounds for actions. Default: [-1, 1]."""
        shape = self.action_shape
        return (
            np.full(shape, -1.0, dtype=np.float32),
            np.full(shape, 1.0, dtype=np.float32),
        )

    @abstractmethod
    def generate_task(self, seed: int) -> TaskConfig:
        """
        Generate a procedural task configuration from seed.

        The seed should deterministically produce the same task config,
        allowing validators to reproduce evaluations.

        Args:
            seed: Random seed for procedural generation

        Returns:
            TaskConfig with randomized positions, physics, etc.
        """
        pass

    @abstractmethod
    def reset(self, task_config: TaskConfig) -> np.ndarray:
        """
        Reset environment with the given task configuration.

        Args:
            task_config: Procedurally generated task specification

        Returns:
            Initial observation as numpy array
        """
        pass

    @abstractmethod
    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute action in environment.

        Args:
            action: Action array matching action_shape

        Returns:
            Tuple of (observation, reward, done, info)
        """
        pass

    @abstractmethod
    def get_success(self) -> bool:
        """
        Check if the current episode achieved success.

        This is separate from 'done' - an episode can be done
        (timeout, failure) without being successful.

        Returns:
            True if task was completed successfully
        """
        pass

    def close(self) -> None:
        """Clean up environment resources. Override if needed."""
        pass

    def render(self) -> np.ndarray | None:
        """Render environment frame. Override if needed."""
        return None
