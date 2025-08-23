"""
Abstract base class defining the standard interface for all agents.

All miner-submitted agents must implement this interface to be evaluated.
"""

from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import torch


class AgentInterface(ABC):
    """
    Standard interface that all miner implementations must follow.

    This ensures a consistent contract between the evaluator and any submitted agent,
    regardless of the underlying model architecture or implementation details.
    """

    def __init__(
        self,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        self.observation_space = observation_space or gym.spaces.Box(
            low=-1, high=1, shape=(100,), dtype=np.float32
        )
        self.action_space = action_space or gym.spaces.Box(
            low=-1, high=1, shape=(4,), dtype=np.float32
        )
        self.seed = seed or np.random.randint(0, 1000000)
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def act(self, obs: dict, **kwargs) -> torch.Tensor:
        """
        Take action given current observation and any additional arguments.
        """
        pass

    def reset(self) -> None:
        """
        Reset agent state for new episode.

        This is called at the beginning of each episode. Stateless agents
        can implement this as a no-op. Agents with internal memory/history
        should reset their state here.
        """
        pass
