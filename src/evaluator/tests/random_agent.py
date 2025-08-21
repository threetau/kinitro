#!/usr/bin/env python3
"""
Random action agent that implements AgentInterface for testing.
"""

import gymnasium as gym
import numpy as np
import torch

from kinitro_eval.agent_interface import AgentInterface


class RandomActionAgent(AgentInterface):
    """Agent that sends random actions for testing purposes."""

    def __init__(
        self,
        observation_space: gym.Space | None = None,
        action_space: gym.Space | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        # Default to MetaWorld-compatible spaces
        if observation_space is None:
            observation_space = gym.spaces.Dict(
                {
                    "base": gym.spaces.Box(
                        low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
                    ),
                    "observation.image": gym.spaces.Box(
                        low=0, high=255, shape=(3, 480, 480), dtype=np.uint8
                    ),
                    "observation.image2": gym.spaces.Box(
                        low=0, high=255, shape=(3, 480, 480), dtype=np.uint8
                    ),
                    "observation.image3": gym.spaces.Box(
                        low=0, high=255, shape=(3, 480, 480), dtype=np.uint8
                    ),
                }
            )

        if action_space is None:
            action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

        super().__init__(observation_space, action_space, seed, **kwargs)

        print(f"RandomActionAgent initialized with seed {self.seed}")

    def act(self, obs: dict, **kwargs) -> torch.Tensor:
        """Return random actions in the action space range."""
        # Generate random action in [-1, 1] range for MetaWorld
        action = self.rng.uniform(-1.0, 1.0, size=self.action_space.shape)
        return torch.tensor(action, dtype=torch.float32)

    def reset(self) -> None:
        """Reset agent state (no-op for random agent)."""
        pass
