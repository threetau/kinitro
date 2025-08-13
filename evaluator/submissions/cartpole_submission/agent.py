"""
CartPole submission: a minimal heuristic agent compatible with the evaluator
interface. It handles dict observations produced by the generic wrappers
(`base` key contains the environment's original observation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np

from storb_eval import AgentInterface


class Agent(AgentInterface):
    def __init__(
        self,
        submission_dir: Path,
        observation_space: gym.Space,
        action_space: gym.Space,
        seed: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__(observation_space, action_space, seed, **kwargs)
        self.submission_dir = submission_dir

    def reset(self) -> None:
        return None

    def act(self, observation, **kwargs: Any):  # type: ignore[override]
        # Unwrap dict observations (from wrappers) to the raw env observation
        if isinstance(observation, dict) and "base" in observation:
            observation = observation["base"]

        obs = np.asarray(observation, dtype=np.float32)

        # Simple linear heuristic on CartPole state [x, x_dot, theta, theta_dot]
        # Push right if pole angle + a bit of angular velocity is positive, else left
        theta = float(obs[2]) if obs.shape[0] >= 3 else 0.0
        theta_dot = float(obs[3]) if obs.shape[0] >= 4 else 0.0
        score = theta + 0.25 * theta_dot

        if hasattr(self.action_space, "n"):
            # Discrete action space: return an int 0/1
            return 1 if score > 0.0 else 0

        # Fallback for continuous spaces (unlikely for CartPole-v1)
        low = getattr(self.action_space, "low", np.array([-1.0], dtype=np.float32))
        high = getattr(self.action_space, "high", np.array([1.0], dtype=np.float32))
        act = np.tanh(np.array([score], dtype=np.float32))
        return np.clip(act, low, high).astype(np.float32)


