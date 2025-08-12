"""
Abstract base class defining the standard interface for all agents.

All miner-submitted agents must implement this interface to be evaluated.
"""

from abc import ABC, abstractmethod

import numpy as np


class AgentInterface(ABC):
    """
    Standard interface that all miner implementations must follow.

    This ensures a consistent contract between the evaluator and any submitted agent,
    regardless of the underlying model architecture or implementation details.
    """

    @abstractmethod
    def act(self, obs: np.ndarray, **kwargs) -> np.ndarray:
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
