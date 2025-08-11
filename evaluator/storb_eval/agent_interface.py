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
    def act(self, observation: np.ndarray, goal_text: str) -> np.ndarray:
        """
        Take action given current observation and goal text.
        
        Args:
            observation: Environment observation array. Shape varies by environment.
                        For MetaWorld, this is typically a 1D array containing:
                        - Robot joint positions/velocities
                        - Object positions/orientations
                        - Goal positions
            goal_text: Natural language description of the goal
                      (e.g., "push the block to the goal")
            
        Returns:
            action: Continuous action vector for the environment.
                   For MetaWorld, this is typically a 4D vector for robot control.
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