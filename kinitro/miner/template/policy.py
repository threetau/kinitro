"""
Miner Policy Implementation

This is where you implement your robotics policy.
The RobotPolicy class is imported by the server and called during evaluation.

Your policy receives:
- Observation with proprio dict: ee_pos, ee_quat, ee_vel_lin, ee_vel_ang, gripper
- Camera images (optional): RGB images from multiple viewpoints in rgb dict

Your policy returns:
- Action with continuous dict: ee_twist (6 values), gripper (1 value)
"""

import uuid

import numpy as np

# Import from local rl_interface (self-contained for Basilica deployment)
from rl_interface import Action, ActionKeys, Observation


class RobotPolicy:
    """
    Your robotics policy implementation.

    Implement the following methods:
    - reset(): Called at the start of each episode
    - act(): Called every timestep to get action (Observation in, Action out)
    """

    def __init__(self):
        """Initialize your policy. Load model weights here."""
        self._episode_id = None
        self._task_config = None
        # TODO: Load your model here
        # self.model = torch.load("model.pt")

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return True  # Return True if your model loaded successfully

    async def reset(self, task_config: dict) -> str:
        """Reset policy for a new episode."""
        self._task_config = task_config
        self._episode_id = uuid.uuid4().hex
        return self._episode_id

    async def act(self, observation: Observation):
        """
        Get action for current observation.

        Args:
            observation: Observation with proprio dict and optional images

        Returns:
            Action
        """
        # TODO: Replace with your policy inference
        # Example:
        # action = self.model(observation)
        # return action
        #
        # NOTE: If seed is provided, ensure your inference is deterministic.
        # The validator may verify that your deployed model matches HuggingFace.

        # Default: random action (seed is already set by server if provided)
        twist = np.random.uniform(-1, 1, size=6).tolist()
        gripper = [float(np.random.uniform(0, 1))]
        return Action(
            continuous={
                ActionKeys.EE_TWIST: twist,
                ActionKeys.GRIPPER: gripper,
            }
        )

    async def cleanup(self):
        """Called on shutdown. Override to release resources."""
        pass
