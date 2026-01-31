"""
Miner Policy Implementation

This is where you implement your robotics policy.
The RobotPolicy class is imported by the server and called during evaluation.

Your policy receives:
- Canonical observation: ee_pos_m, ee_quat_xyzw, ee_lin_vel_mps, ee_ang_vel_rps, gripper_01
- Camera images (optional): RGB images from multiple viewpoints

Your policy returns:
- Canonical action: twist_ee_norm (6 values) + gripper_01
"""

import uuid

import numpy as np

# Import from local rl_interface (self-contained for Basilica deployment)
from rl_interface import CanonicalAction, CanonicalObservation


class RobotPolicy:
    """
    Your robotics policy implementation.

    Implement the following methods:
    - reset(): Called at the start of each episode
    - act(): Called every timestep to get action (CanonicalObservation in, CanonicalAction out)
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

    async def act(self, observation: CanonicalObservation):
        """
        Get action for current observation.

        Args:
            observation: Canonical observation with proprioception and optional images

        Returns:
            CanonicalAction
        """
        # TODO: Replace with your policy inference
        # Example:
        # action = self.model(observation)
        # return action
        #
        # NOTE: If seed is provided, ensure your inference is deterministic.
        # The validator may verify that your deployed model matches HuggingFace.

        # Default: random action (seed is already set by server if provided)
        twist = np.random.uniform(-1, 1, size=6)
        gripper = float(np.random.uniform(0, 1))
        return CanonicalAction(twist_ee_norm=twist.tolist(), gripper_01=gripper)

    async def cleanup(self):
        """Called on shutdown. Override to release resources."""
        pass
