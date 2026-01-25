"""
Miner Policy Implementation

This is where you implement your robotics policy.
The RobotPolicy class is imported by the server and called during evaluation.

Your policy receives:
- Proprioceptive observations: end-effector position + gripper state
- Camera images (optional): RGB images from multiple viewpoints

Your policy returns:
- Action: 4D numpy array with values in [-1, 1]
"""

import uuid


class RobotPolicy:
    """
    Your robotics policy implementation.

    Implement the following methods:
    - reset(): Called at the start of each episode
    - act(): Called every timestep to get action
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

    async def act(self, observation, images=None, seed=None):
        """
        Get action for current observation.

        Args:
            observation: numpy array of proprioceptive state
            images: optional dict of camera images
            seed: optional seed for deterministic inference (used for verification)

        Returns:
            Action as list or numpy array (4D for MetaWorld)
        """
        # TODO: Replace with your policy inference
        # Example:
        # action = self.model(observation)
        # return action
        #
        # NOTE: If seed is provided, ensure your inference is deterministic.
        # The validator may verify that your deployed model matches HuggingFace.

        # Default: random action (seed is already set by server if provided)
        import numpy as np

        return np.random.uniform(-1, 1, size=4).astype(np.float32)

    async def cleanup(self):
        """Called on shutdown. Override to release resources."""
        pass
