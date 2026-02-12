"""
Miner Policy Template for Kinitro

This is the template for creating a policy that can be evaluated
by Kinitro validators.

Your policy must implement the RobotActor class with:
- reset(task_config): Called at the start of each episode
- act(observation): Called each timestep to get action
- cleanup(): Called when evaluation is complete

Interface:
- act() receives a dict matching Observation
- act() returns a dict matching Action

The policy will be run inside a container and queried by validators
across multiple robotics environments.

IMPORTANT: Observations follow the extensible interface:
- Proprioceptive: proprio dict with keys like ee_pos, ee_quat, gripper, etc.
- Visual: rgb dict with camera images
- Object positions are NOT exposed - you must learn from visual input!
"""

import os
from typing import Any

import numpy as np

# Import from local rl_interface (self-contained for Basilica deployment)
from rl_interface import Action, ActionKeys


def decode_image_array(array_data: list) -> np.ndarray:
    """Decode nested list image data to numpy array."""
    return np.array(array_data)


class RobotActor:
    """
    Your robotics policy implementation.

    This class is instantiated once when the container starts,
    then reset() is called for each new episode, and act() is
    called for each timestep.

    Observation format (dict):
        - proprio: dict with keys like:
            - "ee_pos": [x, y, z]
            - "ee_quat": [qx, qy, qz, qw]
            - "ee_vel_lin": [vx, vy, vz]
            - "ee_vel_ang": [wx, wy, wz]
            - "gripper": [state] in [0, 1]
        - rgb: dict[str, data] - Camera views
            - "corner": First corner camera view (84x84 RGB)
            - "corner2": Second corner camera view (84x84 RGB)
        - extra: dict with task-specific info like task_prompt

    Action format (dict):
        - continuous: dict with keys like:
            - "ee_twist": [vx, vy, vz, wx, wy, wz] in [-1, 1]
            - "gripper": [cmd] in [0, 1]
        - discrete: dict for discrete actions (optional)
    """

    def __init__(self):
        """
        Initialize your policy.

        This is called once when the container starts.
        Load your model weights, set up any required state, etc.
        """
        # Example: Load a trained policy
        # self.policy = torch.load("policy.pt")

        # For demonstration, we'll use a simple random policy
        self.policy = None
        self.current_env = None
        self.current_task = None

        # Image preprocessing settings
        self._image_size = (84, 84)
        self._camera_names = ["corner", "corner2"]

    async def reset(
        self, task_config: dict[str, Any]
    ) -> None:  # Any: task config is env-specific JSON
        """
        Reset policy for a new episode.

        Called at the start of each episode with task information.
        Use this to condition your policy on the task if needed.

        Args:
            task_config: Dict containing:
                - env_name: Environment family (metaworld)
                - task_name: Specific task (e.g., pick-place-v3)
                - seed: Random seed for this episode
                - object_positions: Procedural object positions (for env setup only)
                - target_positions: Procedural target positions (for env setup only)
                - physics_params: Randomized physics parameters
                - domain_randomization: Visual randomization params
        """
        self.current_env = task_config.get("env_name", "")
        self.current_task = task_config.get("task_name", "")

        # Example: You might want to condition your policy on the task
        # self.task_embedding = self.encode_task(task_config)

    async def act(
        self, observation: dict[str, Any]
    ) -> dict[str, Any]:  # Any: obs/action are env-specific JSON
        """
        Get action for current observation.

        This is called every timestep. You have ~100ms to respond.

        Args:
            observation: Dict containing observation fields (proprio, rgb, extra)

        Returns:
            action: Dict containing continuous and/or discrete actions
        """
        # Extract proprioceptive data from new format
        proprio = observation.get("proprio", {})
        ee_pos = np.array(proprio.get("ee_pos", [0, 0, 0]), dtype=np.float32)
        ee_quat = np.array(proprio.get("ee_quat", [0, 0, 0, 1]), dtype=np.float32)
        ee_vel_lin = np.array(proprio.get("ee_vel_lin", [0, 0, 0]), dtype=np.float32)
        ee_vel_ang = np.array(proprio.get("ee_vel_ang", [0, 0, 0]), dtype=np.float32)
        gripper_state = proprio.get("gripper", [0.0])[0]
        _proprio_arr = np.concatenate([ee_pos, ee_quat, ee_vel_lin, ee_vel_ang, [gripper_state]])

        camera_images = observation.get("rgb", {})
        images = {}
        for cam_name, img_data in camera_images.items():
            if img_data is not None:
                images[cam_name] = decode_image_array(img_data)

        _ = images

        # =====================================================
        # REPLACE THIS WITH YOUR POLICY
        # =====================================================

        twist = np.random.uniform(-1, 1, size=6).tolist()
        gripper = [float(np.random.uniform(0, 1))]

        # =====================================================

        return Action(
            continuous={
                ActionKeys.EE_TWIST: twist,
                ActionKeys.GRIPPER: gripper,
            }
        ).model_dump(mode="python")

    async def cleanup(self) -> None:
        """
        Clean up resources.

        Called when evaluation is complete.
        Free any resources, close connections, etc.
        """
        pass


# ==============================================================================
# EXAMPLE: Using a PyTorch Vision Policy
# ==============================================================================


class ExampleVisionPolicy:
    """
    Example of integrating a vision-based policy.

    This placeholder keeps the structure for camera + proprio inputs.
    Replace the model loading and inference with your framework of choice.
    """

    def __init__(self, model_path: str = "policy.pt"):
        self.model = None
        self.image_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.image_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        if os.path.exists(model_path):
            self.model = model_path
        else:
            print(f"Warning: Model not found at {model_path}")

    def preprocess_images(self, images: dict[str, np.ndarray]) -> np.ndarray | None:
        """Preprocess camera images for the policy."""
        if not images:
            return None

        processed = []
        for cam_name in sorted(images.keys()):
            img = images[cam_name]
            img_float = img.astype(np.float32) / 255.0
            img_norm = (img_float - self.image_mean) / self.image_std
            processed.append(img_norm)

        if processed:
            return np.stack(processed, axis=0)
        return None

    def __call__(self, proprio: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        if self.model is None:
            return np.zeros(4)
        _ = self.preprocess_images(images)
        return np.zeros(4)


# ==============================================================================
# EXAMPLE: Multi-Task Policy with Task Embeddings
# ==============================================================================


class ExampleMultiTaskPolicy:
    """
    Example of a multi-task policy that conditions on task info.

    For generalizing across environments, you likely need to:
    1. Embed the task/environment information
    2. Process visual observations through CNN
    3. Concatenate with proprioceptive observations
    4. Feed to a shared policy network
    """

    def __init__(self):
        self.task_embeddings = {
            "metaworld/pick-place-v3": np.array([1, 0, 0, 0]),
            "metaworld/push-v3": np.array([0, 1, 0, 0]),
            "metaworld/reach-v3": np.array([0, 0, 1, 0]),
            "metaworld/door-open-v3": np.array([0, 0, 0, 1]),
            # ... add more as needed
        }
        self.current_embedding = np.zeros(4)

    def reset(self, task_config: dict) -> None:
        """Update task embedding based on current task."""
        task_key = f"{task_config['env_name']}/{task_config['task_name']}"
        self.current_embedding = self.task_embeddings.get(task_key, np.zeros(4))

    def __call__(self, proprio: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        # Concatenate observation with task embedding
        _augmented_obs = np.concatenate([proprio, self.current_embedding])  # noqa: F841

        # Process images through CNN encoder
        # image_features = self.cnn_encoder(images)

        # Concatenate all features
        # full_features = np.concatenate([augmented_obs, image_features])

        # Feed to your policy network
        # action = self.policy_network(full_features)

        # Placeholder: random action
        return np.random.uniform(-1, 1, size=4)
