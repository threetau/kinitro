"""
Miner Policy Template for Kinitro

This is the template for creating a policy that can be evaluated
by Kinitro validators.

Your policy must implement the RobotActor class with:
- reset(task_config): Called at the start of each episode
- act(observation): Called each timestep to get action
- cleanup(): Called when evaluation is complete

Canonical interface:
- act() receives a dict matching CanonicalObservation
- act() returns a dict matching CanonicalAction

The policy will be run inside a container and queried by validators
across multiple MetaWorld robotics environments.

IMPORTANT: Observations follow the canonical interface:
- Proprioceptive: ee_pos_m, ee_quat_xyzw, ee_lin_vel_mps, ee_ang_vel_rps, gripper_01
- Visual: RGB camera images from corner cameras (nested lists)
- Object positions are NOT exposed - you must learn from visual input!
"""

import os
from typing import Any

import numpy as np

from kinitro.rl_interface import CanonicalAction


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
        - ee_pos_m: [x, y, z]
        - ee_quat_xyzw: [qx, qy, qz, qw]
        - ee_lin_vel_mps: [vx, vy, vz]
        - ee_ang_vel_rps: [wx, wy, wz]
        - gripper_01: float in [0, 1]
        - rgb: dict[str, list] - Camera views as nested lists
            - "corner": First corner camera view (84x84 RGB)
            - "corner2": Second corner camera view (84x84 RGB)

    Action format (dict):
        - twist_ee_norm: [vx, vy, vz, wx, wy, wz] in [-1, 1]
        - gripper_01: float in [0, 1]
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

    async def reset(self, task_config: dict[str, Any]) -> None:
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

    async def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        """
        Get action for current observation.

        This is called every timestep. You have ~100ms to respond.

        Args:
            observation: Dict containing canonical observation fields

        Returns:
            action: Dict containing twist_ee_norm and gripper_01 (CanonicalAction)
        """
        ee_pos = np.array(observation["ee_pos_m"], dtype=np.float32)
        ee_quat = np.array(observation["ee_quat_xyzw"], dtype=np.float32)
        ee_lin_vel = np.array(observation["ee_lin_vel_mps"], dtype=np.float32)
        ee_ang_vel = np.array(observation["ee_ang_vel_rps"], dtype=np.float32)
        gripper_state = float(observation["gripper_01"])
        _proprio = np.concatenate([ee_pos, ee_quat, ee_lin_vel, ee_ang_vel, [gripper_state]])  # noqa: F841

        camera_images = observation.get("rgb", {})
        images = {}
        for cam_name, img_data in camera_images.items():
            if img_data is not None:
                images[cam_name] = decode_image_array(img_data)

        _ = images

        # =====================================================
        # REPLACE THIS WITH YOUR POLICY
        # =====================================================

        twist = np.random.uniform(-1, 1, size=6)
        gripper = float(np.random.uniform(0, 1))

        # =====================================================

        return CanonicalAction(twist_ee_norm=twist.tolist(), gripper_01=gripper).model_dump(
            mode="python"
        )

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
    Example of integrating a PyTorch vision-based policy.

    This shows how you might load and use a trained neural network
    that processes camera images + proprioceptive observations.
    """

    def __init__(self, model_path: str = "policy.pt"):
        try:
            import torch
            import torchvision.transforms as transforms

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Image preprocessing
            self.transform = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            # Load your trained model
            if os.path.exists(model_path):
                self.model = torch.load(model_path, map_location=self.device)
                self.model.eval()
            else:
                self.model = None
                print(f"Warning: Model not found at {model_path}")

        except ImportError:
            print("PyTorch not available")
            self.model = None
            self.transform = None

    def preprocess_images(self, images: dict[str, np.ndarray]) -> "Any":
        """Preprocess camera images for the policy."""
        if self.transform is None or not images:
            return None

        import torch

        processed = []
        for cam_name in sorted(images.keys()):
            img = images[cam_name]
            img_tensor = self.transform(img)
            processed.append(img_tensor)

        if processed:
            return torch.stack(processed).unsqueeze(0).to(self.device)
        return None

    def __call__(self, proprio: np.ndarray, images: dict[str, np.ndarray]) -> np.ndarray:
        if self.model is None:
            return np.zeros(4)

        import torch

        with torch.no_grad():
            # Proprioceptive input
            proprio_tensor = torch.FloatTensor(proprio).unsqueeze(0).to(self.device)

            # Image input
            images_tensor = self.preprocess_images(images)

            # Forward pass
            if images_tensor is not None:
                action = self.model(proprio_tensor, images_tensor).cpu().numpy()[0]
            else:
                action = self.model(proprio_tensor).cpu().numpy()[0]

        return action


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
