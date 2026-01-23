"""
Miner Policy Template for Kinitro

This is the template for creating a policy that can be evaluated
by Kinitro validators.

Your policy must implement the RobotActor class with:
- reset(task_config): Called at the start of each episode
- act(observation): Called each timestep to get action
- cleanup(): Called when evaluation is complete

The policy will be run inside a container and queried by validators
across multiple MetaWorld robotics environments.

IMPORTANT: Observations are LIMITED to prevent overfitting:
- Proprioceptive: End-effector XYZ position + gripper state (4 values)
- Visual: RGB camera images from corner cameras (base64 encoded)
- Object positions are NOT exposed - you must learn from visual input!
"""

import base64
import io
import os
from typing import Any

import numpy as np


def decode_image(b64_string: str) -> np.ndarray:
    """Decode base64 PNG string to numpy array."""
    from PIL import Image

    buffer = io.BytesIO(base64.b64decode(b64_string))
    img = Image.open(buffer)
    return np.array(img)


class RobotActor:
    """
    Your robotics policy implementation.

    This class is instantiated once when the container starts,
    then reset() is called for each new episode, and act() is
    called for each timestep.

    Observation format (dict):
        - end_effector_pos: [x, y, z] - Robot end-effector position
        - gripper_state: float - Gripper open/close (0=closed, 1=open)
        - camera_images: dict[str, str] - Camera views as base64 PNGs
            - "corner": First corner camera view (84x84 RGB)
            - "corner2": Second corner camera view (84x84 RGB)

    Action format (list[float]):
        - MetaWorld: 4 values [dx, dy, dz, gripper] in range [-1, 1]
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

        # Track action dimensions per environment
        self._action_dims = {
            "metaworld": 4,
        }

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

    async def act(self, observation: dict[str, Any]) -> list[float]:
        """
        Get action for current observation.

        This is called every timestep. You have ~100ms to respond.

        Args:
            observation: Dict containing:
                - end_effector_pos: [x, y, z] - Robot end-effector position
                - gripper_state: float - Gripper open/close state
                - camera_images: dict[str, str] - Camera views as base64 PNGs

        Returns:
            action: Flat list of floats (joint positions/velocities)
                   Values should be in [-1, 1] range
        """
        # Extract proprioceptive observations
        end_effector_pos = np.array(observation["end_effector_pos"], dtype=np.float32)
        gripper_state = float(observation["gripper_state"])
        _proprio = np.concatenate([end_effector_pos, [gripper_state]])  # noqa: F841

        # Extract camera images (if available)
        camera_images = observation.get("camera_images", {})
        images = {}
        for cam_name, b64_img in camera_images.items():
            if b64_img:
                images[cam_name] = decode_image(b64_img)

        # =====================================================
        # REPLACE THIS WITH YOUR POLICY
        # =====================================================

        # Determine action dimension based on environment
        action_dim = self._action_dims.get(self.current_env, 4)

        # Example: Random policy (replace with your trained policy!)
        # action = self.policy(proprio, images)
        action = np.random.uniform(-1, 1, size=action_dim)

        # =====================================================

        return action.tolist()

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
