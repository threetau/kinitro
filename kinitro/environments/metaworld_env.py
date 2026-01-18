"""MetaWorld manipulation environment wrapper with vision-based observations.

This wrapper provides a limited observation space to prevent overfitting:
- Proprioceptive: End-effector XYZ position (3) + gripper state (1) = 4 values
- Visual: RGB camera views from corner cameras

The full state (object positions, etc.) is NOT exposed to miners,
forcing them to learn from visual observations.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from kinitro.environments.base import RoboticsEnvironment, TaskConfig
from kinitro.environments.procedural import ProceduralTaskGenerator

# Lazy import to avoid import errors if metaworld not installed
_metaworld = None
_available_tasks: list[str] | None = None


def _get_metaworld():
    global _metaworld
    if _metaworld is None:
        import metaworld

        _metaworld = metaworld
    return _metaworld


def _get_available_tasks() -> list[str]:
    """Get list of available MetaWorld V3 tasks."""
    global _available_tasks
    if _available_tasks is None:
        metaworld = _get_metaworld()
        _available_tasks = list(metaworld._env_dict.ALL_V3_ENVIRONMENTS.keys())
    return _available_tasks


@dataclass
class MetaWorldObservation:
    """
    Structured observation from MetaWorld environment.

    This is what miners receive - proprioceptive info + camera views.
    Object positions and other task-specific state are NOT included.
    """

    # Proprioceptive observations (what the robot "feels")
    end_effector_pos: np.ndarray  # Shape: (3,) - XYZ position
    gripper_state: float  # Scalar: 0=closed, 1=open

    # Visual observations (what the robot "sees")
    camera_views: dict[str, np.ndarray]  # camera_name -> (H, W, 3) RGB image

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "end_effector_pos": self.end_effector_pos.tolist(),
            "gripper_state": self.gripper_state,
            "camera_views": {name: img.tolist() for name, img in self.camera_views.items()},
        }

    def to_flat_array(self, include_images: bool = False) -> np.ndarray:
        """
        Convert to flat array for simple policies.

        Args:
            include_images: If True, flatten and include camera images

        Returns:
            Flat numpy array
        """
        proprio = np.concatenate(
            [
                self.end_effector_pos,
                np.array([self.gripper_state]),
            ]
        )

        if not include_images:
            return proprio.astype(np.float32)

        # Flatten and normalize images
        image_arrays = []
        for img in self.camera_views.values():
            # Normalize to [0, 1] and flatten
            img_normalized = img.astype(np.float32) / 255.0
            image_arrays.append(img_normalized.flatten())

        if image_arrays:
            images_flat = np.concatenate(image_arrays)
            return np.concatenate([proprio, images_flat]).astype(np.float32)

        return proprio.astype(np.float32)


class MetaWorldEnvironment(RoboticsEnvironment):
    """
    MetaWorld manipulation environment with vision-based observations.

    This wrapper provides LIMITED observations to prevent overfitting:
    - Proprioceptive: End-effector position + gripper state only
    - Visual: RGB camera views from corner cameras

    Object positions, orientations, and other task state are NOT exposed.
    Miners must learn to infer this from visual observations.

    Reference: https://github.com/Farama-Foundation/Metaworld
    """

    # Core tasks we use for evaluation (V3 versions)
    CORE_TASKS = [
        "reach-v3",
        "push-v3",
        "pick-place-v3",
        "door-open-v3",
        "drawer-open-v3",
        "drawer-close-v3",
        "button-press-topdown-v3",
        "peg-insert-side-v3",
    ]

    # Available camera views
    CAMERA_NAMES = ["corner", "corner2"]  # Two corner views for depth perception

    # Image dimensions
    IMAGE_WIDTH = 84
    IMAGE_HEIGHT = 84

    def __init__(
        self,
        task_name: str,
        use_camera: bool = True,
        camera_names: list[str] | None = None,
        image_size: tuple[int, int] = (84, 84),
    ):
        """
        Initialize MetaWorld environment.

        Args:
            task_name: Name of the task (e.g., 'pick-place-v3')
            use_camera: Whether to include camera observations
            camera_names: Which cameras to use (default: ['corner', 'corner2'])
            image_size: (width, height) for camera images
        """
        available = _get_available_tasks()
        if task_name not in available:
            raise ValueError(
                f"Unknown task: {task_name}. Available tasks include: {available[:10]}..."
            )

        self._task_name = task_name
        self._use_camera = use_camera
        self._camera_names = camera_names or self.CAMERA_NAMES
        self._image_width, self._image_height = image_size

        self._env = None
        self._camera_envs: dict[str, Any] = {}  # Separate env instances for each camera
        self._ml1 = None
        self._current_task_config: TaskConfig | None = None
        self._current_task = None
        self._episode_reward = 0.0
        self._episode_success = False

        # Procedural generator with MetaWorld-appropriate ranges
        self._proc_gen = ProceduralTaskGenerator(
            env_id=f"metaworld/{task_name}",
            position_ranges={
                "object": np.array([0.1, 0.1, 0.02]),
                "target": np.array([0.1, 0.1, 0.05]),
            },
            physics_ranges={
                "friction": (0.8, 1.2),
                "damping": (0.9, 1.1),
            },
        )

    def _ensure_env(self) -> None:
        """Lazy initialization of MetaWorld environment."""
        if self._env is None:
            metaworld = _get_metaworld()
            self._ml1 = metaworld.ML1(self._task_name)
            env_cls = self._ml1.train_classes[self._task_name]

            # Main environment (no rendering, for state/physics)
            self._env = env_cls()

            # Camera environments (with rendering)
            if self._use_camera:
                for cam_name in self._camera_names:
                    try:
                        cam_env = env_cls(
                            render_mode="rgb_array",
                            camera_name=cam_name,
                            width=self._image_width,
                            height=self._image_height,
                        )
                        self._camera_envs[cam_name] = cam_env
                    except Exception as e:
                        # Camera may not be available in headless mode
                        import warnings

                        warnings.warn(f"Could not create camera '{cam_name}': {e}")

    @property
    def env_name(self) -> str:
        return "metaworld"

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """
        Shape of proprioceptive observation only.

        For full observation including images, use get_observation().
        """
        # 3 (end effector XYZ) + 1 (gripper state) = 4
        return (4,)

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """Shape of a single camera image: (H, W, C)."""
        return (self._image_height, self._image_width, 3)

    @property
    def num_cameras(self) -> int:
        """Number of camera views."""
        return len(self._camera_names) if self._use_camera else 0

    @property
    def action_shape(self) -> tuple[int, ...]:
        self._ensure_env()
        return (self._env.action_space.shape[0],)

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_env()
        return (
            self._env.action_space.low.astype(np.float32),
            self._env.action_space.high.astype(np.float32),
        )

    def _extract_proprioceptive_obs(self, full_obs: np.ndarray) -> np.ndarray:
        """
        Extract only proprioceptive observations from full state.

        From MetaWorld docs:
        - Indices 0:3 = end-effector XYZ position
        - Index 3 = gripper open/close state

        We do NOT include:
        - Indices 4:7 = object 1 position (would allow cheating)
        - Indices 7:11 = object 1 quaternion
        - Indices 11:14 = object 2 position
        - Indices 14:18 = object 2 quaternion
        - Any other task-specific state
        """
        return full_obs[0:4].astype(np.float32)

    def _get_camera_images(self) -> dict[str, np.ndarray]:
        """Render images from all configured cameras."""
        images = {}

        for cam_name, cam_env in self._camera_envs.items():
            try:
                # Sync camera env state with main env
                if self._current_task is not None:
                    cam_env.set_task(self._current_task)

                # Copy the physics state from main env to camera env
                if hasattr(self._env, "unwrapped") and hasattr(cam_env, "unwrapped"):
                    main_data = self._env.unwrapped.data
                    cam_data = cam_env.unwrapped.data

                    # Copy qpos and qvel
                    cam_data.qpos[:] = main_data.qpos[:]
                    cam_data.qvel[:] = main_data.qvel[:]

                    # Forward kinematics to update derived quantities
                    import mujoco

                    mujoco.mj_forward(cam_env.unwrapped.model, cam_data)

                # Render
                img = cam_env.render()
                if img is not None:
                    images[cam_name] = img
            except Exception:
                # Rendering may fail in headless environments
                pass

        return images

    def get_observation(self) -> MetaWorldObservation:
        """
        Get structured observation with proprioceptive and visual data.

        Returns:
            MetaWorldObservation with end-effector state and camera views
        """
        self._ensure_env()

        # Get current state from environment
        full_obs = self._env.unwrapped._get_obs()

        proprio = self._extract_proprioceptive_obs(full_obs)
        camera_views = self._get_camera_images() if self._use_camera else {}

        return MetaWorldObservation(
            end_effector_pos=proprio[0:3],
            gripper_state=float(proprio[3]),
            camera_views=camera_views,
        )

    def generate_task(self, seed: int) -> TaskConfig:
        """Generate procedural task configuration."""
        self._ensure_env()

        rng = np.random.default_rng(seed)

        # Sample a base task from MetaWorld's task distribution
        task_idx = rng.integers(0, len(self._ml1.train_tasks))

        # Apply procedural randomization
        proc_params = self._proc_gen.generate(seed=seed)

        return TaskConfig(
            env_name=self.env_name,
            task_name=self._task_name,
            seed=seed,
            object_positions=proc_params["object_positions"],
            target_positions=proc_params["target_positions"],
            physics_params=proc_params["physics_params"],
            domain_randomization={
                **proc_params["domain_randomization"],
                "task_idx": int(task_idx),
            },
        )

    def reset(self, task_config: TaskConfig) -> np.ndarray:
        """
        Reset environment with task configuration.

        Returns:
            Proprioceptive observation only (4D: end-effector XYZ + gripper)
            Use get_observation() for full observation with camera views.
        """
        self._ensure_env()
        self._current_task_config = task_config
        self._episode_reward = 0.0
        self._episode_success = False

        # Get task index from domain randomization or use seed-based selection
        task_idx = task_config.domain_randomization.get("task_idx")
        if task_idx is None:
            rng = np.random.default_rng(task_config.seed)
            task_idx = rng.integers(0, len(self._ml1.train_tasks))

        self._current_task = self._ml1.train_tasks[task_idx]
        self._env.set_task(self._current_task)

        # Reset main environment
        full_obs, _ = self._env.reset(seed=task_config.seed)

        # Reset camera environments with same task
        for cam_env in self._camera_envs.values():
            cam_env.set_task(self._current_task)
            cam_env.reset(seed=task_config.seed)

        # Apply physics randomization (if supported)
        self._apply_physics_randomization(task_config.physics_params)

        # Return only proprioceptive observation
        return self._extract_proprioceptive_obs(full_obs)

    def _apply_physics_randomization(self, physics_params: dict[str, float]) -> None:
        """Apply physics parameter randomization to the MuJoCo model."""
        if not physics_params:
            return

        try:
            model = self._env.unwrapped.model

            # Friction randomization
            if "friction" in physics_params:
                friction_scale = physics_params["friction"]
                model.geom_friction[:] *= friction_scale

            # Damping randomization
            if "damping" in physics_params:
                damping_scale = physics_params["damping"]
                model.dof_damping[:] *= damping_scale

        except (AttributeError, TypeError):
            pass

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """
        Execute action in environment.

        Returns:
            Tuple of (proprioceptive_obs, reward, done, info)
            proprioceptive_obs is 4D: end-effector XYZ + gripper state
            Use get_observation() after step for full observation with cameras.
        """
        self._ensure_env()

        # Clip action to valid range
        action = np.clip(action, self._env.action_space.low, self._env.action_space.high)

        full_obs, reward, terminated, truncated, info = self._env.step(action)

        self._episode_reward += reward
        if info.get("success", False):
            self._episode_success = True

        done = terminated or truncated

        # Return only proprioceptive observation
        proprio_obs = self._extract_proprioceptive_obs(full_obs)

        return proprio_obs, float(reward), done, info

    def get_success(self) -> bool:
        """Check if task was completed successfully."""
        return self._episode_success

    def close(self) -> None:
        """Clean up environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

        for cam_env in self._camera_envs.values():
            try:
                cam_env.close()
            except Exception:
                pass
        self._camera_envs.clear()

    def render(self, camera_name: str = "corner") -> np.ndarray | None:
        """
        Render environment frame from specified camera.

        Args:
            camera_name: Which camera to use

        Returns:
            RGB image array or None if rendering fails
        """
        if camera_name in self._camera_envs:
            try:
                # Sync state and render
                cam_env = self._camera_envs[camera_name]
                if hasattr(self._env, "unwrapped") and hasattr(cam_env, "unwrapped"):
                    main_data = self._env.unwrapped.data
                    cam_data = cam_env.unwrapped.data
                    cam_data.qpos[:] = main_data.qpos[:]
                    cam_data.qvel[:] = main_data.qvel[:]
                    import mujoco

                    mujoco.mj_forward(cam_env.unwrapped.model, cam_data)
                return cam_env.render()
            except Exception:
                return None
        return None
