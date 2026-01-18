"""ManiSkill2 dexterous manipulation environment wrapper."""

from typing import Any

import numpy as np

from robo.environments.base import RoboticsEnvironment, TaskConfig
from robo.environments.procedural import ProceduralTaskGenerator

# Lazy import
_mani_skill2 = None


def _get_mani_skill():
    global _mani_skill2
    if _mani_skill2 is None:
        import mani_skill2.envs  # noqa: F401
        import gymnasium as gym

        _mani_skill2 = gym
    return _mani_skill2


class ManiSkillEnvironment(RoboticsEnvironment):
    """
    ManiSkill2 dexterous manipulation environment.

    Provides high-fidelity manipulation tasks with soft-body simulation,
    articulated objects, and diverse robot embodiments.

    Reference: https://github.com/haosulab/ManiSkill2
    """

    # Available tasks
    AVAILABLE_TASKS = [
        # Rigid body tasks
        "PickCube-v1",
        "StackCube-v1",
        "PickSingleYCB-v1",
        "PickSingleEGAD-v1",
        "PickClutterYCB-v1",
        # Articulated object tasks
        "OpenCabinetDoor-v1",
        "OpenCabinetDrawer-v1",
        "PushChair-v2",
        "MoveBucket-v2",
        # Assembly tasks
        "PegInsertionSide-v1",
        "PlugCharger-v1",
        # Soft body tasks
        "Hang-v1",
        "Pour-v1",
        "Excavate-v1",
    ]

    def __init__(self, task_name: str, obs_mode: str = "state"):
        """
        Initialize ManiSkill environment.

        Args:
            task_name: Name of the task (e.g., 'PickCube-v1')
            obs_mode: Observation mode ('state', 'rgbd', 'pointcloud')
        """
        if task_name not in self.AVAILABLE_TASKS:
            raise ValueError(f"Unknown task: {task_name}. Available: {self.AVAILABLE_TASKS}")

        self._task_name = task_name
        self._obs_mode = obs_mode
        self._env = None
        self._current_task_config: TaskConfig | None = None
        self._episode_reward = 0.0
        self._episode_success = False
        self._obs_shape: tuple[int, ...] | None = None
        self._action_shape: tuple[int, ...] | None = None

        # Procedural generator
        self._proc_gen = ProceduralTaskGenerator(
            env_id=f"maniskill/{task_name}",
            position_ranges={
                "object": np.array([0.15, 0.15, 0.05]),
                "target": np.array([0.15, 0.15, 0.1]),
            },
            physics_ranges={
                "friction": (0.5, 1.5),
                "density_scale": (0.8, 1.2),
            },
        )

    def _ensure_env(self) -> None:
        """Lazy initialization of ManiSkill environment."""
        if self._env is None:
            gym = _get_mani_skill()
            self._env = gym.make(
                self._task_name,
                obs_mode=self._obs_mode,
                control_mode="pd_ee_delta_pose",  # End-effector control
                render_mode="rgb_array",
            )
            # Cache shapes
            obs, _ = self._env.reset()
            self._obs_shape = self._flatten_obs(obs).shape
            self._action_shape = (self._env.action_space.shape[0],)

    def _flatten_obs(self, observation: dict | np.ndarray) -> np.ndarray:
        """Flatten observation to 1D array."""
        if isinstance(observation, np.ndarray):
            return observation.flatten().astype(np.float32)

        # Handle dict observations
        if isinstance(observation, dict):
            if "state" in observation:
                # State-based observation
                arrays = []
                state = observation["state"]
                if isinstance(state, dict):
                    for key in sorted(state.keys()):
                        arr = np.asarray(state[key], dtype=np.float32)
                        arrays.append(arr.flatten())
                    return np.concatenate(arrays)
                return np.asarray(state, dtype=np.float32).flatten()

            # Fallback: concatenate all arrays in dict
            arrays = []
            for key in sorted(observation.keys()):
                if isinstance(observation[key], np.ndarray):
                    arrays.append(observation[key].flatten())
            if arrays:
                return np.concatenate(arrays).astype(np.float32)

        raise ValueError(f"Cannot flatten observation of type {type(observation)}")

    @property
    def env_name(self) -> str:
        return "maniskill"

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def observation_shape(self) -> tuple[int, ...]:
        self._ensure_env()
        return self._obs_shape  # type: ignore

    @property
    def action_shape(self) -> tuple[int, ...]:
        self._ensure_env()
        return self._action_shape  # type: ignore

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_env()
        return (
            self._env.action_space.low.astype(np.float32),
            self._env.action_space.high.astype(np.float32),
        )

    def generate_task(self, seed: int) -> TaskConfig:
        """Generate procedural task configuration."""
        proc_params = self._proc_gen.generate(seed=seed)

        return TaskConfig(
            env_name=self.env_name,
            task_name=self._task_name,
            seed=seed,
            object_positions=proc_params["object_positions"],
            target_positions=proc_params["target_positions"],
            physics_params=proc_params["physics_params"],
            domain_randomization=proc_params["domain_randomization"],
        )

    def reset(self, task_config: TaskConfig) -> np.ndarray:
        """Reset environment with task configuration."""
        self._ensure_env()
        self._current_task_config = task_config
        self._episode_reward = 0.0
        self._episode_success = False

        # Reset with seed
        obs, _ = self._env.reset(seed=task_config.seed)

        # Apply physics randomization
        self._apply_physics_randomization(task_config.physics_params)

        return self._flatten_obs(obs)

    def _apply_physics_randomization(self, physics_params: dict[str, float]) -> None:
        """Apply physics parameter randomization."""
        if not physics_params:
            return

        # ManiSkill uses SAPIEN, physics modification is more complex
        # For now, we rely on the procedural seed for variation
        # Full physics randomization would require modifying SAPIEN actors
        pass

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute action in environment."""
        self._ensure_env()

        # Clip action
        action = np.clip(action, self._env.action_space.low, self._env.action_space.high)

        obs, reward, terminated, truncated, info = self._env.step(action)

        self._episode_reward += reward
        if info.get("success", False):
            self._episode_success = True

        done = terminated or truncated

        return self._flatten_obs(obs), float(reward), done, info

    def get_success(self) -> bool:
        """Check if task was completed successfully."""
        return self._episode_success

    def close(self) -> None:
        """Clean up environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def render(self) -> np.ndarray | None:
        """Render environment frame."""
        if self._env is not None:
            try:
                return self._env.render()
            except Exception:
                return None
        return None
