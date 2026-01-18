"""DeepMind Control Suite environment wrapper."""

from typing import Any

import numpy as np

from robo.environments.base import RoboticsEnvironment, TaskConfig
from robo.environments.procedural import ProceduralTaskGenerator

# Lazy import
_dm_control = None


def _get_dm_control():
    global _dm_control
    if _dm_control is None:
        from dm_control import suite

        _dm_control = suite
    return _dm_control


class DMControlEnvironment(RoboticsEnvironment):
    """
    DeepMind Control Suite environment.

    Provides locomotion and control tasks including walking, running,
    balancing, and complex humanoid movements.

    Reference: https://github.com/google-deepmind/dm_control
    """

    # Available domain/task combinations
    AVAILABLE_TASKS = {
        "walker": ["stand", "walk", "run"],
        "cheetah": ["run"],
        "hopper": ["stand", "hop"],
        "humanoid": ["stand", "walk", "run"],
        "quadruped": ["walk", "run"],
        "swimmer": ["swimmer6", "swimmer15"],
        "pendulum": ["swingup"],
        "acrobot": ["swingup", "swingup_sparse"],
        "cartpole": ["balance", "balance_sparse", "swingup", "swingup_sparse"],
        "ball_in_cup": ["catch"],
        "finger": ["spin", "turn_easy", "turn_hard"],
        "reacher": ["easy", "hard"],
    }

    def __init__(self, domain: str, task: str):
        """
        Initialize DM Control environment.

        Args:
            domain: Domain name (e.g., 'walker', 'cheetah')
            task: Task name (e.g., 'walk', 'run')
        """
        if domain not in self.AVAILABLE_TASKS:
            raise ValueError(
                f"Unknown domain: {domain}. Available: {list(self.AVAILABLE_TASKS.keys())}"
            )
        if task not in self.AVAILABLE_TASKS[domain]:
            raise ValueError(
                f"Unknown task '{task}' for domain '{domain}'. "
                f"Available: {self.AVAILABLE_TASKS[domain]}"
            )

        self._domain = domain
        self._task = task
        self._env = None
        self._current_task_config: TaskConfig | None = None
        self._episode_reward = 0.0
        self._timestep = None

        # Procedural generator for locomotion tasks
        self._proc_gen = ProceduralTaskGenerator(
            env_id=f"dm_control/{domain}-{task}",
            position_ranges={},  # Locomotion doesn't have object positions
            physics_ranges={
                "joint_damping_scale": (0.8, 1.2),
                "actuator_strength_scale": (0.9, 1.1),
                "body_mass_scale": (0.9, 1.1),
            },
            domain_config={
                "wind_x": {"type": "uniform", "range": (-0.5, 0.5)},
                "wind_y": {"type": "uniform", "range": (-0.5, 0.5)},
            },
        )

    def _ensure_env(self) -> None:
        """Lazy initialization of DM Control environment."""
        if self._env is None:
            suite = _get_dm_control()
            self._env = suite.load(self._domain, self._task)

    @property
    def env_name(self) -> str:
        return "dm_control"

    @property
    def task_name(self) -> str:
        return f"{self._domain}-{self._task}"

    @property
    def observation_shape(self) -> tuple[int, ...]:
        self._ensure_env()
        # Flatten observation dict into single array
        obs_spec = self._env.observation_spec()
        total_dim = sum(int(np.prod(spec.shape)) for spec in obs_spec.values())
        return (total_dim,)

    @property
    def action_shape(self) -> tuple[int, ...]:
        self._ensure_env()
        return (self._env.action_spec().shape[0],)

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        self._ensure_env()
        action_spec = self._env.action_spec()
        return (
            action_spec.minimum.astype(np.float32),
            action_spec.maximum.astype(np.float32),
        )

    def _flatten_obs(self, observation: dict) -> np.ndarray:
        """Flatten observation dict to array."""
        arrays = []
        for key in sorted(observation.keys()):
            arr = np.asarray(observation[key], dtype=np.float32)
            arrays.append(arr.flatten())
        return np.concatenate(arrays)

    def generate_task(self, seed: int) -> TaskConfig:
        """Generate procedural task configuration."""
        proc_params = self._proc_gen.generate(seed=seed)

        return TaskConfig(
            env_name=self.env_name,
            task_name=self.task_name,
            seed=seed,
            object_positions=np.zeros(3),  # Not used for locomotion
            target_positions=np.zeros(3),
            physics_params=proc_params["physics_params"],
            domain_randomization=proc_params["domain_randomization"],
        )

    def reset(self, task_config: TaskConfig) -> np.ndarray:
        """Reset environment with task configuration."""
        self._ensure_env()
        self._current_task_config = task_config
        self._episode_reward = 0.0

        # Reset with random seed
        np.random.seed(task_config.seed)
        self._timestep = self._env.reset()

        # Apply physics randomization
        self._apply_physics_randomization(task_config.physics_params)

        return self._flatten_obs(self._timestep.observation)

    def _apply_physics_randomization(self, physics_params: dict[str, float]) -> None:
        """Apply physics parameter randomization."""
        if not physics_params:
            return

        try:
            physics = self._env.physics

            # Joint damping
            if "joint_damping_scale" in physics_params:
                scale = physics_params["joint_damping_scale"]
                physics.model.dof_damping[:] *= scale

            # Actuator strength
            if "actuator_strength_scale" in physics_params:
                scale = physics_params["actuator_strength_scale"]
                physics.model.actuator_gear[:] *= scale

            # Body mass
            if "body_mass_scale" in physics_params:
                scale = physics_params["body_mass_scale"]
                physics.model.body_mass[:] *= scale

        except (AttributeError, TypeError):
            pass

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
        """Execute action in environment."""
        self._ensure_env()

        # Clip action
        action_spec = self._env.action_spec()
        action = np.clip(action, action_spec.minimum, action_spec.maximum)

        self._timestep = self._env.step(action)
        reward = self._timestep.reward or 0.0
        self._episode_reward += reward

        done = self._timestep.last()
        obs = self._flatten_obs(self._timestep.observation)

        return obs, float(reward), done, {}

    def get_success(self) -> bool:
        """
        Check if task was completed successfully.

        For locomotion tasks, success is typically defined as
        achieving high cumulative reward.
        """
        # DM Control uses reward ∈ [0, 1], so success threshold varies by task
        success_thresholds = {
            "walker-walk": 800,
            "walker-run": 600,
            "cheetah-run": 700,
            "humanoid-walk": 600,
            "humanoid-run": 400,
        }
        threshold = success_thresholds.get(self.task_name, 500)
        return self._episode_reward >= threshold

    def close(self) -> None:
        """Clean up environment."""
        if self._env is not None:
            self._env.close()
            self._env = None
