from typing import Any

import numpy as np
import structlog

from kinitro.environments.base import RoboticsEnvironment, TaskConfig
from kinitro.environments.procedural import ProceduralTaskGenerator
from kinitro.rl_interface import (
    CanonicalAction,
    CanonicalObservation,
    coerce_action,
    normalize_quaternion,
)

# Lazy import to avoid import errors if metaworld not installed
_metaworld = None
_available_tasks: list[str] | None = None

logger = structlog.get_logger()


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


class MetaWorldEnvironment(RoboticsEnvironment):
    """
    MetaWorld manipulation environment with canonical observations.

    This wrapper provides canonical observations:
    - Proprioceptive: ee_pos_m, ee_quat_xyzw, ee_lin_vel_mps, ee_ang_vel_rps, gripper_01
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
        control_dt: float | None = None,
        v_max: float = 0.05,
        w_max: float = 1.0,
        action_repeat: int = 1,
        ee_site_name: str = "end_effector",
        action_format: str = "auto",
        warn_on_orientation_mismatch: bool = True,
    ):
        """
        Initialize MetaWorld environment.

        Args:
            task_name: Name of the task (e.g., 'pick-place-v3')
            use_camera: Whether to include camera observations
            camera_names: Which cameras to use (default: ['corner', 'corner2'])
            image_size: (width, height) for camera images
            control_dt: Control timestep (seconds). If None, derive from env.
            v_max: Max linear velocity (m/s) for twist scaling
            w_max: Max angular velocity (rad/s) for twist scaling
            action_repeat: Number of simulation steps per action
            ee_site_name: MuJoCo site name for end-effector pose
            action_format: Override action format (auto, xyz_gripper, xyz_quat, xyz_quat_gripper).
            warn_on_orientation_mismatch: Warn once when twist/gripper inputs are ignored.
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
        self._prev_ee_pos: np.ndarray | None = None
        self._prev_ee_quat: np.ndarray | None = None
        self._prev_gripper: float | None = None
        self._control_dt = control_dt
        self._v_max = v_max
        self._w_max = w_max
        self._action_repeat = max(1, int(action_repeat))
        self._ee_site_name = ee_site_name
        self._action_format = action_format
        self._warn_on_orientation_mismatch = warn_on_orientation_mismatch
        self._warned_keys: set[str] = set()
        self._resolved_action_format: str | None = None
        self._env_id = f"metaworld/{task_name}"

        # Procedural generator with MetaWorld-appropriate ranges
        self._proc_gen = ProceduralTaskGenerator(
            env_id=self._env_id,
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
            self._resolve_action_format()

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
                        logger.warning(
                            "metaworld_camera_unavailable",
                            camera_name=cam_name,
                            error=str(e),
                        )

    @property
    def env_name(self) -> str:
        return "metaworld"

    @property
    def task_name(self) -> str:
        return self._task_name

    @property
    def observation_shape(self) -> tuple[int, ...]:
        """Shape of canonical proprioceptive observation."""
        # 3 + 4 + 3 + 3 + 1 = 14 canonical proprio values
        return (14,)

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
        if self._resolved_action_format is None:
            self._resolve_action_format()
        if self._resolved_action_format == "xyz_gripper":
            return (4,)
        if self._resolved_action_format == "xyz_quat":
            return (7,)
        if self._resolved_action_format == "xyz_quat_gripper":
            return (8,)
        return (7,)

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.full(self.action_shape, -1.0, dtype=np.float32)
        high = np.full(self.action_shape, 1.0, dtype=np.float32)
        return (low, high)

    def _warn_once(self, key: str, message: str, **kwargs: Any) -> None:
        if key in self._warned_keys:
            return
        self._warned_keys.add(key)
        logger.warning(message, **kwargs)

    def _resolve_action_format(self) -> None:
        if self._env is None:
            return

        action_dim = int(self._env.action_space.shape[0])
        valid_formats = {"auto", "xyz_gripper", "xyz_quat", "xyz_quat_gripper"}
        if self._action_format not in valid_formats:
            raise ValueError(
                f"Invalid action_format '{self._action_format}'. "
                f"Expected one of {sorted(valid_formats)}."
            )

        if self._action_format == "auto":
            if action_dim == 4:
                self._resolved_action_format = "xyz_gripper"
            elif action_dim == 7:
                self._resolved_action_format = "xyz_quat"
            else:
                raise ValueError(
                    f"Unsupported action dimension {action_dim} for MetaWorld. "
                    "Set action_format explicitly to proceed."
                )
        else:
            self._resolved_action_format = self._action_format

            expected_dim = {
                "xyz_gripper": 4,
                "xyz_quat": 7,
                "xyz_quat_gripper": 8,
            }[self._resolved_action_format]
            if action_dim != expected_dim:
                raise ValueError(
                    "action_format mismatch: requested "
                    f"{self._resolved_action_format} but env action_dim={action_dim}."
                )

        self._warned_keys.clear()

    def _extract_proprioceptive_obs(self, full_obs: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Extract end-effector position and gripper state from full state.

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
        proprio = full_obs[0:4].astype(np.float32)
        return proprio[0:3], float(proprio[3])

    def _get_camera_images(self) -> dict[str, np.ndarray]:
        """Render images from all configured cameras."""
        import mujoco

        images = {}

        for cam_name, cam_env in self._camera_envs.items():
            try:
                # Copy the physics state from main env to camera env
                if hasattr(self._env, "unwrapped") and hasattr(cam_env, "unwrapped"):
                    main_data = self._env.unwrapped.data
                    cam_data = cam_env.unwrapped.data

                    # Copy qpos and qvel
                    cam_data.qpos[:] = main_data.qpos[:]
                    cam_data.qvel[:] = main_data.qvel[:]

                    # Forward kinematics to update derived quantities
                    mujoco.mj_forward(cam_env.unwrapped.model, cam_data)

                # Render
                img = cam_env.render()
                if img is not None:
                    # Flip vertically - MuJoCo renders with origin at bottom-left
                    images[cam_name] = np.flipud(img)
            except Exception:
                # Rendering may fail in headless environments
                pass

        return images

    def _get_ee_quaternion(self) -> np.ndarray:
        """Return end-effector quaternion in XYZW order."""
        self._ensure_env()
        try:
            data = self._env.unwrapped.data
            ee_site = self._env.unwrapped.model.site_name2id(self._ee_site_name)
            quat_wxyz = data.site_xquat[ee_site]
            quat_xyzw = np.array([quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]])
            return normalize_quaternion(quat_xyzw.astype(np.float32))
        except Exception:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)

    def _quat_multiply(self, a_xyzw: np.ndarray, b_xyzw: np.ndarray) -> np.ndarray:
        ax, ay, az, aw = a_xyzw
        bx, by, bz, bw = b_xyzw
        return np.array(
            [
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
                aw * bw - ax * bx - ay * by - az * bz,
            ],
            dtype=np.float32,
        )

    def _quat_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        half = 0.5 * angle
        sin_half = np.sin(half)
        return np.array(
            [axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half, np.cos(half)],
            dtype=np.float32,
        )

    def _orientation_delta(self, omega: np.ndarray, dt: float) -> np.ndarray:
        angle = float(np.linalg.norm(omega) * dt)
        if angle <= 0.0:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        axis = omega / (np.linalg.norm(omega) + 1e-8)
        return self._quat_from_axis_angle(axis.astype(np.float32), angle)

    def _build_observation(self, full_obs: np.ndarray) -> CanonicalObservation:
        ee_pos, gripper_state = self._extract_proprioceptive_obs(full_obs)
        ee_quat = self._get_ee_quaternion()

        ee_lin_vel = None
        ee_ang_vel = None

        try:
            data = self._env.unwrapped.data
            ee_site = self._env.unwrapped.model.site_name2id(self._ee_site_name)
            ee_lin_vel = np.array(data.site_xvelp[ee_site], dtype=np.float32)
            ee_ang_vel = np.array(data.site_xvelr[ee_site], dtype=np.float32)
        except Exception:
            pass

        if ee_lin_vel is None:
            if self._prev_ee_pos is None:
                ee_lin_vel = np.zeros(3, dtype=np.float32)
            else:
                ee_lin_vel = (ee_pos - self._prev_ee_pos) / self._control_dt

        if ee_ang_vel is None:
            if self._prev_ee_quat is None:
                ee_ang_vel = np.zeros(3, dtype=np.float32)
            else:
                delta = ee_quat - self._prev_ee_quat
                ee_ang_vel = delta[:3] / self._control_dt

        gripper_norm = float(np.clip((gripper_state + 1.0) / 2.0, 0.0, 1.0))

        self._prev_ee_pos = ee_pos.copy()
        self._prev_ee_quat = ee_quat.copy()
        self._prev_gripper = gripper_norm

        camera_views = self._get_camera_images() if self._use_camera else {}

        if self._use_camera and not camera_views and self._warn_on_orientation_mismatch:
            self._warn_once(
                f"camera_missing:{self._env_id}",
                "MetaWorld cameras unavailable; rgb observations are empty.",
                env_id=self._env_id,
            )

        return CanonicalObservation(
            ee_pos_m=ee_pos.tolist(),
            ee_quat_xyzw=ee_quat.tolist(),
            ee_lin_vel_mps=ee_lin_vel.tolist(),
            ee_ang_vel_rps=ee_ang_vel.tolist(),
            gripper_01=gripper_norm,
            rgb={name: img.tolist() for name, img in camera_views.items()},
            depth=None,
            cam_intrinsics_K=None,
            cam_extrinsics_T_world_cam=None,
        )

    def get_observation(self) -> CanonicalObservation:
        """
        Get canonical observation with proprioceptive and visual data.

        Returns:
            CanonicalObservation with end-effector state and camera views
        """
        self._ensure_env()
        full_obs = self._env.unwrapped._get_obs()
        return self._build_observation(full_obs)

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

    def reset(self, task_config: TaskConfig) -> CanonicalObservation:
        """
        Reset environment with task configuration.

        Returns:
            Canonical observation
        """
        self._ensure_env()
        self._current_task_config = task_config
        self._episode_reward = 0.0
        self._episode_success = False
        self._prev_ee_pos = None
        self._prev_ee_quat = None
        self._prev_gripper = None
        if self._control_dt is None:
            self._control_dt = float(self._env.dt) if hasattr(self._env, "dt") else 0.05

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

        return self._build_observation(full_obs)

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

    def step(
        self,
        action: CanonicalAction | dict[str, Any] | np.ndarray,
    ) -> tuple[CanonicalObservation, float, bool, dict[str, Any]]:
        """
        Execute action in environment.

        Returns:
            Tuple of (canonical observation, reward, done, info)
        """
        self._ensure_env()

        canonical_action = coerce_action(action)
        twist = np.clip(np.array(canonical_action.twist_ee_norm, dtype=np.float32), -1.0, 1.0)
        gripper = float(np.clip(canonical_action.gripper_01, 0.0, 1.0))

        if self._resolved_action_format is None:
            self._resolve_action_format()
        action_format = self._resolved_action_format
        if action_format is None:
            raise ValueError("Action format resolution failed for MetaWorld environment.")

        v = twist[:3] * self._v_max
        w = twist[3:6] * self._w_max
        delta_pos = v * self._control_dt

        if action_format == "xyz_gripper":
            if np.linalg.norm(w) > 0.0 and self._warn_on_orientation_mismatch:
                self._warn_once(
                    f"orientation_ignored:{self._env_id}",
                    "MetaWorld action format ignores angular twist (xyz+gripper).",
                    env_id=self._env_id,
                    action_format=action_format,
                )
            mw_action = np.zeros(4, dtype=np.float32)
            mw_action[0:3] = delta_pos
            mw_action[3] = gripper * 2.0 - 1.0
        elif action_format == "xyz_quat":
            if gripper != 0.0 and self._warn_on_orientation_mismatch:
                self._warn_once(
                    f"gripper_ignored:{self._env_id}",
                    "MetaWorld action format ignores gripper control (xyz+quat).",
                    env_id=self._env_id,
                    action_format=action_format,
                )
            delta_quat = self._orientation_delta(w, self._control_dt)
            current_quat = self._get_ee_quaternion()
            target_quat = normalize_quaternion(self._quat_multiply(delta_quat, current_quat))
            mw_action = np.zeros(7, dtype=np.float32)
            mw_action[0:3] = delta_pos
            mw_action[3:7] = target_quat
        elif action_format == "xyz_quat_gripper":
            delta_quat = self._orientation_delta(w, self._control_dt)
            current_quat = self._get_ee_quaternion()
            target_quat = normalize_quaternion(self._quat_multiply(delta_quat, current_quat))
            mw_action = np.zeros(8, dtype=np.float32)
            mw_action[0:3] = delta_pos
            mw_action[3:7] = target_quat
            mw_action[7] = gripper * 2.0 - 1.0
        else:
            raise ValueError(f"Unsupported action_format '{action_format}'.")

        mw_action = np.clip(mw_action, self._env.action_space.low, self._env.action_space.high)

        total_reward = 0.0
        info = {}
        terminated = False
        truncated = False
        full_obs = None

        for _ in range(self._action_repeat):
            full_obs, reward, terminated, truncated, info = self._env.step(mw_action)
            total_reward += reward
            if terminated or truncated:
                break

        self._episode_reward += total_reward
        if info.get("success", False):
            self._episode_success = True

        done = terminated or truncated

        return self._build_observation(full_obs), float(total_reward), done, info

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
