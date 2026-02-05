"""ManiSkill3 robotics environment with canonical observations."""

from typing import Any, cast

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

logger = structlog.get_logger()

# ManiSkill3 task mapping: internal name -> ManiSkill3 env ID
MANISKILL3_TASKS = {
    "pick-cube-v1": "PickCube-v1",
    "stack-cube-v1": "StackCube-v1",
    "peg-insertion-v1": "PegInsertionSide-v1",
    "push-cube-v1": "PushCube-v1",
    "pull-cube-v1": "PullCube-v1",
    "lift-cube-v1": "LiftCube-v1",
    "turn-faucet-v1": "TurnFaucet-v1",
    "open-cabinet-door-v1": "OpenCabinetDoor-v1",
}


class ManiSkill3Environment(RoboticsEnvironment):
    """
    ManiSkill3 manipulation environment with canonical observations.

    This wrapper provides canonical observations:
    - Proprioceptive: ee_pos_m, ee_quat_xyzw, ee_lin_vel_mps, ee_ang_vel_rps, gripper_01
    - Visual: RGB camera views

    Object positions, orientations, and other task state are NOT exposed.
    Miners must learn to infer this from visual observations.

    Reference: https://github.com/haosulab/ManiSkill
    """

    # Core tasks for evaluation
    CORE_TASKS = list(MANISKILL3_TASKS.keys())

    # Available camera views (ManiSkill3 default is base_camera only)
    CAMERA_NAMES = ["base_camera"]

    # Image dimensions
    IMAGE_WIDTH = 128
    IMAGE_HEIGHT = 128

    def __init__(
        self,
        task_name: str,
        use_camera: bool = True,
        camera_names: list[str] | None = None,
        image_size: tuple[int, int] = (128, 128),
        control_mode: str = "pd_ee_delta_pose",
        sim_backend: str = "auto",
        v_max: float = 0.1,
        w_max: float = 1.0,
    ):
        """
        Initialize ManiSkill3 environment.

        Args:
            task_name: Name of the task (e.g., 'pick-cube-v1')
            use_camera: Whether to include camera observations
            camera_names: Which cameras to use (default: ['base_camera', 'hand_camera'])
            image_size: (width, height) for camera images
            control_mode: ManiSkill3 control mode (default: pd_ee_delta_pose)
            sim_backend: Simulation backend ('auto', 'cpu', 'gpu')
            v_max: Max linear velocity (m/s) for twist scaling
            w_max: Max angular velocity (rad/s) for twist scaling
        """
        if task_name not in MANISKILL3_TASKS:
            raise ValueError(
                f"Unknown task: {task_name}. Available tasks: {list(MANISKILL3_TASKS.keys())}"
            )

        self._task_name = task_name
        self._ms3_task_id = MANISKILL3_TASKS[task_name]
        self._use_camera = use_camera
        self._camera_names = camera_names or self.CAMERA_NAMES
        self._image_width, self._image_height = image_size
        self._control_mode = control_mode
        self._sim_backend = sim_backend
        self._v_max = v_max
        self._w_max = w_max

        self._env: Any | None = None
        self._current_task_config: TaskConfig | None = None
        self._episode_reward = 0.0
        self._episode_success = False
        self._prev_ee_pos: np.ndarray | None = None
        self._control_dt: float | None = None
        self._env_id = f"maniskill3/{task_name}"

        # Procedural generator with ManiSkill3-appropriate ranges
        self._proc_gen = ProceduralTaskGenerator(
            env_id=self._env_id,
            position_ranges={
                "object": np.array([0.05, 0.05, 0.02]),
                "target": np.array([0.05, 0.05, 0.03]),
            },
            physics_ranges={
                "friction": (0.8, 1.2),
                "damping": (0.9, 1.1),
            },
        )

    def _ensure_env(self) -> None:
        """Lazy initialization of ManiSkill3 environment."""
        if self._env is None:
            try:
                import gymnasium as gym  # noqa: PLC0415
                import mani_skill.envs  # type: ignore[import-untyped]  # noqa: F401, PLC0415
            except ImportError as e:
                raise ImportError(
                    "ManiSkill3 is not installed. Install with: pip install mani-skill"
                ) from e

            # Determine render mode based on camera usage
            render_mode = "rgb_array" if self._use_camera else None

            # Build sensor configs for cameras
            sensor_configs = {}
            if self._use_camera:
                sensor_configs = {
                    "width": self._image_width,
                    "height": self._image_height,
                }

            # Build kwargs for gym.make
            # Use state_dict for easier parsing, or rgbd when cameras are needed
            make_kwargs = {
                "obs_mode": "rgbd" if self._use_camera else "state_dict",
                "control_mode": self._control_mode,
            }
            if render_mode:
                make_kwargs["render_mode"] = render_mode
            if sensor_configs:
                make_kwargs["sensor_configs"] = sensor_configs
            if self._sim_backend != "auto":
                make_kwargs["sim_backend"] = self._sim_backend

            # Create environment
            self._env = gym.make(self._ms3_task_id, **make_kwargs)  # type: ignore[arg-type]

            # Get control timestep
            if hasattr(self._env, "unwrapped"):
                unwrapped = cast(Any, self._env.unwrapped)
                if hasattr(unwrapped, "control_freq"):
                    self._control_dt = 1.0 / float(unwrapped.control_freq)
                elif hasattr(unwrapped, "sim_freq") and hasattr(unwrapped, "control_freq_ratio"):
                    self._control_dt = float(unwrapped.control_freq_ratio) / float(
                        unwrapped.sim_freq
                    )
            if self._control_dt is None:
                self._control_dt = 0.05  # Default 20Hz

    @property
    def env_name(self) -> str:
        return "maniskill3"

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
        if not self._use_camera:
            return 0
        return len(self._camera_names)

    @property
    def action_shape(self) -> tuple[int, ...]:
        """Canonical action shape: 7 (6-DOF twist + gripper)."""
        return (7,)

    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        low = np.full(self.action_shape, -1.0, dtype=np.float32)
        high = np.full(self.action_shape, 1.0, dtype=np.float32)
        return (low, high)

    def _to_numpy(self, value: Any) -> np.ndarray:
        """Convert tensor or array to numpy, handling batched data."""
        if hasattr(value, "cpu"):  # torch tensor
            arr = value.cpu().numpy()
        else:
            arr = np.asarray(value)
        # ManiSkill3 returns batched data (batch_size, ...), squeeze if batch_size=1
        if arr.ndim > 1 and arr.shape[0] == 1:
            arr = arr[0]
        return arr.astype(np.float32)

    def _extract_ee_state(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Extract end-effector state from ManiSkill3 observation.

        Returns:
            Tuple of (ee_pos, ee_quat_xyzw, gripper_state)
        """
        # End-effector pose from TCP (tool center point)
        ee_pos = np.zeros(3, dtype=np.float32)
        ee_quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        gripper_state = 0.0

        # ManiSkill3 stores agent state in obs["agent"]
        agent_obs = obs.get("agent", {})

        if isinstance(agent_obs, dict) and "qpos" in agent_obs:
            # Joint positions - gripper is typically last joints
            qpos = self._to_numpy(agent_obs["qpos"])
            # For Panda robot, gripper width is from last 2 joints
            if qpos.size >= 2:
                gripper_width = float(qpos[-1] + qpos[-2])
                # Normalize to [0, 1] - Panda gripper max width ~0.08m
                gripper_state = float(np.clip(gripper_width / 0.08, 0.0, 1.0))

        # Get TCP pose from extra info - this is the primary source
        extra = obs.get("extra", {})
        if isinstance(extra, dict) and "tcp_pose" in extra:
            tcp = self._to_numpy(extra["tcp_pose"])
            if tcp.size >= 7:
                ee_pos = tcp[:3].copy()
                # ManiSkill3 uses [x, y, z, qw, qx, qy, qz], convert to xyzw
                ee_quat = np.array([tcp[4], tcp[5], tcp[6], tcp[3]], dtype=np.float32)

        return ee_pos, normalize_quaternion(ee_quat), gripper_state

    def _extract_ee_velocity(self, obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        """Extract end-effector velocity from observation."""
        ee_lin_vel = np.zeros(3, dtype=np.float32)
        ee_ang_vel = np.zeros(3, dtype=np.float32)

        agent_obs = obs.get("agent", {})
        if isinstance(agent_obs, dict) and "qvel" in agent_obs:
            qvel = self._to_numpy(agent_obs["qvel"])
            # Approximate ee velocity from joint velocities
            # This is a simplification - proper FK would be needed for accuracy
            if qvel.size >= 6:
                ee_lin_vel = qvel[:3] * 0.1  # Scale factor approximation
                ee_ang_vel = qvel[3:6] * 0.1

        return ee_lin_vel, ee_ang_vel

    def _get_camera_images(self, obs: dict[str, Any]) -> dict[str, np.ndarray]:
        """Extract camera images from observation."""
        images = {}

        if not self._use_camera:
            return images

        # ManiSkill3 stores sensor data in obs["sensor_data"] or obs["sensor_param"]
        sensor_data = obs.get("sensor_data", obs.get("image", {}))
        if not isinstance(sensor_data, dict):
            return images

        for cam_name in self._camera_names:
            cam_data = sensor_data.get(cam_name, {})
            if isinstance(cam_data, dict) and "rgb" in cam_data:
                rgb = cam_data["rgb"]
                # Convert tensor to numpy
                if hasattr(rgb, "cpu"):
                    rgb = rgb.cpu().numpy()
                else:
                    rgb = np.asarray(rgb)
                # Handle batched data (batch_size, H, W, C)
                if rgb.ndim == 4 and rgb.shape[0] == 1:
                    rgb = rgb[0]
                # Ensure uint8 format
                if rgb.dtype != np.uint8:
                    if rgb.max() <= 1.0:
                        rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
                    else:
                        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
                images[cam_name] = rgb

        return images

    def _build_observation(self, obs: dict[str, Any]) -> CanonicalObservation:
        """Build canonical observation from ManiSkill3 observation."""
        ee_pos, ee_quat, gripper_state = self._extract_ee_state(obs)
        ee_lin_vel, ee_ang_vel = self._extract_ee_velocity(obs)

        # Compute velocity from position delta if not available
        if np.allclose(ee_lin_vel, 0) and self._prev_ee_pos is not None:
            dt = self._control_dt or 0.05
            ee_lin_vel = (ee_pos - self._prev_ee_pos) / dt

        self._prev_ee_pos = ee_pos.copy()

        camera_views = self._get_camera_images(obs) if self._use_camera else {}

        return CanonicalObservation(
            ee_pos_m=ee_pos.tolist(),
            ee_quat_xyzw=ee_quat.tolist(),
            ee_lin_vel_mps=ee_lin_vel.tolist(),
            ee_ang_vel_rps=ee_ang_vel.tolist(),
            gripper_01=gripper_state,
            rgb={name: img.tolist() for name, img in camera_views.items()},
            depth=None,
            cam_intrinsics_K=None,
            cam_extrinsics_T_world_cam=None,
        )

    def generate_task(self, seed: int) -> TaskConfig:
        """Generate procedural task configuration."""
        self._ensure_env()

        # Apply procedural randomization
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

    def reset(self, task_config: TaskConfig) -> CanonicalObservation:
        """Reset environment with task configuration."""
        self._ensure_env()
        env = cast(Any, self._env)

        self._current_task_config = task_config
        self._episode_reward = 0.0
        self._episode_success = False
        self._prev_ee_pos = None

        # Reset environment with seed
        obs, info = env.reset(seed=task_config.seed)

        # Convert observation to dict if needed (ManiSkill3 can return different formats)
        if not isinstance(obs, dict):
            obs = {"state": obs}

        return self._build_observation(obs)

    def step(
        self,
        action: CanonicalAction | dict[str, Any] | np.ndarray,
    ) -> tuple[CanonicalObservation, float, bool, dict[str, Any]]:
        """Execute action in environment."""
        self._ensure_env()
        env = cast(Any, self._env)

        canonical_action = coerce_action(action)
        twist = np.clip(np.array(canonical_action.twist_ee_norm, dtype=np.float32), -1.0, 1.0)
        gripper = float(np.clip(canonical_action.gripper_01, 0.0, 1.0))

        # Convert canonical twist to ManiSkill3 action format
        # For pd_ee_delta_pose control mode: [delta_pos(3), delta_rot(3), gripper(1)]
        control_dt = self._control_dt if self._control_dt is not None else 0.05
        delta_pos = twist[:3] * self._v_max * control_dt
        delta_rot = twist[3:6] * self._w_max * control_dt

        # Build action array: [dx, dy, dz, drx, dry, drz, gripper]
        # Gripper: -1 = close, 1 = open in ManiSkill3
        ms3_action = np.zeros(7, dtype=np.float32)
        ms3_action[:3] = delta_pos
        ms3_action[3:6] = delta_rot
        ms3_action[6] = gripper * 2.0 - 1.0  # Convert [0,1] to [-1,1]

        # Step environment
        obs, reward, terminated, truncated, info = env.step(ms3_action)

        # Convert observation
        if not isinstance(obs, dict):
            obs = {"state": obs}

        # Convert tensor values to Python scalars (ManiSkill3 returns batched tensors)
        if hasattr(reward, "item"):
            reward = reward.item()
        elif hasattr(reward, "cpu"):
            reward = float(reward.cpu().numpy().flat[0])

        if hasattr(terminated, "item"):
            terminated = terminated.item()
        elif hasattr(terminated, "cpu"):
            terminated = bool(terminated.cpu().numpy().flat[0])

        if hasattr(truncated, "item"):
            truncated = truncated.item()
        elif hasattr(truncated, "cpu"):
            truncated = bool(truncated.cpu().numpy().flat[0])

        self._episode_reward += reward

        # Check success from info (also a tensor in ManiSkill3)
        success = info.get("success", False)
        if hasattr(success, "item"):
            success = success.item()
        elif hasattr(success, "cpu"):
            success = bool(success.cpu().numpy().flat[0])
        if success:
            self._episode_success = True

        done = terminated or truncated

        return self._build_observation(obs), float(reward), bool(done), info

    def get_success(self) -> bool:
        """Check if task was completed successfully."""
        return self._episode_success

    def close(self) -> None:
        """Clean up environment."""
        if self._env is not None:
            self._env.close()
            self._env = None

    def render(self, camera_name: str = "base_camera") -> np.ndarray | None:
        """Render environment frame from specified camera."""
        if self._env is None:
            return None
        try:
            return self._env.render()
        except Exception as e:
            logger.debug("maniskill3_render_failed", camera_name=camera_name, error=str(e))
            return None
