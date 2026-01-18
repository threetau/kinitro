"""Episode rollout engine for evaluating miner policies."""

import asyncio
import base64
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from robo.environments.base import EpisodeResult, RoboticsEnvironment, TaskConfig


@dataclass
class ObservationData:
    """
    Observation data sent to miner policies.

    Contains both proprioceptive state and camera images.
    """

    # Proprioceptive observations
    end_effector_pos: list[float]  # [x, y, z]
    gripper_state: float  # 0=closed, 1=open

    # Camera images as base64-encoded PNGs (for efficient transfer)
    camera_images: dict[str, str]  # camera_name -> base64 encoded image

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "end_effector_pos": self.end_effector_pos,
            "gripper_state": self.gripper_state,
            "camera_images": self.camera_images,
        }


def encode_image_base64(image: np.ndarray) -> str:
    """Encode numpy image array to base64 PNG string."""
    import io

    from PIL import Image

    img = Image.fromarray(image.astype(np.uint8))
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def decode_image_base64(b64_string: str) -> np.ndarray:
    """Decode base64 PNG string to numpy array."""
    import io

    from PIL import Image

    buffer = io.BytesIO(base64.b64decode(b64_string))
    img = Image.open(buffer)
    return np.array(img)


class PolicyInterface(Protocol):
    """Protocol for miner policy interface."""

    async def reset(self, task_config: dict[str, Any]) -> None:
        """Reset policy for new episode."""
        ...

    async def act(self, observation: dict[str, Any]) -> list[float]:
        """
        Get action for observation.

        Args:
            observation: Dict with keys:
                - end_effector_pos: [x, y, z]
                - gripper_state: float
                - camera_images: {camera_name: base64_png_string}

        Returns:
            Action as list of floats
        """
        ...


@dataclass
class RolloutConfig:
    """Configuration for episode rollouts."""

    max_timesteps: int = 500
    action_timeout_ms: int = 100  # Increased for image processing
    include_cameras: bool = True
    render: bool = False
    collect_frames: bool = False


@dataclass
class DetailedEpisodeResult:
    """Extended episode result with additional diagnostics."""

    result: EpisodeResult
    observations: list[np.ndarray] = field(default_factory=list)
    actions: list[np.ndarray] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    frames: list[np.ndarray] = field(default_factory=list)
    failure_reason: str | None = None


def _build_observation_data(
    env: RoboticsEnvironment,
    proprio_obs: np.ndarray,
    include_cameras: bool = True,
) -> ObservationData:
    """
    Build observation data from environment state.

    Args:
        env: Environment instance
        proprio_obs: Proprioceptive observation (4D: end-effector XYZ + gripper)
        include_cameras: Whether to include camera images

    Returns:
        ObservationData with proprioceptive and visual observations
    """
    camera_images = {}

    if include_cameras and hasattr(env, "get_observation"):
        try:
            full_obs = env.get_observation()
            for cam_name, img in full_obs.camera_views.items():
                camera_images[cam_name] = encode_image_base64(img)
        except Exception:
            # Camera rendering may fail
            pass

    return ObservationData(
        end_effector_pos=proprio_obs[0:3].tolist(),
        gripper_state=float(proprio_obs[3]) if len(proprio_obs) > 3 else 1.0,
        camera_images=camera_images,
    )


async def run_episode(
    env: RoboticsEnvironment,
    task_config: TaskConfig,
    policy: PolicyInterface,
    config: RolloutConfig | None = None,
) -> EpisodeResult:
    """
    Run a single episode, querying miner policy for actions.

    This is the core evaluation loop:
    1. Reset environment with procedural task config
    2. Reset miner policy with task info
    3. Loop: get observation -> query policy -> step environment
    4. Return success/reward metrics

    Observations sent to policy include:
    - Proprioceptive: end-effector XYZ + gripper state
    - Visual: camera images as base64 PNGs

    Args:
        env: Robotics environment instance
        task_config: Procedurally generated task configuration
        policy: Miner policy interface (typically Basilica container)
        config: Rollout configuration

    Returns:
        EpisodeResult with success, reward, and timestep info
    """
    if config is None:
        config = RolloutConfig()

    # Reset environment and policy
    proprio_obs = env.reset(task_config)
    await policy.reset(task_config=task_config.to_dict())

    total_reward = 0.0
    timesteps = 0

    for t in range(config.max_timesteps):
        # Build observation data
        obs_data = _build_observation_data(env, proprio_obs, config.include_cameras)

        # Query miner policy with timeout
        try:
            action_list = await asyncio.wait_for(
                policy.act(observation=obs_data.to_dict()),
                timeout=config.action_timeout_ms / 1000.0,
            )
            action = np.array(action_list, dtype=np.float32)
        except TimeoutError:
            # Miner too slow - episode fails
            return EpisodeResult(
                success=False,
                total_reward=total_reward,
                timesteps=t,
                info={"failure_reason": "action_timeout"},
            )
        except Exception as e:
            # Policy error - episode fails
            return EpisodeResult(
                success=False,
                total_reward=total_reward,
                timesteps=t,
                info={"failure_reason": f"policy_error: {e}"},
            )

        # Validate action shape
        expected_shape = env.action_shape
        if action.shape != expected_shape:
            return EpisodeResult(
                success=False,
                total_reward=total_reward,
                timesteps=t,
                info={
                    "failure_reason": f"invalid_action_shape: expected {expected_shape}, got {action.shape}"
                },
            )

        # Clip action to valid bounds
        low, high = env.action_bounds
        action = np.clip(action, low, high)

        # Step environment
        try:
            proprio_obs, reward, done, info = env.step(action)
        except Exception as e:
            return EpisodeResult(
                success=False,
                total_reward=total_reward,
                timesteps=t,
                info={"failure_reason": f"env_error: {e}"},
            )

        total_reward += reward
        timesteps = t + 1

        if done:
            break

    return EpisodeResult(
        success=env.get_success(),
        total_reward=total_reward,
        timesteps=timesteps,
        info={},
    )


async def run_episode_detailed(
    env: RoboticsEnvironment,
    task_config: TaskConfig,
    policy: PolicyInterface,
    config: RolloutConfig | None = None,
) -> DetailedEpisodeResult:
    """
    Run episode with full trajectory recording.

    Useful for debugging and visualization.

    Args:
        env: Robotics environment instance
        task_config: Procedurally generated task configuration
        policy: Miner policy interface
        config: Rollout configuration

    Returns:
        DetailedEpisodeResult with full trajectory
    """
    if config is None:
        config = RolloutConfig()

    # Reset
    proprio_obs = env.reset(task_config)
    await policy.reset(task_config=task_config.to_dict())

    observations: list[np.ndarray] = [proprio_obs.copy()]
    actions: list[np.ndarray] = []
    rewards: list[float] = []
    frames: list[np.ndarray] = []

    total_reward = 0.0
    timesteps = 0
    failure_reason = None

    for t in range(config.max_timesteps):
        # Collect frame if requested
        if config.collect_frames and hasattr(env, "render"):
            frame = env.render()
            if frame is not None:
                frames.append(frame)

        # Build observation data
        obs_data = _build_observation_data(env, proprio_obs, config.include_cameras)

        # Query policy
        try:
            action_list = await asyncio.wait_for(
                policy.act(observation=obs_data.to_dict()),
                timeout=config.action_timeout_ms / 1000.0,
            )
            action = np.array(action_list, dtype=np.float32)
        except TimeoutError:
            failure_reason = "action_timeout"
            break
        except Exception as e:
            failure_reason = f"policy_error: {e}"
            break

        # Validate and clip
        if action.shape != env.action_shape:
            failure_reason = f"invalid_action_shape: {action.shape}"
            break

        low, high = env.action_bounds
        action = np.clip(action, low, high)
        actions.append(action.copy())

        # Step
        try:
            proprio_obs, reward, done, _ = env.step(action)
        except Exception as e:
            failure_reason = f"env_error: {e}"
            break

        observations.append(proprio_obs.copy())
        rewards.append(float(reward))
        total_reward += reward
        timesteps = t + 1

        if done:
            break

    result = EpisodeResult(
        success=env.get_success() if failure_reason is None else False,
        total_reward=total_reward,
        timesteps=timesteps,
        info={"failure_reason": failure_reason} if failure_reason else {},
    )

    return DetailedEpisodeResult(
        result=result,
        observations=observations,
        actions=actions,
        rewards=rewards,
        frames=frames,
        failure_reason=failure_reason,
    )
