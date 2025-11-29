"""
Environment wrappers and utilities with task discovery

We use Gymnasium-compatible interfaces and expose a factory that can
instantiate environments and discover all tasks for provider-agnostic evaluation.
"""

from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, TypedDict, cast

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from metaworld.wrappers import OneHotWrapper
from PIL import Image

from evaluator.providers.swarm.core.moving_drone import MovingDroneAviary
from evaluator.providers.swarm.validator.task_gen import random_task

from ..providers import swarm as swarm_provider
from ..providers.metaworld import load_benchmark_definition

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPISODE_STEPS = 10
DEFAULT_EPISODES_PER_TASK = 3
DEFAULT_TASKS_PER_ENV = 5

# Metaworld observation wrapper constants
OBS_STATE_IDX_END = 4
OBS_TASK_ONE_HOT_START_IDX = 39
DEFAULT_RENDER_SIZE = 480


def configure_headless_rendering():
    """Configure environment variables for headless MuJoCo rendering using EGL."""
    if "DISPLAY" not in os.environ:
        # Only configure headless mode if no display is available
        os.environ["MUJOCO_GL"] = "egl"
        os.environ["PYOPENGL_PLATFORM"] = "egl"
        logger.info("Configured headless rendering with EGL backend")
    else:
        logger.info("Display available, using default OpenGL rendering")


@dataclass
class EnvSpec:
    """Specification for a single environment."""

    env_name: str
    benchmark_name: str
    provider: str
    config: Dict[str, Any] = field(default_factory=dict)

    # Runtime controls (optional configuration)
    episodes_per_task: int = DEFAULT_EPISODES_PER_TASK
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS
    render_mode: str | None = "rgb_array"

    # Observation capture options
    camera_attribute: str | None = "camera_name"
    camera_names: tuple[str, ...] = ("corner",)

    def __str__(self) -> str:
        return f"{self.provider}/{self.benchmark_name}/{self.env_name}"


@dataclass
class EpisodeResult:
    """Result from a single episode."""

    env_spec: EnvSpec
    episode_id: int
    success: bool
    reward: float
    steps: int
    info: Dict[str, Any] = field(default_factory=dict)


# TODO: add task name from i.e. task_spec to the EnvResult?
# this would be useful for identifying which task within a multi-task benchmark/env
@dataclass
class EnvResult:
    """Aggregated results from all episodes of an environment in a benchmark."""

    env_spec: EnvSpec
    episodes: List[EpisodeResult]
    success_rate: float
    mean_reward: float
    mean_steps: float

    @classmethod
    def from_episodes(
        cls, env_spec: EnvSpec, episodes: List[EpisodeResult]
    ) -> "EnvResult":
        if not episodes:
            return cls(env_spec, [], 0.0, 0.0, 0.0)

        success_rate = sum(ep.success for ep in episodes) / len(episodes)
        mean_reward = float(np.mean([ep.reward for ep in episodes]))
        mean_steps = float(np.mean([ep.steps for ep in episodes]))

        return cls(env_spec, episodes, success_rate, mean_reward, mean_steps)


class BenchmarkConfig(TypedDict, total=False):
    """Type hints for benchmark configuration."""

    env_name: str  # Name of environment from the benchmark
    episodes_per_task: int  # Number of episodes to run per task
    max_episode_steps: int  # Maximum steps per episode
    tasks_per_env: int  # Number of task variants to sample per environment
    task_seed: int  # Optional RNG seed for task generation


@dataclass
class BenchmarkSpec:
    """Specification for a benchmark and its environments."""

    provider: str  # "metaworld", etc.
    benchmark_name: str  # "MT1", "MT10", etc.

    # Additional configuration
    config: BenchmarkConfig = field(default_factory=dict)

    # Runtime controls
    render_mode: str | None = "rgb_array"

    # Observation capture options
    # TODO: make these per-env-spec instead of per-benchmark-spec?
    camera_names: tuple[str, ...] = ("corner",)
    camera_attribute: str | None = "camera_name"

    def __str__(self) -> str:
        return f"{self.provider}/{self.benchmark_name}"


class MetaworldObsWrapper(ObservationWrapper):
    """
    Observation wrapper for metaworld that augments observations with rendered images

    Behavior:
      - Always returns a Dict observation with key "base" holding the original
        observation.
      - If the environment supports rendering to RGB arrays, captures one or
        more views and includes them as CHW uint8 tensors under keys:
        "observation.image", "observation.image2", "observation.image3", ...
      - If the environment exposes `camera_name`, it will be set to each name in
        `camera_names` to capture multiple views. If not, a single render will
        be captured without switching cameras.
      - Optionally saves images to disk for debugging/analysis.
    """

    def __init__(
        self,
        env: gym.Env,
        camera_attribute: str | None = "camera_name",
        camera_names: tuple[str, ...] = ("corner",),
        save_images: bool = False,
        image_save_dir: str | None = None,
        submission_id: str | None = None,
    ):
        super().__init__(env)
        self._base_env = getattr(env, "unwrapped", env)
        if camera_attribute not in (None, "camera_name"):
            logger.warning(
                "MetaworldObsWrapper only supports 'camera_name'; overriding provided "
                "camera_attribute '%s'",
                camera_attribute,
            )
        self._camera_attribute = "camera_name"
        self._camera_names = tuple(camera_names) if camera_names else tuple()
        self._can_switch_cameras = bool(
            len(self._camera_names) > 0
            and (hasattr(env, "camera_name") or hasattr(self._base_env, "camera_name"))
        )
        self._num_views = len(self._camera_names) if self._can_switch_cameras else 1

        # Image saving configuration
        self._save_images = save_images
        self._submission_id = submission_id
        self._step_count = 0
        self._episode_count = 0

        # Setup image save directory if enabled
        if self._save_images and image_save_dir:
            self._image_save_dir = Path(image_save_dir)
            if self._submission_id:
                self._image_save_dir = self._image_save_dir / str(self._submission_id)

            # Create directories for each camera
            for camera_name in self._camera_names or ("default",):
                camera_dir = self._image_save_dir / camera_name
                camera_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Image saving enabled to: {self._image_save_dir}")
        else:
            self._image_save_dir = None

        # Determine render dimensions from the base (unwrapped) env, falling back to a safe default
        render_width = getattr(self._base_env, "width", None)
        render_height = getattr(self._base_env, "height", None)

        # Cache for later fallback usage
        self._render_width = (
            int(render_width) if render_width is not None else DEFAULT_RENDER_SIZE
        )
        self._render_height = (
            int(render_height) if render_height is not None else DEFAULT_RENDER_SIZE
        )

        # Build observation space dynamically based on number of views requested
        image_box = Box(
            low=0,
            high=255,
            # Use cached render dimensions from the base env
            shape=(3, self._render_height, self._render_width),
            dtype=np.uint8,
        )

        space_dict: dict[str, Box | gym.spaces.Space] = {
            "observation.state": env.observation_space
        }
        for idx in range(self._num_views):
            key = "observation.image" if idx == 0 else f"observation.image{idx + 1}"
            space_dict[key] = image_box

        self.observation_space = DictSpace(space_dict)

    def _render_rgb_array(self) -> np.ndarray:
        # Expect HWC uint8 output from env.render()
        try:
            frame = self.env.render()
            if frame is None:
                # Some environments need to be stepped before rendering works
                # Return a black image as fallback
                logger.warning(
                    "Environment render returned None, using fallback black image"
                )
                return np.zeros(
                    (self._render_height, self._render_width, 3), dtype=np.uint8
                )
        except Exception as e:
            logger.warning(
                f"Failed to render environment: {e}, using fallback black image"
            )
            return np.zeros(
                (self._render_height, self._render_width, 3), dtype=np.uint8
            )

        return np.asarray(frame)

    def _camera_targets(self) -> tuple[gym.Env, ...]:
        if self._base_env is self.env:
            return (self.env,)
        return (self.env, self._base_env)

    def _apply_camera(
        self, camera_value: str | None, *, log_on_error: bool = True
    ) -> bool:
        """Attempt to set `camera_name` and keep the renderer in sync."""
        success = False
        for target in self._camera_targets():
            if hasattr(target, "camera_name"):
                try:
                    setattr(target, "camera_name", camera_value)
                    success = True
                except Exception as exc:
                    if log_on_error:
                        logger.warning(
                            "Failed to set camera '%s' on %s: %s",
                            camera_value,
                            target.__class__.__name__,
                            exc,
                        )

            renderer = getattr(target, "mujoco_renderer", None)
            if renderer is not None:
                if camera_value is not None and mujoco is not None:
                    try:
                        camera_id = mujoco.mj_name2id(
                            renderer.model,
                            mujoco.mjtObj.mjOBJ_CAMERA,
                            camera_value,
                        )
                        if camera_id == -1:
                            raise ValueError(
                                f"Camera '{camera_value}' not found in model"
                            )
                        renderer.camera_id = camera_id
                        setattr(renderer, "camera_name", camera_value)
                        success = True
                    except Exception as exc:
                        if log_on_error:
                            logger.warning(
                                "Failed to set renderer camera '%s' on %s: %s",
                                camera_value,
                                target.__class__.__name__,
                                exc,
                            )
                elif camera_value is not None:
                    setattr(renderer, "camera_name", camera_value)
                    success = True
                elif camera_value is None:
                    setattr(renderer, "camera_name", None)

        return success

    def _get_camera(self) -> str | None:
        if hasattr(self.env, "camera_name"):
            return getattr(self.env, "camera_name", None)
        if hasattr(self._base_env, "camera_name"):
            return getattr(self._base_env, "camera_name", None)
        renderer = getattr(self.env, "mujoco_renderer", None)
        if renderer is None and self._base_env is not self.env:
            renderer = getattr(self._base_env, "mujoco_renderer", None)
        if renderer is not None and hasattr(renderer, "camera_name"):
            return getattr(renderer, "camera_name", None)
        return None

    def capture_and_save_images(self) -> tuple[list[np.ndarray], list[str]]:
        """Capture images from all configured camera views and optionally save them.

        Returns:
            Tuple of (images_hwc, camera_names_used) where images_hwc is a list of
            HWC format numpy arrays and camera_names_used is the list of camera names.
        """
        images_hwc: list[np.ndarray] = []
        camera_names_used: list[str] = []

        if self._can_switch_cameras:
            original_camera = self._get_camera()
            # capture renderer camera_ids to restore later
            renderer_restore: list[tuple[Any, Any]] = []
            seen_renderers: set[int] = set()
            for target in self._camera_targets():
                renderer = getattr(target, "mujoco_renderer", None)
                if renderer is not None and id(renderer) not in seen_renderers:
                    seen_renderers.add(id(renderer))
                    original_id = getattr(renderer, "camera_id", None)
                    renderer_restore.append((renderer, original_id))
            try:
                for camera_name in self._camera_names:
                    applied = self._apply_camera(camera_name)
                    if not applied:
                        logger.warning(
                            "Unable to set camera '%s'; using current camera for rendering",
                            camera_name,
                        )
                    img = self._render_rgb_array()
                    images_hwc.append(img)
                    camera_names_used.append(camera_name)
            finally:
                self._apply_camera(original_camera, log_on_error=False)
                for renderer, camera_id in renderer_restore:
                    if camera_id is not None:
                        try:
                            renderer.camera_id = camera_id
                        except Exception:
                            pass
        else:
            img = self._render_rgb_array()
            images_hwc.append(img)
            current_camera = self._get_camera()
            camera_names_used.append(
                str(current_camera) if current_camera else "default"
            )

        # Save images to disk if enabled
        if self._save_images and self._image_save_dir:
            self._save_images_to_disk(images_hwc, camera_names_used)

        return images_hwc, camera_names_used

    def observation(self, obs) -> Dict[str, Any]:
        self._step_count += 1

        # The only information we send to the agent are:
        # - 0:2: XYZ coordinates of the end-effector
        # - 3: gripper open/close state
        # - 39:n: one hot vector indicating the task
        # and an image from one or more cameras
        state = obs[:OBS_STATE_IDX_END]
        # one hot vector indicating the task
        one_hot = obs[OBS_TASK_ONE_HOT_START_IDX:]
        obs = np.concatenate([state, one_hot])
        new_obs: Dict[str, Any] = {"observation.state": obs}

        images_hwc, _ = self.capture_and_save_images()
        for idx, img in enumerate(images_hwc):
            # Convert HWC to CHW
            img_chw = np.transpose(img, (2, 0, 1))
            # TODO: we flip the image because it is upside down for some reason
            # this appears to be something to do with metaworld/mujoco rendering? look more into it
            img_chw = np.flip(img_chw, axis=1)
            key = "observation.image" if idx == 0 else f"observation.image{idx + 1}"
            new_obs[key] = img_chw

        return new_obs

    def _save_images_to_disk(
        self, images_hwc: list[np.ndarray], camera_names: list[str]
    ) -> None:
        """Save rendered images to disk organized by camera."""
        try:
            for img, camera_name in zip(images_hwc, camera_names):
                if self._image_save_dir:
                    camera_dir = self._image_save_dir / camera_name
                    camera_dir.mkdir(parents=True, exist_ok=True)
                    filename = (
                        f"ep{self._episode_count:04d}_step{self._step_count:06d}.png"
                    )
                    filepath = camera_dir / filename

                    # Convert numpy array to PIL Image and save
                    # img is HWC format, convert to PIL format
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    pil_img.save(filepath)

        except Exception as e:
            logger.warning(f"Failed to save image: {e}")

    def reset(self, **kwargs):
        """Reset the environment and increment episode counter."""
        self._episode_count += 1
        self._step_count = 0
        return super().reset(**kwargs)


class DroneObsWrapper(ObservationWrapper):
    """
    Observation wrapper for drone environments to add rendered images to observations.

    Behavior:
      - Always returns a Dict observation with key "base" holding the original
        observation.
      - Captures an image from the drone's camera
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.observation_space = DictSpace(
            {
                "observation.state": env.observation_space,
            }
        )

    def observation(self, obs) -> Dict[str, Any]:
        new_obs: Dict[str, Any] = {"observation.state": obs}
        return new_obs

    def capture_and_save_images(self) -> tuple[list[np.ndarray], list[str]]:
        """Capture image from the drone's camera."""
        print("Capturing drone images for observation")
        env: MovingDroneAviary = cast(MovingDroneAviary, self.env)
        # BaseAviary expects the drone index (0-based), not the PyBullet body ID
        rbg = env.get_third_person_rgb(distance=2)
        # convert h,w,c,a to h,w,c
        rbg = rbg[:, :, :3]
        print("Captured drone images for observation")
        return [rbg], ["rgb"]


class EnvManager:
    """Manager for creating environments and discovering tasks."""

    def __init__(self):
        # Configure headless rendering on initialization
        configure_headless_rendering()

    def get_benchmark_envs(self, benchmark_spec: BenchmarkSpec) -> list[EnvSpec]:
        """Get all test environments for a given benchmark"""
        if benchmark_spec.provider == "metaworld":
            return self._get_metaworld_benchmark_envs(benchmark_spec)
        if benchmark_spec.provider == "swarm":
            return self._get_swarm_benchmark_envs(benchmark_spec)
        else:
            raise ValueError(
                f"Unsupported environment provider: {benchmark_spec.provider}"
            )

    def _get_metaworld_benchmark_envs(
        self, benchmark_spec: BenchmarkSpec
    ) -> list[EnvSpec]:
        """Get all MetaWorld test environments and tasks for the specified benchmark."""
        tasks_per_env = int(
            benchmark_spec.config.get("tasks_per_env", DEFAULT_TASKS_PER_ENV)
        )
        if tasks_per_env < 1:
            raise ValueError("tasks_per_env must be >= 1")

        task_seed = benchmark_spec.config.get("task_seed")
        if task_seed is None:
            task_seed = random.randint(0, 2**31 - 1)
            benchmark_spec.config["task_seed"] = task_seed
        env_name_override = benchmark_spec.config.get("env_name")

        benchmark_data = load_benchmark_definition(
            benchmark_spec.benchmark_name,
            tasks_per_env=tasks_per_env,
            env_name=env_name_override,
            seed=task_seed,
        )

        if benchmark_spec.benchmark_name in {"MT1", "MT10", "MT25", "MT50"}:
            class_lookup = benchmark_data.train_classes
            task_source = benchmark_data.train_tasks
        elif benchmark_spec.benchmark_name in {"ML10", "ML25", "ML45"}:
            class_lookup = benchmark_data.test_classes
            task_source = benchmark_data.test_tasks
        else:
            raise ValueError(
                f"Unsupported MetaWorld benchmark: {benchmark_spec.benchmark_name}"
            )

        env_specs: list[EnvSpec] = []
        class_order = tuple(class_lookup.keys())

        if env_name_override and env_name_override not in class_lookup:
            raise ValueError(
                f"Environment '{env_name_override}' not found in benchmark {benchmark_spec.benchmark_name}"
            )

        for env_id, env_name in enumerate(class_order):
            if env_name_override and env_name != env_name_override:
                continue

            env_tasks = [task for task in task_source if task.env_name == env_name]

            for task_idx, task in enumerate(env_tasks):
                env_spec = EnvSpec(
                    env_name=env_name,
                    benchmark_name=benchmark_spec.benchmark_name,
                    provider="metaworld",
                    config={
                        "task_idx": task_idx,
                        "task_data": task,
                        "env_cls": class_lookup[env_name],
                        "env_id": env_id,
                        "class_order": class_order,
                        "task_seed": task_seed,
                    },
                    episodes_per_task=benchmark_spec.config.get(
                        "episodes_per_task", DEFAULT_EPISODES_PER_TASK
                    ),
                    max_episode_steps=benchmark_spec.config.get(
                        "max_episode_steps", DEFAULT_MAX_EPISODE_STEPS
                    ),
                    render_mode=benchmark_spec.render_mode,
                    camera_attribute=benchmark_spec.camera_attribute,
                    camera_names=benchmark_spec.camera_names,
                )
                env_specs.append(env_spec)

        logger.info(
            "Found %d test tasks across all environments for %s",
            len(env_specs),
            benchmark_spec,
        )
        return env_specs

    def _get_swarm_benchmark_envs(self, benchmark_spec: BenchmarkSpec) -> list[EnvSpec]:
        """Generate MapTask-based environments for the Swarm PyBullet provider."""
        tasks_per_env = int(
            benchmark_spec.config.get("tasks_per_env", DEFAULT_TASKS_PER_ENV)
        )
        if tasks_per_env < 1:
            raise ValueError("tasks_per_env must be >= 1")

        task_seed = benchmark_spec.config.get("task_seed")
        if task_seed is not None:
            task_seed = int(task_seed)
        else:
            task_seed = random.randint(0, 2**31 - 1)
            benchmark_spec.config["task_seed"] = task_seed

        sim_dt = float(benchmark_spec.config.get("sim_dt", swarm_provider.SIM_DT))
        horizon = float(
            benchmark_spec.config.get("horizon", swarm_provider.HORIZON_SEC)
        )
        gui = bool(benchmark_spec.config.get("gui", False))
        env_name = benchmark_spec.config.get("env_name", "swarm-moving-drone")
        episodes_per_task = int(
            benchmark_spec.config.get("episodes_per_task", DEFAULT_EPISODES_PER_TASK)
        )
        max_episode_steps_override = benchmark_spec.config.get("max_episode_steps")
        payload_mode = bool(benchmark_spec.config.get("payload_mode", False))
        challenge_type = benchmark_spec.config.get("challenge_type")

        if sim_dt <= 0:
            raise ValueError("sim_dt must be > 0 for Swarm provider")
        if horizon <= 0:
            raise ValueError("horizon must be > 0 for Swarm provider")
        if max_episode_steps_override is not None:
            if int(max_episode_steps_override) < 1:
                raise ValueError("max_episode_steps must be >= 1 for Swarm provider")

        env_specs: list[EnvSpec] = []
        for task_idx in range(tasks_per_env):
            seed = task_seed + task_idx if task_seed is not None else None
            task = swarm_provider.random_task(
                sim_dt,
                horizon,
                seed=seed,
                payload=payload_mode,
                challenge_type=challenge_type,
            )

            max_episode_steps = (
                int(max_episode_steps_override)
                if max_episode_steps_override is not None
                else int(max(1, round(task.horizon / task.sim_dt)))
            )

            env_specs.append(
                EnvSpec(
                    env_name=env_name,
                    benchmark_name=benchmark_spec.benchmark_name,
                    provider="swarm",
                    config={
                        "task": task,
                        "task_idx": task_idx,
                        "task_seed": seed,
                        "gui": gui,
                        "payload_mode": payload_mode,
                        "challenge_type": challenge_type,
                    },
                    episodes_per_task=episodes_per_task,
                    max_episode_steps=max_episode_steps,
                    render_mode=None,
                    camera_attribute=None,
                    camera_names=tuple(),
                )
            )

        logger.info(
            "Generated %d Swarm tasks (sim_dt=%.4f, horizon=%.2f) for %s",
            len(env_specs),
            sim_dt,
            horizon,
            benchmark_spec,
        )
        return env_specs

    def make_env(
        self,
        env_spec: EnvSpec,
        submission_id: str | None = None,
        save_images: bool = False,  # TODO: keep or move/remove?
    ) -> gym.Env:
        """Create an environment for a specific environment spec."""
        if env_spec.provider == "metaworld":
            env = self._make_metaworld_env(env_spec, submission_id, save_images)
        elif env_spec.provider == "swarm":
            env = self._make_swarm_env(env_spec)
        else:
            raise ValueError(f"Unsupported environment provider: {env_spec.provider}")

        return env

    def _make_metaworld_env(
        self, env_spec: EnvSpec, submission_id: str | None, save_images: bool
    ) -> gym.Env:
        """Create a MetaWorld environment for the specified environment spec."""
        config = env_spec.config
        env_cls = config.get("env_cls")
        if env_cls is None:
            raise ValueError("EnvSpec config missing 'env_cls' for MetaWorld env")

        render_mode = env_spec.render_mode
        env = env_cls(render_mode=render_mode, camera_name="corner")

        task = config.get("task_data")
        if task is None:
            raise ValueError("EnvSpec config missing 'task_data' for MetaWorld env")
        env.set_task(task)

        class_order: tuple[str, ...] | tuple[()] = config.get("class_order", tuple())
        env_id = config.get("env_id")
        if class_order and env_id is not None:
            env = OneHotWrapper(env, env_id, len(class_order))

        logger.debug("Applying MetaworldObsWrapper")
        env = MetaworldObsWrapper(
            env,
            camera_attribute=env_spec.camera_attribute,
            camera_names=env_spec.camera_names,
            save_images=save_images,
            image_save_dir="data" if save_images else None,
            submission_id=submission_id,
        )

        env = gym.wrappers.TimeLimit(env, max_episode_steps=env_spec.max_episode_steps)

        return env

    def _make_swarm_env(self, env_spec: EnvSpec) -> gym.Env:
        """Create a Swarm PyBullet environment from the provided task config."""
        config = env_spec.config
        task = config.get("task")
        seed = config.get("task_seed")
        challenge_type = config.get("challenge_type")
        if task is None:
            task = random_task(
                sim_dt=swarm_provider.SIM_DT,
                horizon=swarm_provider.HORIZON_SEC,
                seed=seed,
                payload=bool(config.get("payload_mode", False)),
                challenge_type=challenge_type,
            )

        gui = bool(config.get("gui", False))
        env = swarm_provider.make_env(task, gui=gui)
        env = DroneObsWrapper(env)

        if env_spec.max_episode_steps:
            env = gym.wrappers.TimeLimit(
                env, max_episode_steps=env_spec.max_episode_steps
            )

        return env


# Global environment manager instance
env_manager = EnvManager()
