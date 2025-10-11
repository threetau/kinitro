"""
Environment wrappers and utilities with task discovery

We use Gymnasium-compatible interfaces and expose a factory that can
instantiate environments and discover all tasks for provider-agnostic evaluation.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, TypedDict

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from metaworld.wrappers import OneHotWrapper
from PIL import Image

from ..providers.metaworld import load_benchmark_definition

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPISODE_STEPS = 10
DEFAULT_EPISODES_PER_TASK = 3
DEFAULT_TASKS_PER_ENV = 5

# Metaworld observation wrapper constants
OBS_STATE_IDX_END = 4
OBS_TASK_ONE_HOT_START_IDX = 39


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
      - If a `camera_attribute` is provided and exists on the env, it will be
        set to each name in `camera_names` to capture multiple views. If not,
        a single render will be captured without switching cameras.
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
        self._camera_attribute = camera_attribute
        self._camera_names = tuple(camera_names) if camera_names else tuple()

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
            for camera_name in self._camera_names:
                camera_dir = self._image_save_dir / camera_name
                camera_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Image saving enabled to: {self._image_save_dir}")
        else:
            self._image_save_dir = None

        # Decide how many views we will expose in the observation space
        can_switch_cameras = (
            self._camera_attribute is not None
            and hasattr(env, self._camera_attribute)
            and len(self._camera_names) > 0
        )
        num_views = len(self._camera_names) if can_switch_cameras else 1

        # Determine render dimensions from the base (unwrapped) env, falling back to a safe default
        base_env = getattr(env, "unwrapped", env)
        render_width = getattr(base_env, "width", None)
        render_height = getattr(base_env, "height", None)

        # Cache for later fallback usage
        self._render_width = int(render_width)
        self._render_height = int(render_height)

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
        for idx in range(num_views):
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

    def capture_and_save_images(self) -> tuple[list[np.ndarray], list[str]]:
        """Capture images from all configured camera views and optionally save them.

        Returns:
            Tuple of (images_hwc, camera_names_used) where images_hwc is a list of
            HWC format numpy arrays and camera_names_used is the list of camera names.
        """
        images_hwc: list[np.ndarray] = []
        camera_names_used: list[str] = []

        img = self._render_rgb_array()
        images_hwc.append(img)
        camera_names_used.append("default")

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

        for camera in self._camera_names:
            img = self._render_rgb_array()
            # Convert HWC to CHW
            img_chw = np.transpose(img, (2, 0, 1))
            # TODO: we flip the image because it is upside down for some reason
            # this appears to be something to do with metaworld/mujoco rendering? look more into it
            img_chw = np.flip(img_chw, axis=1)
            key = (
                "observation.image"
                if camera == self._camera_names[0]
                else f"observation.image{self._camera_names.index(camera) + 1}"
            )
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


class EnvManager:
    """Manager for creating environments and discovering tasks."""

    def __init__(self):
        # Configure headless rendering on initialization
        configure_headless_rendering()

    def get_benchmark_envs(self, benchmark_spec: BenchmarkSpec) -> list[EnvSpec]:
        """Get all test environments for a given benchmark"""
        if benchmark_spec.provider == "metaworld":
            return self._get_metaworld_benchmark_envs(benchmark_spec)
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

    def make_env(
        self,
        env_spec: EnvSpec,
        submission_id: str | None = None,
        save_images: bool = False,  # TODO: keep or move/remove?
    ) -> gym.Env:
        """Create an environment for a specific environment spec."""
        if env_spec.provider == "metaworld":
            env = self._make_metaworld_env(env_spec, submission_id, save_images)
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


# Global environment manager instance
env_manager = EnvManager()
