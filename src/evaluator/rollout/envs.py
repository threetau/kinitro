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
from typing import Any, Dict, List

import gymnasium as gym
import metaworld
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.wrappers import HumanRendering
from PIL import Image

logger = logging.getLogger(__name__)

# Constant ripped from lerobot.constants
OBS_IMAGES = "observation.images"


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
    max_episode_steps: int = 200
    render_mode: str | None = "rgb_array"

    # Observation capture options
    enable_image_obs: bool = True
    image_size: tuple[int, int] = (64, 64)
    camera_attribute: str | None = "camera_name"
    camera_names: tuple[str, ...] = ("topview", "corner3", "gripperPOV")

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


@dataclass
class BenchmarkSpec:
    """Specification for a benchmark and its environments."""

    provider: str  # "metaworld", etc.
    benchmark_name: str  # "MT1", "MT10", etc.

    # Additional configuration
    config: Dict[str, Any] = field(default_factory=dict)

    # Runtime controls
    max_episode_steps: int = 200
    render_mode: str | None = "rgb_array"

    # Observation capture options
    enable_image_obs: bool = True
    image_size: tuple[int, int] = (64, 64)
    camera_attribute: str | None = "camera_name"
    camera_names: tuple[str, ...] = ("topview", "corner3", "gripperPOV")

    def __str__(self) -> str:
        return f"{self.provider}/{self.benchmark_name}"


class MultiViewImageObsWrapper(ObservationWrapper):
    """
    Generic observation wrapper that augments observations with rendered images.

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
        image_size: tuple[int, int] = (64, 64),
        camera_attribute: str | None = "camera_name",
        camera_names: tuple[str, ...] = ("topview", "corner3", "gripperPOV"),
        save_images: bool = False,
        image_save_dir: str | None = None,
        submission_id: str | None = None,
    ):
        super().__init__(env)
        self._img_h, self._img_w = int(image_size[0]), int(image_size[1])
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

        # Build observation space dynamically based on number of views requested
        image_box = Box(
            low=0,
            high=255,
            shape=(3, self._img_h, self._img_w),
            dtype=np.uint8,
        )

        space_dict: dict[str, Box | gym.spaces.Space] = {"base": env.observation_space}
        for idx in range(num_views):
            key = "observation.image" if idx == 0 else f"observation.image{idx + 1}"
            space_dict[key] = image_box

        self.observation_space = DictSpace(space_dict)

    def _set_camera_if_possible(self, name: str | None) -> None:
        if self._camera_attribute is None or name is None:
            return
        try:
            if hasattr(self.env, self._camera_attribute):
                setattr(self.env, self._camera_attribute, name)
        except Exception:
            # Best-effort; ignore if env does not support dynamic camera switching
            pass

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
                return np.zeros((self._img_h, self._img_w, 3), dtype=np.uint8)
        except Exception as e:
            logger.warning(
                f"Failed to render environment: {e}, using fallback black image"
            )
            return np.zeros((self._img_h, self._img_w, 3), dtype=np.uint8)

        return np.asarray(frame)

    def observation(self, obs):  # type: ignore[override]
        images_hwc: list[np.ndarray] = []
        camera_names_used: list[str] = []

        if (
            self._camera_attribute is not None
            and hasattr(self.env, self._camera_attribute)
            and len(self._camera_names) > 0
        ):
            # Save current camera to restore later
            try:
                current_cam = getattr(self.env, self._camera_attribute)
            except Exception:
                current_cam = None

            for cam in self._camera_names:
                self._set_camera_if_possible(cam)
                img = self._render_rgb_array()
                images_hwc.append(img)
                camera_names_used.append(cam)

            # Restore original camera
            self._set_camera_if_possible(current_cam)
        else:
            # Single-view capture without camera switching
            img = self._render_rgb_array()
            images_hwc.append(img)
            camera_names_used.append("default")

        # Save images to disk if enabled
        if self._save_images and self._image_save_dir:
            self._save_images_to_disk(images_hwc, camera_names_used)

        # Convert images to CHW uint8
        images_chw = [
            np.transpose(img, (2, 0, 1)).astype(np.uint8) for img in images_hwc
        ]

        # For now, return base observation directly to avoid RPC message size limits
        # Images are still saved to disk for analysis
        # This avoids both the concatenation issue and the message size problem
        
        # Increment step counter
        self._step_count += 1
        
        # Return the original observation to avoid serialization issues
        return obs
    
    def _save_images_to_disk(self, images_hwc: list[np.ndarray], camera_names: list[str]) -> None:
        """Save rendered images to disk organized by camera."""
        try:
            for img, camera_name in zip(images_hwc, camera_names):
                if self._image_save_dir:
                    camera_dir = self._image_save_dir / camera_name
                    filename = f"ep{self._episode_count:04d}_step{self._step_count:06d}.png"
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
        self._metaworld_cache = {}
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
        envs = []

        if benchmark_spec.benchmark_name == "MT1":
            # MT1 has train tasks only (no test tasks), so we use train_tasks for evaluation
            # User must specify which environment when creating MT1
            env_name = benchmark_spec.config.get("env_name")
            if not env_name:
                raise ValueError("MT1 requires 'env_name' in benchmark config")

            mt1 = metaworld.MT1(env_name)
            tasks = [task for task in mt1.train_tasks if task.env_name == env_name]

            for i, task in enumerate(tasks):
                env = EnvSpec(
                    env_name=env_name,
                    benchmark_name="MT1",
                    provider="metaworld",
                    config={"task_idx": i, "task_data": task},
                    # Inherit settings from benchmark_spec
                    max_episode_steps=benchmark_spec.max_episode_steps,
                    render_mode=benchmark_spec.render_mode,
                    enable_image_obs=benchmark_spec.enable_image_obs,
                    image_size=benchmark_spec.image_size,
                    camera_attribute=benchmark_spec.camera_attribute,
                    camera_names=benchmark_spec.camera_names,
                )
                envs.append(env)

        elif benchmark_spec.benchmark_name == "MT10":
            # MT10 has train tasks only (no test tasks), so we use train_tasks for evaluation
            mt10 = metaworld.MT10()

            for env_name in mt10.train_classes.keys():
                tasks = [task for task in mt10.train_tasks if task.env_name == env_name]

                for i, task in enumerate(tasks):
                    env = EnvSpec(
                        env_name=env_name,
                        benchmark_name="MT10",
                        provider="metaworld",
                        config={"task_idx": i, "task_data": task},
                        # Inherit settings from benchmark_spec
                        max_episode_steps=benchmark_spec.max_episode_steps,
                        render_mode=benchmark_spec.render_mode,
                        enable_image_obs=benchmark_spec.enable_image_obs,
                        image_size=benchmark_spec.image_size,
                        camera_attribute=benchmark_spec.camera_attribute,
                        camera_names=benchmark_spec.camera_names,
                    )
                    envs.append(env)

        elif benchmark_spec.benchmark_name == "MT25":
            # MT25 has train tasks only (no test tasks), so we use train_tasks for evaluation
            mt25 = metaworld.MT25()

            for env_name in mt25.train_classes.keys():
                tasks = [task for task in mt25.train_tasks if task.env_name == env_name]

                for i, task in enumerate(tasks):
                    env = EnvSpec(
                        env_name=env_name,
                        benchmark_name="MT25",
                        provider="metaworld",
                        config={"task_idx": i, "task_data": task},
                        # Inherit settings from benchmark_spec
                        max_episode_steps=benchmark_spec.max_episode_steps,
                        render_mode=benchmark_spec.render_mode,
                        enable_image_obs=benchmark_spec.enable_image_obs,
                        image_size=benchmark_spec.image_size,
                        camera_attribute=benchmark_spec.camera_attribute,
                        camera_names=benchmark_spec.camera_names,
                    )
                    envs.append(env)

        elif benchmark_spec.benchmark_name == "MT50":
            # MT50 has train tasks only (no test tasks), so we use train_tasks for evaluation
            mt50 = metaworld.MT50()

            for env_name in mt50.train_classes.keys():
                tasks = [task for task in mt50.train_tasks if task.env_name == env_name]

                for i, task in enumerate(tasks):
                    env = EnvSpec(
                        env_name=env_name,
                        benchmark_name="MT50",
                        provider="metaworld",
                        config={"task_idx": i, "task_data": task},
                        # Inherit settings from benchmark_spec
                        max_episode_steps=benchmark_spec.max_episode_steps,
                        render_mode=benchmark_spec.render_mode,
                        enable_image_obs=benchmark_spec.enable_image_obs,
                        image_size=benchmark_spec.image_size,
                        camera_attribute=benchmark_spec.camera_attribute,
                        camera_names=benchmark_spec.camera_names,
                    )
                    envs.append(env)

        elif benchmark_spec.benchmark_name == "ML10":
            # ML10 has actual test tasks and test environments
            ml10 = metaworld.ML10()

            for env_name in ml10.test_classes.keys():
                tasks = [task for task in ml10.test_tasks if task.env_name == env_name]

                for i, task in enumerate(tasks):
                    env = EnvSpec(
                        env_name=env_name,
                        benchmark_name="ML10",
                        provider="metaworld",
                        config={"task_idx": i, "task_data": task},
                        # Inherit settings from benchmark_spec
                        max_episode_steps=benchmark_spec.max_episode_steps,
                        render_mode=benchmark_spec.render_mode,
                        enable_image_obs=benchmark_spec.enable_image_obs,
                        image_size=benchmark_spec.image_size,
                        camera_attribute=benchmark_spec.camera_attribute,
                        camera_names=benchmark_spec.camera_names,
                    )
                    envs.append(env)

        else:
            raise ValueError(
                f"Unsupported MetaWorld benchmark: {benchmark_spec.benchmark_name}"
            )

        logger.info(
            f"Found {len(envs)} test tasks across all environments for {benchmark_spec}"
        )
        return envs

    def make_env(self, env_spec: EnvSpec, save_images: bool = False, submission_id: str | None = None) -> gym.Env:
        """Create an environment for a specific environment spec."""
        if env_spec.provider == "metaworld":
            env = self._make_metaworld_env(env_spec)
        else:
            raise ValueError(f"Unsupported environment provider: {env_spec.provider}")

        # Apply wrappers based on env_spec configuration
        # Apply image observation wrapper if requested
        effective_render_mode = env_spec.render_mode
        if effective_render_mode == "human":
            effective_render_mode = "rgb_array"

        if (
            env_spec.enable_image_obs
            and effective_render_mode == "rgb_array"
            and env_spec.render_mode != "human"
        ):
            env = MultiViewImageObsWrapper(
                env,
                image_size=env_spec.image_size,
                camera_attribute=env_spec.camera_attribute,
                camera_names=env_spec.camera_names,
                save_images=save_images,
                image_save_dir="data" if save_images else None,
                submission_id=submission_id,
            )

        # Optional human display wrapper if user requested human rendering
        if env_spec.render_mode == "human":
            env = HumanRendering(env)

        env = gym.wrappers.TimeLimit(env, max_episode_steps=env_spec.max_episode_steps)

        return env

    def _make_metaworld_env(self, env_spec: EnvSpec) -> gym.Env:
        """Create a MetaWorld environment for the specified environment spec."""
        config = env_spec.config
        benchmark = env_spec.benchmark_name
        env_name = env_spec.env_name
        
        # Use appropriate render mode - None for headless, rgb_array for rendering
        render_mode = env_spec.render_mode if env_spec.enable_image_obs else None

        if benchmark == "MT1":
            # For MT1, create a single-task environment
            mt1 = metaworld.MT1(env_name)
            env = mt1.train_classes[env_name](render_mode=render_mode)
            # Use the specific task from the config
            task = config["task_data"]
            env.set_task(task)

        elif benchmark in ["MT10", "MT25", "MT50"]:
            # For MT benchmarks, use train_classes since test_classes is empty
            if benchmark == "MT10":
                mt_benchmark = metaworld.MT10()
            elif benchmark == "MT25":
                mt_benchmark = metaworld.MT25()
            else:  # MT50
                mt_benchmark = metaworld.MT50()

            env_cls = mt_benchmark.train_classes[env_name]
            env = env_cls(render_mode=render_mode)
            # Use the specific task from the config
            task = config["task_data"]
            env.set_task(task)

        elif benchmark == "ML10":
            # For ML10, use test_classes and test_tasks
            ml10 = metaworld.ML10()
            env_cls = ml10.test_classes[env_name]
            env = env_cls(render_mode=render_mode)
            # Use the specific task from the config
            task = config["task_data"]
            env.set_task(task)

        else:
            raise ValueError(f"Unsupported MetaWorld benchmark: {benchmark}")

        return env


# Global environment manager instance
env_manager = EnvManager()
