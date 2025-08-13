"""
Environment wrappers and utilities

We use Gymnasium-compatible interfaces and expose a tiny factory that can
instantiate a single-task environment. This module is intentionally generic so
that any Gymnasium-like API can be supported by configuring `EnvSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gymnasium as gym  # type: ignore
import metaworld  # noqa: F401
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box
from gymnasium.spaces import Dict as DictSpace
from gymnasium.wrappers import HumanRendering

# Constant ripped from lerobot.constants
OBS_IMAGES = "observation.images"


@dataclass
class EnvSpec:
    """Specification for an environment.

    This structure is designed to be provider-agnostic. There are two ways to
    specify an environment:
      1) Generic Gym ID via `env_id` and optional `make_kwargs`
      2) Backward-compatible Meta-World style via `env_provider` + `env_name`
    """

    # Generic Gymnasium ID, e.g. "CartPole-v1", "Meta-World/MT1"
    env_id: str | None = None

    # Backward-compat (used when `env_id` is None)
    env_name: str | None = "push-v3"
    env_provider: str | None = "Meta-World/MT1"

    # Additional kwargs forwarded to gym.make
    make_kwargs: dict[str, object] = field(default_factory=dict)

    # Runtime controls
    max_episode_steps: int = 200
    render_mode: str | None = "rgb_array"

    # Observation capture options
    enable_image_obs: bool = True
    image_size: tuple[int, int] = (64, 64)
    # If the env exposes a camera selector attribute (e.g., "camera_name"), set it
    # to each of the provided names to collect multi-view renders. Leave as None to
    # skip camera switching and just capture a single render per step.
    camera_attribute: str | None = "camera_name"
    camera_names: tuple[str, ...] = ("topview", "corner3", "gripperPOV")


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
    """

    def __init__(
        self,
        env: gym.Env,
        image_size: tuple[int, int] = (64, 64),
        camera_attribute: str | None = "camera_name",
        camera_names: tuple[str, ...] = ("topview", "corner3", "gripperPOV"),
    ):
        super().__init__(env)
        self._img_h, self._img_w = int(image_size[0]), int(image_size[1])
        self._camera_attribute = camera_attribute
        self._camera_names = tuple(camera_names) if camera_names else tuple()

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
        frame = self.env.render()
        if frame is None:
            raise RuntimeError(
                "Environment returned None from render(); ensure render_mode='rgb_array'"
            )
        return np.asarray(frame)

    def observation(self, obs):  # type: ignore[override]
        images_hwc: list[np.ndarray] = []

        if self._camera_attribute is not None and hasattr(
            self.env, self._camera_attribute
        ) and len(self._camera_names) > 0:
            # Save current camera to restore later
            try:
                current_cam = getattr(self.env, self._camera_attribute)
            except Exception:
                current_cam = None

            for cam in self._camera_names:
                self._set_camera_if_possible(cam)
                images_hwc.append(self._render_rgb_array())

            # Restore original camera
            self._set_camera_if_possible(current_cam)
        else:
            # Single-view capture without camera switching
            images_hwc.append(self._render_rgb_array())

        # Convert images to CHW uint8
        images_chw = [np.transpose(img, (2, 0, 1)).astype(np.uint8) for img in images_hwc]

        # Build dict observation
        out: dict[str, object] = {"base": obs}
        for idx, img in enumerate(images_chw):
            key = "observation.image" if idx == 0 else f"observation.image{idx + 1}"
            out[key] = img
        return out


def make_env(spec: EnvSpec) -> gym.Env:
    """Create an environment according to the provided `EnvSpec`.

    This function prefers `spec.env_id` if provided; otherwise, it falls back to
    the previous `env_provider` + `env_name` behavior for Meta-World.
    """
    make_kwargs = dict(spec.make_kwargs)
    # If user requested human rendering, create the base env with rgb_array so
    # HumanRendering wrapper can display frames.
    base_render_mode = spec.render_mode
    if base_render_mode == "human":
        base_render_mode = "rgb_array"
    if base_render_mode is not None:
        make_kwargs.setdefault("render_mode", base_render_mode)

    if spec.env_id is not None:
        env = gym.make(spec.env_id, **make_kwargs)
    else:
        # Backward-compat Meta-World usage
        env_provider = spec.env_provider or "Meta-World/MT1"
        env_name = spec.env_name
        # Only Meta-World envs accept camera_name; inject a default if desired
        if (
            spec.camera_attribute == "camera_name"
            and "camera_name" not in make_kwargs
            and len(spec.camera_names) > 0
        ):
            make_kwargs["camera_name"] = spec.camera_names[0]
        env = gym.make(env_provider, env_name=env_name, **make_kwargs)

    # Optionally add image observations
    # Only enable image capture when using rgb_array rendering, and skip if the
    # user asked for human display mode.
    effective_render_mode = make_kwargs.get("render_mode", base_render_mode)
    if (
        spec.enable_image_obs
        and effective_render_mode == "rgb_array"
        and spec.render_mode != "human"
    ):
        env = MultiViewImageObsWrapper(
            env,
            image_size=spec.image_size,
            camera_attribute=spec.camera_attribute,
            camera_names=spec.camera_names,
        )

    # Optional human display wrapper if user requested human rendering
    if spec.render_mode == "human":
        env = HumanRendering(env)

    env = gym.wrappers.TimeLimit(env, max_episode_steps=spec.max_episode_steps)
    return env
