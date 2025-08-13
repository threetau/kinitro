"""
Environment wrappers and utilities

We use Gymnasium-compatible interfaces and expose a tiny factory that can
instantiate a single-task environment.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    """
    Specification for an environment.
    """

    env_name: str = "push-v3"
    env_provider: str = "Meta-World/MT1"
    max_episode_steps: int = 200
    render_mode: str | None = "rgb_array"


class MultiviewObsWrapper(ObservationWrapper):
    """
    Wraps an environment to return a dict observation with:
      - "base": the original observation
      - "observation.image": a CHW uint8 image captured from the topview camera
      - "observation.image2": a CHW uint8 image captured from the corner3 camera
      - "observation.image3": a CHW uint8 image captured from the gripper POV camera

    The image is resized to a fixed shape for a stable observation space.
    """

    def __init__(self, env: gym.Env, image_size: tuple[int, int] = (64, 64)):
        super().__init__(env)
        self._img_h, self._img_w = int(image_size[0]), int(image_size[1])
        # Define observation space as Dict of original space and image box (C,H,W)
        self.observation_space = DictSpace(
            {
                "base": env.observation_space,
                "observation.image": Box(
                    low=0,
                    high=255,
                    shape=(3, self._img_h, self._img_w),
                    dtype=np.uint8,
                ),
                "observation.image2": Box(
                    low=0,
                    high=255,
                    shape=(3, self._img_h, self._img_w),
                    dtype=np.uint8,
                ),
                "observation.image3": Box(
                    low=0,
                    high=255,
                    shape=(3, self._img_h, self._img_w),
                    dtype=np.uint8,
                ),
            }
        )

    def observation(self, obs):  # type: ignore[override]
        # Env must be created with render_mode="rgb_array" and camera_name="topview"
        self.env.camera_name = "topview"
        topview_img_hwc = self.env.render()  # (H, W, 3), uint8
        self.env.camera_name = "corner3"
        corner3_img_hwc = self.env.render()  # (H, W, 3), uint8
        self.env.camera_name = "gripperPOV"
        gripper_pov_img_hwc = self.env.render()  # (H, W, 3), uint8

        # convert the images to CHW
        topview_img_chw = np.transpose(topview_img_hwc, (2, 0, 1)).astype(np.uint8)
        corner3_img_chw = np.transpose(corner3_img_hwc, (2, 0, 1)).astype(np.uint8)
        gripper_pov_img_chw = np.transpose(gripper_pov_img_hwc, (2, 0, 1)).astype(
            np.uint8
        )

        # # then batch them into a 4d tensor
        # images = np.stack(
        #     [topview_img_chw, corner3_img_chw, gripper_pov_img_chw], axis=0
        # )

        # change it back to topview camera
        self.env.camera_name = "topview"

        return {
            "base": obs,
            "observation.image": topview_img_chw,
            "observation.image2": corner3_img_chw,
            "observation.image3": gripper_pov_img_chw,
        }


def make_env(spec: EnvSpec) -> gym.Env:
    env = gym.make(
        spec.env_provider,
        env_name=spec.env_name,
        render_mode=spec.render_mode,
        camera_name="topview",
    )
    # Always wrap with topview observation dict
    env = MultiviewObsWrapper(env, image_size=(64, 64))
    # Optional human display wrapper if user requested human rendering
    if spec.render_mode == "human":
        env = HumanRendering(env)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=spec.max_episode_steps)
    return env
