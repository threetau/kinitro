"""
Environment wrappers and utilities for MetaWorld.

We use Gymnasium-compatible interfaces and expose a tiny factory that can
instantiate a single-task MetaWorld environment such as "push-v2" or a
predefined benchmark like MT10. For the initial validator MVP, we use single
task environments to keep dependency and runtime overhead small.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import gymnasium as gym  # type: ignore
import numpy as np


@dataclass
class EnvSpec:
    task_name: str = "push-v3"
    max_episode_steps: int = 200


def make_env(spec: EnvSpec) -> gym.Env:
    try:
        metaworld = importlib.import_module("metaworld")
    except Exception as exc:  # pragma: no cover - provide clear error in runtime envs without metaworld
        raise RuntimeError(
            "metaworld is not installed or failed to import. Ensure it is present in the environment."
        ) from exc

    # Use MetaWorld's Task API to construct a single-task env (V3 names).
    ml = metaworld.ML1(spec.task_name)  # type: ignore[attr-defined]
    env_cls = ml.train_classes[spec.task_name]
    env: gym.Env = env_cls()
    task = ml.train_tasks[0]
    env.set_task(task)

    # Ensure Gymnasium-style API with time limit.
    env = gym.wrappers.TimeLimit(env, max_episode_steps=spec.max_episode_steps)
    return env


def obs_size(env: gym.Env) -> int:
    space = env.observation_space
    if hasattr(space, "shape") and space.shape is not None:
        size = int(np.prod(space.shape))
        return size
    raise ValueError("Unsupported observation space for simple evaluator")


def action_size(env: gym.Env) -> int:
    space = env.action_space
    if hasattr(space, "shape") and space.shape is not None:
        size = int(np.prod(space.shape))
        return size
    raise ValueError("Unsupported action space for simple evaluator")


