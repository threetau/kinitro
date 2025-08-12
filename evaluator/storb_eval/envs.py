"""
Environment wrappers and utilities

We use Gymnasium-compatible interfaces and expose a tiny factory that can
instantiate a single-task environment.
"""

from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym  # type: ignore
import metaworld  # noqa: F401


@dataclass
class EnvSpec:
    """
    Specification for an environment.
    """

    env_name: str = "push-v3"
    env_provider: str = "Meta-World/MT1"
    max_episode_steps: int = 200
    render_mode: str | None = None


def make_env(spec: EnvSpec) -> gym.Env:
    env = gym.make(
        spec.env_provider,
        env_name=spec.env_name,
        render_mode=spec.render_mode,
    )
    env = gym.wrappers.TimeLimit(env, max_episode_steps=spec.max_episode_steps)

    return env
