"""
Swarm environment provider implementation.

Provides PyBullet drone simulation environments.
"""

import logging
import random
from typing import Any, Dict, List, Type

import gymnasium as gym
from gymnasium import ObservationWrapper

from . import swarm as swarm_module
from .registry import BenchmarkSpec, EnvironmentProvider, EnvSpec

logger = logging.getLogger(__name__)

DEFAULT_EPISODES_PER_TASK = 3
DEFAULT_TASKS_PER_ENV = 5


class SwarmProvider(EnvironmentProvider):
    """
    Swarm PyBullet environment provider.

    Provides access to drone simulation environments using PyBullet.
    """

    @property
    def name(self) -> str:
        return "swarm"

    def get_benchmark_specs(self, config: Dict[str, Any]) -> List[BenchmarkSpec]:
        """Get benchmark specifications from config."""
        benchmark_name = config.get("benchmark_name", "swarm-default")
        return [
            BenchmarkSpec(
                provider=self.name,
                benchmark_name=benchmark_name,
                config=config,
                render_mode=None,  # Swarm doesn't use standard render mode
                camera_names=tuple(),
                camera_attribute=None,
            )
        ]

    def get_env_specs(self, benchmark_spec: BenchmarkSpec) -> List[EnvSpec]:
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

        sim_dt = float(benchmark_spec.config.get("sim_dt", swarm_module.SIM_DT))
        horizon = float(benchmark_spec.config.get("horizon", swarm_module.HORIZON_SEC))
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

        env_specs: List[EnvSpec] = []
        for task_idx in range(tasks_per_env):
            seed = task_seed + task_idx if task_seed is not None else None
            task = swarm_module.random_task(
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
                    provider=self.name,
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
        spec: EnvSpec,
        submission_id: str | None = None,
        save_images: bool = False,
    ) -> gym.Env:
        """Create a Swarm PyBullet environment from the provided task config."""
        # Import here to avoid circular imports
        from ..rollout.envs import DroneObsWrapper  # noqa: PLC0415

        config = spec.config
        task = config.get("task")
        seed = config.get("task_seed")
        challenge_type = config.get("challenge_type")

        if task is None:
            task = swarm_module.random_task(
                sim_dt=swarm_module.SIM_DT,
                horizon=swarm_module.HORIZON_SEC,
                seed=seed,
                payload=bool(config.get("payload_mode", False)),
                challenge_type=challenge_type,
            )

        gui = bool(config.get("gui", False))
        env = swarm_module.make_env(task, gui=gui)
        env = DroneObsWrapper(env)

        if spec.max_episode_steps:
            env = gym.wrappers.TimeLimit(env, max_episode_steps=spec.max_episode_steps)

        return env

    def get_observation_wrapper(self) -> Type[ObservationWrapper] | None:
        """Get the observation wrapper class for Swarm."""
        from ..rollout.envs import DroneObsWrapper  # noqa: PLC0415

        return DroneObsWrapper
