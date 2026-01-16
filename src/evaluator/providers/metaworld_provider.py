"""
Metaworld environment provider implementation.

Provides MetaWorld MT1/MT10/MT50 and ML10/ML45 benchmarks.
"""

import logging
import random
from typing import Any, Dict, List, Type

import gymnasium as gym
from gymnasium import ObservationWrapper
from metaworld.wrappers import OneHotWrapper

from .metaworld import load_benchmark_definition
from .registry import BenchmarkSpec, EnvironmentProvider, EnvSpec

logger = logging.getLogger(__name__)

DEFAULT_MAX_EPISODE_STEPS = 10
DEFAULT_EPISODES_PER_TASK = 3
DEFAULT_TASKS_PER_ENV = 5


class MetaworldProvider(EnvironmentProvider):
    """
    Metaworld environment provider.

    Provides access to MetaWorld benchmarks:
    - MT1, MT10, MT25, MT50 (multi-task)
    - ML10, ML25, ML45 (meta-learning)
    """

    @property
    def name(self) -> str:
        return "metaworld"

    def get_benchmark_specs(self, config: Dict[str, Any]) -> List[BenchmarkSpec]:
        """Get benchmark specifications from config."""
        benchmark_name = config.get("benchmark_name", "MT1")
        return [
            BenchmarkSpec(
                provider=self.name,
                benchmark_name=benchmark_name,
                config=config,
                render_mode=config.get("render_mode", "rgb_array"),
                camera_names=tuple(config.get("camera_names", ["corner"])),
                camera_attribute=config.get("camera_attribute", "camera_name"),
            )
        ]

    def get_env_specs(self, benchmark_spec: BenchmarkSpec) -> List[EnvSpec]:
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

        env_specs: List[EnvSpec] = []
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
                    provider=self.name,
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

    def make_env(
        self,
        spec: EnvSpec,
        submission_id: str | None = None,
        save_images: bool = False,
    ) -> gym.Env:
        """Create a MetaWorld environment for the specified environment spec."""
        # Import here to avoid circular imports
        from ..rollout.envs import MetaworldObsWrapper  # noqa: PLC0415

        config = spec.config
        env_cls = config.get("env_cls")
        if env_cls is None:
            raise ValueError("EnvSpec config missing 'env_cls' for MetaWorld env")

        render_mode = spec.render_mode
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
            camera_attribute=spec.camera_attribute,
            camera_names=spec.camera_names,
            save_images=save_images,
            image_save_dir="data" if save_images else None,
            submission_id=submission_id,
        )

        env = gym.wrappers.TimeLimit(env, max_episode_steps=spec.max_episode_steps)

        return env

    def get_observation_wrapper(self) -> Type[ObservationWrapper] | None:
        """Get the observation wrapper class for MetaWorld."""
        from ..rollout.envs import MetaworldObsWrapper  # noqa: PLC0415

        return MetaworldObsWrapper
