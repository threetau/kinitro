"""Worker that executes evaluation tasks using affinetes."""

import asyncio

import structlog

from kinitro.backend.models import Task, TaskResult
from kinitro.executor.config import ExecutorConfig
from kinitro.executor.env_loader import (
    build_load_kwargs,
    force_remove_container,
    load_and_warmup_env,
    run_evaluation,
)
from kinitro.types import AffinetesEnv, env_family_from_id

logger = structlog.get_logger()


class Worker:
    """
    Worker that executes evaluation tasks using affinetes.

    The worker loads an affinetes-managed evaluation environment
    and uses it to run evaluations against miner policy endpoints.
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config
        # Per-family eval environments: family -> affinetes env
        self._envs: dict[str, AffinetesEnv] = {}
        self._env_lock = asyncio.Lock()

    def _get_family(self, env_id: str) -> str:
        """Extract family from env_id (e.g., 'metaworld' from 'metaworld/pick-place-v3')."""
        return env_family_from_id(env_id)

    async def _get_eval_environment(self, env_id: str) -> AffinetesEnv:
        """
        Get or create the affinetes-managed eval environment for a given env_id.

        Each environment family (metaworld, genesis, etc.) uses a separate
        Docker container with family-specific dependencies.

        Args:
            env_id: Environment ID (e.g., 'metaworld/pick-place-v3')

        Returns:
            affinetes environment instance for the family
        """
        family = self._get_family(env_id)
        image = self.config.get_image_for_env(env_id)

        async with self._env_lock:
            # Check if we have a valid environment for this family
            if family in self._envs:
                try:
                    if self._envs[family].is_ready():
                        return self._envs[family]
                except Exception:
                    pass
                # Environment not ready, remove it
                del self._envs[family]

            logger.info(
                "loading_eval_environment",
                family=family,
                image=image,
                mode=self.config.eval_mode,
            )

            # Load eval environment via affinetes
            load_kwargs = build_load_kwargs(
                image=image,
                eval_mode=self.config.eval_mode,
                mem_limit=self.config.eval_mem_limit,
                executor_id=self.config.executor_id,
                family=family,
                hosts=self.config.eval_hosts,
                eval_timeout=self.config.eval_timeout,
                gpu_enabled=self.config.eval_gpu,
            )

            env = await load_and_warmup_env(family, image, load_kwargs)
            self._envs[family] = env
            return env

    async def execute_task(self, task: Task) -> TaskResult:
        """
        Execute a single evaluation task.

        Args:
            task: Task to execute

        Returns:
            TaskResult with execution outcome
        """
        logger.info(
            "executing_task",
            task_uuid=task.task_uuid,
            miner_uid=task.miner_uid,
            env_id=task.env_id,
            seed=task.seed,
        )

        try:
            env = await self._get_eval_environment(task.env_id)

            task_result = await run_evaluation(
                env=env,
                task=task,
                max_timesteps=self.config.max_timesteps,
                action_timeout=self.config.action_timeout,
                use_images=self.config.use_images,
                eval_timeout=self.config.eval_timeout,
            )

            logger.info(
                "task_executed",
                task_uuid=task.task_uuid,
                success=task_result.success,
                score=task_result.score,
            )

            return task_result

        except TimeoutError:
            logger.warning("task_timeout", task_uuid=task.task_uuid)
            return TaskResult(
                task_uuid=task.task_uuid,
                success=False,
                score=0.0,
                error="Evaluation timeout",
            )

        except Exception as e:
            logger.error("task_error", task_uuid=task.task_uuid, error=str(e))
            return TaskResult(
                task_uuid=task.task_uuid,
                success=False,
                score=0.0,
                error=str(e),
            )

    async def execute_batch(self, tasks: list[Task]) -> list[TaskResult]:
        """
        Execute a batch of tasks.

        Args:
            tasks: List of tasks to execute

        Returns:
            List of task results
        """
        results = []
        for task in tasks:
            result = await self.execute_task(task)
            results.append(result)
        return results

    async def cleanup(self) -> None:
        """Cleanup all eval environments."""
        async with self._env_lock:
            for family, env in list(self._envs.items()):
                try:
                    await env.cleanup()
                    logger.info("cleanup_environment", family=family)
                except Exception as e:
                    logger.warning("cleanup_error", family=family, error=str(e))
            self._envs.clear()

    def force_cleanup(self) -> None:
        """Force cleanup by killing docker containers directly."""
        # Clean up all family-specific containers
        for family in list(self._envs.keys()):
            container_name = f"kinitro-eval-{self.config.executor_id}-{family}"
            logger.info("force_cleanup_container", container=container_name, family=family)
            force_remove_container(container_name)

        # Also try to clean up any containers matching the executor pattern
        # in case there are orphaned containers from previous runs
        for family in self.config.eval_images.keys():
            container_name = f"kinitro-eval-{self.config.executor_id}-{family}"
            force_remove_container(container_name)

        self._envs.clear()
