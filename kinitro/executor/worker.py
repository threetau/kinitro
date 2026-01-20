"""Worker that executes evaluation tasks using affinetes."""

import asyncio
import os
from dataclasses import dataclass

import structlog

from kinitro.backend.models import Task, TaskResult
from kinitro.executor.config import ExecutorConfig

logger = structlog.get_logger()


@dataclass
class EvalEnvConfig:
    """Configuration for the evaluation environment."""

    image: str
    mode: str
    mem_limit: str
    hosts: list[str]
    max_timesteps: int
    action_timeout: float
    eval_timeout: int
    use_images: bool


class Worker:
    """
    Worker that executes evaluation tasks using affinetes.

    The worker loads an affinetes-managed evaluation environment
    and uses it to run evaluations against miner policy endpoints.
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config
        self._env = None
        self._env_lock = asyncio.Lock()

    async def _get_eval_environment(self):
        """Get or create the affinetes-managed eval environment."""
        async with self._env_lock:
            if self._env is not None:
                try:
                    if self._env.is_ready():
                        return self._env
                except Exception:
                    pass
                self._env = None

            try:
                import affinetes as af_env
            except ImportError:
                raise ImportError(
                    "affinetes is required for evaluation. Install with: pip install affinetes"
                )

            logger.info(
                "loading_eval_environment",
                image=self.config.eval_image,
                mode=self.config.eval_mode,
            )

            # Load eval environment via affinetes
            load_kwargs = {
                "image": self.config.eval_image,
                "mode": self.config.eval_mode,
                "mem_limit": self.config.eval_mem_limit,
                "env_vars": {
                    "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
                },
                "pull": True,
            }

            if self.config.eval_mode == "docker":
                load_kwargs.update(
                    {
                        "hosts": self.config.eval_hosts,
                        "container_name": f"kinitro-eval-{self.config.executor_id}",
                        "force_recreate": True,
                    }
                )
            elif self.config.eval_mode == "basilica":
                load_kwargs.update(
                    {
                        "cpu_limit": "2000m",
                        "ttl_buffer": self.config.eval_timeout + 60,
                    }
                )

            self._env = await asyncio.to_thread(af_env.load_env, **load_kwargs)

            # Warm-up call
            logger.info("warmup_call_starting")
            try:
                await self._env.list_environments()
                logger.info("warmup_call_succeeded")
            except Exception as e:
                logger.info(
                    "warmup_call_absorbed_expected_error",
                    error=str(e)[:100],
                )

            logger.info("eval_environment_loaded")
            return self._env

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
            task_id=task.id,
            miner_uid=task.miner_uid,
            env_id=task.env_id,
            seed=task.task_id,
        )

        try:
            env = await self._get_eval_environment()

            result = await env.evaluate(
                task_id=task.task_id,
                model=f"miner-{task.miner_uid}",  # Identifier for logging
                base_url=task.miner_endpoint,
                env_id=task.env_id,
                max_timesteps=self.config.max_timesteps,
                action_timeout=self.config.action_timeout,
                use_images=self.config.use_images,
                _timeout=self.config.eval_timeout,
            )

            success = result.get("success", False)
            score = result.get("score", 0.0)
            extra = result.get("extra", {})

            logger.info(
                "task_executed",
                task_id=task.id,
                success=success,
                score=score,
            )

            return TaskResult(
                task_id=task.id,
                success=success,
                score=score,
                total_reward=extra.get("total_reward", 0.0),
                timesteps=extra.get("timesteps", 0),
                error=None,
            )

        except TimeoutError:
            logger.warning("task_timeout", task_id=task.id)
            return TaskResult(
                task_id=task.id,
                success=False,
                score=0.0,
                error="Evaluation timeout",
            )

        except Exception as e:
            logger.error("task_error", task_id=task.id, error=str(e))
            return TaskResult(
                task_id=task.id,
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
        """Cleanup eval environment."""
        async with self._env_lock:
            if self._env is not None:
                try:
                    await self._env.cleanup()
                except Exception as e:
                    logger.warning("cleanup_error", error=str(e))
                self._env = None

    def force_cleanup(self) -> None:
        """Force cleanup by killing docker container directly."""
        import subprocess

        container_name = f"kinitro-eval-{self.config.executor_id}"
        logger.info("force_cleanup_container", container=container_name)

        try:
            subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
        except Exception as e:
            logger.warning("docker_cleanup_failed", error=str(e))

        self._env = None
