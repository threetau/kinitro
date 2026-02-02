"""Worker that executes evaluation tasks using affinetes."""

import asyncio
import os
import subprocess
from dataclasses import dataclass
from typing import Any

import affinetes as af_env
import structlog

from kinitro.backend.models import Task, TaskResult
from kinitro.executor.config import ExecutorConfig
from kinitro.executor.miner_deployment import MinerDeploymentConfig, MinerDeploymentManager

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
    It manages Basilica deployments for miners on-demand.
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config
        # Per-family eval environments: family -> affinetes env
        self._envs: dict[str, Any] = {}
        self._env_lock = asyncio.Lock()

        # Initialize miner deployment manager if enabled
        self._deployment_manager: MinerDeploymentManager | None = None
        if config.miner_deployment_enabled:
            # For basilica mode, we need an API token
            if config.eval_mode == "basilica" and not config.miner_basilica_api_token:
                logger.warning(
                    "miner_deployment_manager_disabled",
                    reason="basilica mode requires miner_basilica_api_token",
                )
            else:
                deployment_config = MinerDeploymentConfig(
                    image=config.miner_deployment_image,
                    mode=config.eval_mode,  # Use same mode as eval environments
                    basilica_api_token=config.miner_basilica_api_token,
                    hf_token=config.miner_hf_token,
                    ttl_seconds=config.miner_deployment_ttl_seconds,
                    warmup_timeout=config.miner_deployment_warmup_timeout,
                    mem_limit=config.eval_mem_limit,
                    gpu_count=config.miner_deployment_gpu_count,
                    min_gpu_memory_gb=config.miner_deployment_min_gpu_memory_gb,
                    cpu=config.miner_deployment_cpu,
                    memory=config.miner_deployment_memory,
                )
                self._deployment_manager = MinerDeploymentManager(deployment_config)
                logger.info(
                    "miner_deployment_manager_enabled",
                    mode=config.eval_mode,
                    image=config.miner_deployment_image,
                    ttl_seconds=config.miner_deployment_ttl_seconds,
                )

    def _get_family(self, env_id: str) -> str:
        """Extract family from env_id (e.g., 'metaworld' from 'metaworld/pick-place-v3')."""
        return env_id.split("/")[0] if "/" in env_id else env_id

    async def _get_eval_environment(self, env_id: str):
        """
        Get or create the affinetes-managed eval environment for a given env_id.

        Each environment family (metaworld, procthor, etc.) uses a separate
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
            load_kwargs = {
                "image": image,
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
                        # Unique container name per family
                        "container_name": f"kinitro-eval-{self.config.executor_id}-{family}",
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

            env = await asyncio.to_thread(af_env.load_env, **load_kwargs)

            # Warm-up call
            logger.info("warmup_call_starting", family=family)
            try:
                await env.list_environments()
                logger.info("warmup_call_succeeded", family=family)
            except Exception as e:
                logger.info(
                    "warmup_call_absorbed_expected_error",
                    family=family,
                    error=str(e)[:100],
                )

            logger.info("eval_environment_loaded", family=family, image=image)
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

        # Resolve miner endpoint via deployment manager or use provided endpoint
        miner_endpoint: str | None = None
        if self._deployment_manager and task.miner_repo and task.miner_revision:
            try:
                miner_endpoint = await self._deployment_manager.get_or_create_deployment(
                    miner_uid=task.miner_uid,
                    repo=task.miner_repo,
                    revision=task.miner_revision,
                )
            except Exception as e:
                logger.error(
                    "deployment_creation_failed",
                    task_uuid=task.task_uuid,
                    miner_uid=task.miner_uid,
                    repo=task.miner_repo,
                    error=str(e),
                )
                return TaskResult(
                    task_uuid=task.task_uuid,
                    success=False,
                    score=0.0,
                    error=f"Deployment failed: {e}",
                )
        else:
            # Fallback for legacy tasks with pre-set endpoint
            miner_endpoint = task.miner_endpoint

        if not miner_endpoint:
            logger.error(
                "no_miner_endpoint",
                task_uuid=task.task_uuid,
                miner_uid=task.miner_uid,
            )
            return TaskResult(
                task_uuid=task.task_uuid,
                success=False,
                score=0.0,
                error="No miner endpoint available",
            )

        try:
            env = await self._get_eval_environment(task.env_id)

            result = await env.evaluate(
                task_id=task.seed,  # Use seed for environment reproducibility
                model=f"miner-{task.miner_uid}",  # Identifier for logging
                base_url=miner_endpoint,
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
                task_uuid=task.task_uuid,
                success=success,
                score=score,
            )

            return TaskResult(
                task_uuid=task.task_uuid,
                success=success,
                score=score,
                total_reward=extra.get("total_reward", 0.0),
                timesteps=extra.get("timesteps", 0),
                error=None,
            )

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

        # Cleanup expired deployments after batch
        if self._deployment_manager:
            try:
                cleaned = await self._deployment_manager.cleanup_expired()
                if cleaned > 0:
                    logger.info("deployments_cleaned_after_batch", count=cleaned)
            except Exception as e:
                logger.warning("deployment_cleanup_error", error=str(e))

        return results

    async def cleanup(self) -> None:
        """Cleanup all eval environments and miner deployments."""
        async with self._env_lock:
            for family, env in list(self._envs.items()):
                try:
                    await env.cleanup()
                    logger.info("cleanup_environment", family=family)
                except Exception as e:
                    logger.warning("cleanup_error", family=family, error=str(e))
            self._envs.clear()

        # Shutdown miner deployments
        if self._deployment_manager is not None:
            await self._deployment_manager.shutdown()

    def force_cleanup(self) -> None:
        """Force cleanup by killing docker containers directly."""
        # Clean up all family-specific containers
        for family in list(self._envs.keys()):
            container_name = f"kinitro-eval-{self.config.executor_id}-{family}"
            logger.info("force_cleanup_container", container=container_name, family=family)

            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except Exception as e:
                logger.warning("docker_cleanup_failed", family=family, error=str(e))

        # Also try to clean up any containers matching the executor pattern
        # in case there are orphaned containers from previous runs
        for family in self.config.eval_images.keys():
            container_name = f"kinitro-eval-{self.config.executor_id}-{family}"
            try:
                subprocess.run(
                    ["docker", "rm", "-f", container_name],
                    capture_output=True,
                    text=True,
                    timeout=5,
                    check=False,
                )
            except Exception:
                pass  # Ignore errors for containers that don't exist

        self._envs.clear()
