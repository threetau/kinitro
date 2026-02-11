"""Worker that executes evaluation tasks using affinetes."""

import asyncio
import subprocess
from dataclasses import dataclass
from typing import Any

import affinetes as af_env
import docker.types
import structlog

from kinitro.backend.models import Task, TaskResult
from kinitro.executor.config import ExecutorConfig
from kinitro.executor.verification import PolicyVerifier, VerificationResult

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
    It also performs spot-check verification to ensure deployed models
    match what miners uploaded to HuggingFace.
    """

    def __init__(self, config: ExecutorConfig):
        self.config = config
        # Per-family eval environments: family -> affinetes env
        self._envs: dict[str, Any] = {}
        self._env_lock = asyncio.Lock()

        # Initialize verifier if enabled
        self._verifier: PolicyVerifier | None = None
        if config.verification_enabled:
            self._verifier = PolicyVerifier(
                verification_rate=config.verification_rate,
                tolerance=config.verification_tolerance,
                num_samples=config.verification_samples,
                cache_dir=config.verification_cache_dir,
                max_repo_size_gb=config.verification_max_repo_size_gb,
            )
            logger.info(
                "verification_enabled",
                rate=config.verification_rate,
                tolerance=config.verification_tolerance,
                samples=config.verification_samples,
                max_repo_size_gb=config.verification_max_repo_size_gb,
            )

        # Track verification results for reporting
        self._verification_results: list[VerificationResult] = []
        self._verified_miners: set[str] = set()  # Track which miners we've verified this cycle

    def _get_family(self, env_id: str) -> str:
        """Extract family from env_id (e.g., 'metaworld' from 'metaworld/pick-place-v3')."""
        return env_id.split("/")[0] if "/" in env_id else env_id

    async def _get_eval_environment(self, env_id: str):
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
            load_kwargs = {
                "image": image,
                "mode": self.config.eval_mode,
                "mem_limit": self.config.eval_mem_limit,
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
                if self.config.eval_gpu:
                    load_kwargs["device_requests"] = [
                        docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                    ]
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

        # Perform spot-check verification if enabled and not yet verified this miner
        await self._maybe_verify_miner(task)

        try:
            env = await self._get_eval_environment(task.env_id)

            result = await env.evaluate(
                task_id=task.seed,  # Use seed for environment reproducibility
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

    async def _maybe_verify_miner(self, task: Task) -> None:
        """
        Perform spot-check verification if conditions are met.

        Verification is performed if:
        - Verification is enabled
        - Miner has repo and revision info
        - Miner hasn't been verified yet this cycle
        - Random chance based on verification_rate
        """
        if self._verifier is None:
            return

        # Skip if missing repo/revision info
        if not task.miner_repo or not task.miner_revision:
            logger.debug(
                "verification_skipped_no_repo",
                miner_uid=task.miner_uid,
            )
            return

        # Only verify each miner once per cycle
        miner_key = f"{task.miner_uid}:{task.miner_revision}"
        if miner_key in self._verified_miners:
            return

        # Random spot-check
        if not self._verifier.should_verify():
            return

        # Mark as verified (even if verification fails, don't retry)
        self._verified_miners.add(miner_key)

        logger.info(
            "verification_triggered",
            miner_uid=task.miner_uid,
            repo=task.miner_repo,
            revision=task.miner_revision[:12] if task.miner_revision else None,
        )

        try:
            result = await self._verifier.verify_miner(
                miner_uid=task.miner_uid,
                miner_hotkey=task.miner_hotkey,
                repo=task.miner_repo,
                revision=task.miner_revision,
                endpoint=task.miner_endpoint,
            )

            self._verification_results.append(result)

            if not result.verified:
                logger.warning(
                    "verification_mismatch",
                    miner_uid=task.miner_uid,
                    match_score=result.match_score,
                    error=result.error,
                )
            else:
                logger.info(
                    "verification_passed",
                    miner_uid=task.miner_uid,
                    match_score=result.match_score,
                )

        except Exception as e:
            logger.error(
                "verification_error",
                miner_uid=task.miner_uid,
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

    def get_verification_results(self) -> list[VerificationResult]:
        """Get all verification results from this worker."""
        return self._verification_results.copy()

    def get_failed_verifications(self) -> list[VerificationResult]:
        """Get verification results where miner failed verification."""
        return [r for r in self._verification_results if not r.verified]

    def reset_verification_state(self) -> None:
        """Reset verification state for a new evaluation cycle."""
        self._verified_miners.clear()
        self._verification_results.clear()

    async def cleanup(self) -> None:
        """Cleanup all eval environments and verifier."""
        async with self._env_lock:
            for family, env in list(self._envs.items()):
                try:
                    await env.cleanup()
                    logger.info("cleanup_environment", family=family)
                except Exception as e:
                    logger.warning("cleanup_error", family=family, error=str(e))
            self._envs.clear()

        # Cleanup verifier
        if self._verifier is not None:
            self._verifier.cleanup()

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
