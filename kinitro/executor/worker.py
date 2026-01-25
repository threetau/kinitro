"""Worker that executes evaluation tasks using affinetes."""

import asyncio
import os
from dataclasses import dataclass

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
        self._env = None
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
            task_uuid=task.task_uuid,
            miner_uid=task.miner_uid,
            env_id=task.env_id,
            seed=task.seed,
        )

        # Perform spot-check verification if enabled and not yet verified this miner
        await self._maybe_verify_miner(task)

        try:
            env = await self._get_eval_environment()

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
        """Cleanup eval environment and verifier."""
        async with self._env_lock:
            if self._env is not None:
                try:
                    await self._env.cleanup()
                except Exception as e:
                    logger.warning("cleanup_error", error=str(e))
                self._env = None

        # Cleanup verifier
        if self._verifier is not None:
            self._verifier.cleanup()

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
