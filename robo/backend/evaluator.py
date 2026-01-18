"""
Robotics Evaluator using Affinetes

This module provides the RoboticsEvaluator class that uses affinetes
to run the robotics evaluation environment and query miner policy endpoints.

The evaluation flow:
1. Load eval environment via affinetes (contains MuJoCo + MetaWorld)
2. For each miner, call env.evaluate() with their endpoint URL
3. The eval environment handles simulation and miner HTTP calls
4. Collect scores and return results
"""

import asyncio
import os
from dataclasses import dataclass, field
from typing import Any, Optional

import structlog

from ..chain.commitments import MinerCommitment
from ..environments.registry import get_all_environment_ids

logger = structlog.get_logger()

# Global flag for shutdown - checked by evaluator
_shutdown_requested = False


def request_shutdown():
    """Signal that shutdown has been requested."""
    global _shutdown_requested
    _shutdown_requested = True


def is_shutdown_requested() -> bool:
    """Check if shutdown has been requested."""
    return _shutdown_requested


@dataclass
class EvaluatorConfig:
    """Configuration for the robotics evaluator."""
    
    # Docker image for eval environment
    eval_image: str = "robo-subnet/eval-env:v1"
    
    # Affinetes mode: "docker" or "basilica"
    mode: str = "docker"
    
    # Memory limit for eval container
    mem_limit: str = "8g"
    
    # Hosts for Docker mode (can include SSH remotes)
    hosts: list[str] = field(default_factory=lambda: ["localhost"])
    
    # Timeout for individual evaluation (seconds)
    eval_timeout: int = 300
    
    # Max timesteps per episode
    max_timesteps: int = 500
    
    # Action timeout per step (seconds)
    action_timeout: float = 0.5
    
    # Whether to include camera images
    use_images: bool = True


class RoboticsEvaluator:
    """
    Evaluates miner policies using affinetes-managed eval environment.
    
    The eval environment runs MuJoCo/MetaWorld simulation and makes
    HTTP calls to miner policy endpoints to get actions.
    
    Usage:
        evaluator = RoboticsEvaluator(config)
        results = await evaluator.evaluate_miner(miner, task_ids)
        await evaluator.cleanup()
    """
    
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        self._env = None
        self._env_lock = asyncio.Lock()
    
    def _load_env_sync(self, load_kwargs: dict):
        """Synchronous wrapper for affinetes load_env (runs in thread)."""
        if is_shutdown_requested():
            raise asyncio.CancelledError("Shutdown requested")
        import affinetes as af_env
        return af_env.load_env(**load_kwargs)
    
    async def _get_eval_environment(self):
        """Get or create the affinetes-managed eval environment."""
        async with self._env_lock:
            if self._env is not None:
                # Check if still ready
                try:
                    if self._env.is_ready():
                        return self._env
                except Exception:
                    pass
                self._env = None
            
            try:
                import affinetes as af_env  # noqa: F401
            except ImportError:
                raise ImportError(
                    "affinetes is required for evaluation. "
                    "Install with: pip install affinetes"
                )
            
            logger.info(
                "loading_eval_environment",
                image=self.config.eval_image,
                mode=self.config.mode,
            )
            
            # Load eval environment via affinetes
            load_kwargs = {
                "image": self.config.eval_image,
                "mode": self.config.mode,
                "mem_limit": self.config.mem_limit,
                "env_vars": {
                    "MUJOCO_GL": os.environ.get("MUJOCO_GL", "egl"),
                },
                "pull": True,
            }
            
            if self.config.mode == "docker":
                load_kwargs.update({
                    "hosts": self.config.hosts,
                    "container_name": "robo-eval-env",
                    "force_recreate": True,
                })
            elif self.config.mode == "basilica":
                load_kwargs.update({
                    "cpu_limit": "2000m",
                    "ttl_buffer": self.config.eval_timeout + 60,
                })
            
            # Run in thread so it can be cancelled
            self._env = await asyncio.to_thread(self._load_env_sync, load_kwargs)
            logger.info("eval_environment_loaded")
            return self._env
    
    def _get_miner_base_url(self, miner: MinerCommitment) -> str:
        """
        Build miner's base URL from commitment.
        
        Miners can deploy to:
        1. Chutes: chute_id -> https://{slug}.chutes.ai
        2. Self-hosted: chute_id can be a full URL (http://... or https://...)
        """
        if miner.chute_id:
            # If chute_id is already a URL, use it directly (for testing/self-hosted)
            if miner.chute_id.startswith("http://") or miner.chute_id.startswith("https://"):
                return miner.chute_id.rstrip("/")
            
            # Otherwise, build Chutes URL from slug
            # Chute IDs are typically in format: username-modelname-version
            slug = miner.chute_id.replace("_", "-").lower()
            return f"https://{slug}.chutes.ai"
        
        # Fallback: use docker_image as indicator
        # In practice, miners should always have a chute_id
        raise ValueError(
            f"Miner {miner.uid} has no chute_id. "
            "Miners must deploy their policy to Chutes."
        )
    
    async def evaluate_miner(
        self,
        miner: MinerCommitment,
        task_ids: list[int],
        env_id: str = "metaworld/pick-place-v3",
    ) -> dict[str, Any]:
        """
        Evaluate a miner on multiple tasks.
        
        Args:
            miner: Miner commitment with endpoint info
            task_ids: List of task IDs to evaluate
            env_id: Environment ID to use
            
        Returns:
            Dict with:
            - uid: Miner UID
            - env_id: Environment ID
            - success_rate: Fraction of successful episodes
            - mean_reward: Average episode reward
            - results: List of individual episode results
        """
        env = await self._get_eval_environment()
        
        try:
            base_url = self._get_miner_base_url(miner)
        except ValueError as e:
            logger.warning("miner_no_endpoint", uid=miner.uid, error=str(e))
            return {
                "uid": miner.uid,
                "env_id": env_id,
                "success_rate": 0.0,
                "mean_reward": 0.0,
                "error": str(e),
                "results": [],
            }
        
        logger.info(
            "evaluating_miner",
            uid=miner.uid,
            env_id=env_id,
            base_url=base_url,
            n_tasks=len(task_ids),
        )
        
        results = []
        for task_id in task_ids:
            try:
                result = await env.evaluate(
                    task_id=task_id,
                    model=miner.huggingface_repo,
                    base_url=base_url,
                    env_id=env_id,
                    max_timesteps=self.config.max_timesteps,
                    action_timeout=self.config.action_timeout,
                    use_images=self.config.use_images,
                    _timeout=self.config.eval_timeout,
                )
                results.append(result)
            except asyncio.TimeoutError:
                logger.warning(
                    "evaluation_timeout",
                    uid=miner.uid,
                    task_id=task_id,
                )
                results.append({
                    "task_id": task_id,
                    "score": 0.0,
                    "success": False,
                    "error": "Evaluation timeout",
                })
            except Exception as e:
                logger.error(
                    "evaluation_error",
                    uid=miner.uid,
                    task_id=task_id,
                    error=str(e),
                )
                results.append({
                    "task_id": task_id,
                    "score": 0.0,
                    "success": False,
                    "error": str(e),
                })
        
        # Aggregate results
        successes = sum(1 for r in results if r.get("success", False))
        rewards = [r.get("extra", {}).get("total_reward", 0.0) for r in results]
        
        return {
            "uid": miner.uid,
            "env_id": env_id,
            "success_rate": successes / len(results) if results else 0.0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0.0,
            "results": results,
        }
    
    async def evaluate_all_miners(
        self,
        miners: list[MinerCommitment],
        env_ids: list[str],
        episodes_per_env: int,
        block_number: int,
    ) -> dict[int, dict[str, float]]:
        """
        Evaluate all miners on all environments.
        
        Args:
            miners: List of miner commitments
            env_ids: List of environment IDs to evaluate
            episodes_per_env: Number of episodes per environment
            block_number: Current block number (used for seed generation)
            
        Returns:
            Dict mapping uid -> env_id -> success_rate
        """
        results: dict[int, dict[str, float]] = {}
        
        for miner in miners:
            results[miner.uid] = {}
            
            for env_id in env_ids:
                # Generate deterministic task IDs from block number
                task_ids = [
                    hash(f"{block_number}:{env_id}:{miner.uid}:{i}") % (2**31)
                    for i in range(episodes_per_env)
                ]
                
                eval_result = await self.evaluate_miner(
                    miner=miner,
                    task_ids=task_ids,
                    env_id=env_id,
                )
                
                results[miner.uid][env_id] = eval_result["success_rate"]
                
                logger.info(
                    "miner_env_evaluated",
                    uid=miner.uid,
                    env_id=env_id,
                    success_rate=eval_result["success_rate"],
                )
        
        return results
    
    async def cleanup(self):
        """Cleanup eval environment."""
        async with self._env_lock:
            if self._env is not None:
                try:
                    await self._env.cleanup()
                except Exception as e:
                    logger.warning("cleanup_error", error=str(e))
                self._env = None
    
    def force_cleanup(self):
        """Force cleanup by killing docker container directly."""
        import subprocess
        
        container_name = "robo-eval-env"
        logger.info("force_cleanup_container", container=container_name)
        
        try:
            # Kill the container
            result = subprocess.run(
                ["docker", "kill", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            logger.info("docker_kill_result", returncode=result.returncode, stderr=result.stderr.strip() if result.stderr else "")
        except Exception as e:
            logger.warning("docker_kill_failed", error=str(e))
        
        try:
            # Remove the container
            result = subprocess.run(
                ["docker", "rm", "-f", container_name],
                capture_output=True,
                text=True,
                timeout=5,
            )
            logger.info("docker_rm_result", returncode=result.returncode, stderr=result.stderr.strip() if result.stderr else "")
        except Exception as e:
            logger.warning("docker_rm_failed", error=str(e))
        
        self._env = None
        logger.info("force_cleanup_complete")


def get_default_environment_ids() -> list[str]:
    """Get default list of environment IDs for evaluation."""
    all_envs = get_all_environment_ids()
    
    # For now, focus on MetaWorld environments
    metaworld_envs = [e for e in all_envs if e.startswith("metaworld/")]
    
    if metaworld_envs:
        return metaworld_envs
    
    return all_envs
