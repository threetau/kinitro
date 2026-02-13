"""
Affinetes-compatible Genesis evaluation environment.

This Actor class runs inside an affinetes-managed container and:
1. Manages Genesis physics simulation (humanoid, quadruped, manipulation)
2. Queries miner policy endpoints (on Basilica or self-hosted) for actions
3. Returns evaluation scores

Usage (from backend):
    import affinetes as af_env

    env = af_env.load_env(image="kinitro/genesis:v1")

    result = await env.evaluate(
        task_id=789,
        base_url="https://example.deployments.basilica.ai",
        env_id="genesis/g1-v0"
    )
"""

import asyncio
import time
import traceback
from typing import Any, TypedDict

import httpx
import numpy as np
import structlog

# Import from kinitro package (installed in container via PYTHONPATH)
from kinitro.environments import get_environment
from kinitro.environments.base import RoboticsEnvironment
from kinitro.environments.registry import get_environments_by_family
from kinitro.rl_interface import Action
from kinitro.types import EnvironmentId

logger = structlog.get_logger()


class EvalResult(TypedDict, total=False):
    """Result of a single evaluation run."""

    task_name: str
    score: float
    success: bool
    time_taken: float
    extra: dict[str, Any]
    error: str


class Actor:
    """
    Genesis evaluation actor for affinetes.

    Runs Genesis physics simulation and queries miner policy endpoints
    to get actions, matching the Affine (SN120) evaluation pattern.
    """

    def __init__(self) -> None:
        """Initialize the evaluation actor."""
        self._env_cache = {}
        self._env_locks: dict[EnvironmentId, asyncio.Lock] = {}
        # Don't cache the HTTP client - create fresh for each evaluation
        # to avoid event loop binding issues when affinetes calls methods
        # from different event loops

    def _get_env(self, env_id: EnvironmentId) -> RoboticsEnvironment:
        """Get or create a robotics environment."""
        if env_id not in self._env_cache:
            self._env_cache[env_id] = get_environment(env_id)
        return self._env_cache[env_id]

    async def _call_miner(
        self,
        base_url: str,
        path: str,
        payload: dict,
        timeout: float = 0.5,
    ) -> dict:
        """
        Call miner's policy endpoint.

        Args:
            base_url: Miner's base URL (e.g., https://example.deployments.basilica.ai)
            path: Request path (e.g., "act" or "reset")
            payload: JSON payload to send
            timeout: Request timeout in seconds

        Returns:
            JSON response from miner
        """
        url = f"{base_url.rstrip('/')}/{path}"

        # Create a fresh client for each call to avoid event loop binding issues
        connect_timeout = min(timeout * 2, 5.0)
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=connect_timeout),
        ) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def list_environments(self) -> list[EnvironmentId]:
        """List available Genesis environments."""
        return get_environments_by_family("genesis")

    async def evaluate(
        self,
        task_id: int,
        base_url: str,
        seed: int | None = None,
        model: str | None = None,
        env_id: EnvironmentId = EnvironmentId("genesis/g1-v0"),
        max_timesteps: int = 500,
        action_timeout: float = 0.5,
        use_images: bool = True,
        timeout: int = 300,
        **kwargs,
    ) -> EvalResult:
        """
        Evaluate a miner's policy on a Genesis task.

        This method:
        1. Calls the miner's /reset endpoint (task metadata only)
        2. Resets the simulation environment
        3. Loops: get observation -> call miner's /act -> step simulation
        4. Returns success/failure score

        Args:
            task_id: Task identifier for reproducibility
            base_url: Miner's policy endpoint URL
                      (e.g., "https://example.deployments.basilica.ai")
            seed: Random seed (defaults to task_id)
            model: Miner's model name (for logging)
            env_id: Environment ID (e.g., "genesis/g1-v0")
            max_timesteps: Maximum steps per episode
            action_timeout: Timeout for each action request (seconds)
            use_images: Whether to send camera images to miner
            timeout: Overall evaluation timeout (seconds)

        Returns:
            EvalResult with score 1.0 (success) or 0.0 (failure).
            Score is strictly binary — no partial credit.
        """
        # Validate env_id is a genesis environment
        if not env_id.startswith("genesis/"):
            return self._build_error_result(
                env_id=env_id,
                task_id=task_id,
                seed=seed if seed is not None else task_id,
                start_time=time.time(),
                error=f"Invalid env_id for Genesis container: {env_id}. Must start with 'genesis/'",
            )

        if seed is None:
            seed = task_id

        start_time = time.time()

        # Serialize concurrent evaluations for the same env_id to prevent
        # data races on the shared cached environment instance.
        lock = self._env_locks.setdefault(env_id, asyncio.Lock())
        async with lock:
            try:
                return await self._run_evaluation(
                    task_id=task_id,
                    seed=seed,
                    model=model,
                    base_url=base_url,
                    env_id=env_id,
                    max_timesteps=max_timesteps,
                    action_timeout=action_timeout,
                    use_images=use_images,
                    start_time=start_time,
                    overall_timeout=timeout,
                )
            except Exception as e:
                return self._build_error_result(
                    env_id=env_id,
                    task_id=task_id,
                    seed=seed,
                    start_time=start_time,
                    error=f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}",
                )

    async def _run_evaluation(
        self,
        task_id: int,
        seed: int,
        model: str | None,
        base_url: str,
        env_id: EnvironmentId,
        max_timesteps: int,
        action_timeout: float,
        use_images: bool,
        start_time: float,
        overall_timeout: int = 300,
    ) -> EvalResult:
        """Internal evaluation loop."""

        # Get environment
        env = self._get_env(env_id)

        # Generate task configuration
        task_config = env.generate_task(seed=seed)
        task_config_dict = {
            "env_id": env_id,
            "env_name": task_config.env_name,
            "task_name": task_config.task_name,
            "seed": seed,
            "task_id": task_id,
        }

        # Reset miner policy
        try:
            await self._call_miner(
                base_url=base_url,
                path="reset",
                payload={"task_config": task_config_dict},
                timeout=5.0,
            )
        except Exception as e:
            return self._build_error_result(
                env_id=env_id,
                task_id=task_id,
                seed=seed,
                start_time=start_time,
                error=f"Miner reset failed: {type(e).__name__}: {str(e)}",
            )

        # Reset simulation environment
        obs = env.reset(task_config)

        total_reward = 0.0
        timesteps = 0
        action_times = []

        for t in range(max_timesteps):
            # Check overall timeout
            if time.time() - start_time > overall_timeout:
                return self._build_error_result(
                    env_id=env_id,
                    task_id=task_id,
                    seed=seed,
                    start_time=start_time,
                    error=f"Evaluation timeout after {overall_timeout}s",
                    extra={"timesteps_completed": t},
                )

            # Build request payload
            payload = {"obs": obs.to_payload(include_images=use_images)}

            # Get action from miner
            action_start = time.time()
            try:
                response = await self._call_miner(
                    base_url=base_url,
                    path="act",
                    payload=payload,
                    timeout=action_timeout,
                )
                action_data = response.get("action")
                if action_data is None:
                    return self._build_error_result(
                        env_id=env_id,
                        task_id=task_id,
                        seed=seed,
                        start_time=start_time,
                        error=f"Miner returned no action at step {t}",
                        extra={"timesteps_completed": t},
                    )
                action = Action.model_validate(action_data)
            except httpx.TimeoutException:
                return self._build_error_result(
                    env_id=env_id,
                    task_id=task_id,
                    seed=seed,
                    start_time=start_time,
                    error=f"Miner action timeout at step {t}",
                    extra={"timesteps_completed": t},
                )
            except Exception as e:
                return self._build_error_result(
                    env_id=env_id,
                    task_id=task_id,
                    seed=seed,
                    start_time=start_time,
                    error=f"Miner action failed at step {t}: {type(e).__name__}: {str(e)}",
                    extra={"timesteps_completed": t},
                )

            action_times.append(time.time() - action_start)

            # Step environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            timesteps = t + 1

            if done:
                break

        # Get success status
        success = env.get_success()
        score = 1.0 if success else 0.0

        return {
            "task_name": f"robotics:{env_id}",
            "score": score,
            "success": success,
            "time_taken": time.time() - start_time,
            "extra": {
                "task_id": task_id,
                "seed": seed,
                "env_id": env_id,
                "timesteps": timesteps,
                "total_reward": float(total_reward),
                "mean_action_time": float(np.mean(action_times)) if action_times else 0.0,
                "max_action_time": float(np.max(action_times)) if action_times else 0.0,
                "model": model,
                "base_url": base_url,
            },
        }

    def _build_error_result(
        self,
        env_id: EnvironmentId,
        task_id: int,
        seed: int,
        start_time: float,
        error: str,
        extra: dict[str, object] | None = None,
    ) -> EvalResult:
        """Build error result dict."""
        extra_fields: dict[str, object] = {
            "task_id": task_id,
            "seed": seed,
            "env_id": env_id,
        }
        if extra:
            extra_fields.update(extra)
        return {
            "task_name": f"robotics:{env_id}",
            "score": 0.0,
            "success": False,
            "error": error,
            "time_taken": time.time() - start_time,
            "extra": extra_fields,
        }

    async def cleanup(self) -> None:
        """Cleanup resources."""
        # HTTP clients are now created per-request, no need to close here

        # Close environments
        for env_id, env in self._env_cache.items():
            try:
                env.close()
            except Exception as e:
                logger.debug("env_close_error", env_id=env_id, error=str(e))
        self._env_cache.clear()
