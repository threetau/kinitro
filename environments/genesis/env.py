"""
Affinetes-compatible Genesis evaluation environment.

This Actor class runs inside an affinetes-managed container and:
1. Manages Genesis physics simulation (humanoid, quadruped, manipulation)
2. Queries miner policy endpoints (on Basilica or self-hosted) for actions
3. Returns evaluation scores

Supported environments:
- genesis/g1-v0: Unitree G1 humanoid (navigate, pickup, place, push)

Usage (from backend):
    import affinetes as af_env

    env = af_env.load_env(image="kinitro/genesis:v1")

    result = await env.evaluate(
        task_id=789,
        base_url="https://xxx.deployments.basilica.ai",
        env_id="genesis/g1-v0"
    )
"""

import time
import traceback

import httpx
import numpy as np
import structlog

# Import from kinitro package (installed in container via PYTHONPATH)
from kinitro.environments import get_environment
from kinitro.environments.registry import get_all_environment_ids
from kinitro.rl_interface import Action

logger = structlog.get_logger()


class Actor:
    """
    Genesis evaluation actor for affinetes.

    Runs Genesis physics simulation and queries miner policy endpoints
    to get actions, matching the Affine (SN120) evaluation pattern.
    """

    def __init__(self):
        """Initialize the evaluation actor."""
        self._env_cache = {}
        # Don't cache the HTTP client - create fresh for each evaluation
        # to avoid event loop binding issues when affinetes calls methods
        # from different event loops

    def _get_env(self, env_id: str):
        """Get or create a robotics environment."""
        if env_id not in self._env_cache:
            self._env_cache[env_id] = get_environment(env_id)
        return self._env_cache[env_id]

    async def _call_miner(
        self,
        base_url: str,
        endpoint: str,
        payload: dict,
        timeout: float = 0.5,
    ) -> dict:
        """
        Call miner's policy endpoint.

        Args:
            base_url: Miner's base URL (e.g., https://xxx.deployments.basilica.ai)
            endpoint: Endpoint path (e.g., "act" or "reset")
            payload: JSON payload to send
            timeout: Request timeout in seconds

        Returns:
            JSON response from miner
        """
        url = f"{base_url.rstrip('/')}/{endpoint}"

        # Create a fresh client for each call to avoid event loop binding issues
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def list_environments(self) -> list[str]:
        """List available Genesis environments."""
        return [env_id for env_id in get_all_environment_ids() if env_id.startswith("genesis/")]

    async def evaluate(
        self,
        task_id: int,
        seed: int | None = None,
        model: str | None = None,
        base_url: str | None = None,
        env_id: str = "genesis/g1-v0",
        max_timesteps: int = 500,
        action_timeout: float = 0.5,
        use_images: bool = True,
        timeout: int = 300,
        **kwargs,
    ) -> dict:
        """
        Evaluate a miner's policy on a Genesis task.

        This method:
        1. Resets the simulation environment
        2. Calls the miner's /reset endpoint
        3. Loops: get observation -> call miner's /act -> step simulation
        4. Returns success/failure score

        Args:
            task_id: Task identifier for reproducibility
            seed: Random seed (defaults to task_id)
            model: Miner's model name (for logging)
            base_url: Miner's policy endpoint URL (required)
                      e.g., "https://xxx.deployments.basilica.ai"
            env_id: Environment ID (e.g., "genesis/g1-v0")
            max_timesteps: Maximum steps per episode
            action_timeout: Timeout for each action request (seconds)
            use_images: Whether to send camera images to miner
            timeout: Overall evaluation timeout (seconds)

        Returns:
            Evaluation result dict with:
            - task_name: Environment identifier
            - score: 1.0 for success, 0.0 for failure
            - success: Boolean success flag
            - time_taken: Total evaluation time
            - extra: Additional metrics and metadata
            - error: Error message if evaluation failed
        """
        # Validate env_id is a genesis environment
        if not env_id.startswith("genesis/"):
            return self._build_error_result(
                env_id=env_id,
                task_id=task_id,
                seed=seed or task_id,
                start_time=time.time(),
                error=f"Invalid env_id for Genesis container: {env_id}. Must start with 'genesis/'",
            )

        if base_url is None:
            raise ValueError("base_url (miner endpoint) is required")

        if seed is None:
            seed = task_id

        start_time = time.time()

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
        env_id: str,
        max_timesteps: int,
        action_timeout: float,
        use_images: bool,
        start_time: float,
        overall_timeout: int = 300,
    ) -> dict:
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
                endpoint="reset",
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
                    endpoint="act",
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
        env_id: str,
        task_id: int,
        seed: int,
        start_time: float,
        error: str,
        extra: dict[str, object] | None = None,
    ) -> dict:
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

    async def cleanup(self):
        """Cleanup resources."""
        # HTTP clients are now created per-request, no need to close here

        # Close environments
        for env in self._env_cache.values():
            try:
                env.close()
            except Exception:
                pass
        self._env_cache.clear()
