"""
Affinetes-compatible evaluation environment template.

This Actor class runs inside an affinetes-managed container and:
1. Manages simulation (MuJoCo, PyBullet, Isaac, etc.)
2. Queries miner policy endpoints for actions
3. Returns evaluation scores

Usage (from backend):
    import affinetes as af_env
    env = af_env.load_env(image="kinitro/myenv:v1")
    result = await env.evaluate(
        task_id=123,
        base_url="https://xxx.deployments.basilica.ai",
        env_id="myenv/v0"
    )
"""

import time
import traceback

import httpx
import numpy as np
import structlog

from kinitro.environments import get_environment
from kinitro.environments.registry import get_all_environment_ids
from kinitro.rl_interface import Action, ActionKeys

logger = structlog.get_logger()


class Actor:
    """Evaluation actor for affinetes."""

    def __init__(self):
        """Initialize the evaluation actor."""
        self._env_cache = {}

    def _get_env(self, env_id: str):
        """Get or create a robotics environment (lazy loading)."""
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
            endpoint: Endpoint path ("act" or "reset")
            payload: JSON payload to send
            timeout: Request timeout in seconds

        Returns:
            JSON response from miner
        """
        url = f"{base_url.rstrip('/')}/{endpoint}"
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0)) as client:
            resp = await client.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    async def list_environments(self) -> list[str]:
        """List available environments in this family."""
        # TODO: Change "myenv/" to your environment family prefix
        return [e for e in get_all_environment_ids() if e.startswith("myenv/")]

    async def evaluate(
        self,
        task_id: int,
        seed: int | None = None,
        model: str | None = None,
        base_url: str | None = None,
        env_id: str = "myenv/v0",  # TODO: Change default env_id
        max_timesteps: int = 500,
        action_timeout: float = 0.5,
        use_images: bool = True,
        timeout: int = 300,
        **kwargs,
    ) -> dict:
        """
        Evaluate a miner's policy.

        Returns:
            {
                "task_name": "robotics:myenv/v0",
                "score": 0.0 or 1.0,
                "success": bool,
                "time_taken": float,
                "extra": {...},
                "error": optional error string
            }
        """
        # TODO: Change "myenv/" to your environment family prefix
        if not env_id.startswith("myenv/"):
            return self._build_error_result(
                env_id=env_id,
                task_id=task_id,
                seed=seed or task_id,
                start_time=time.time(),
                error=f"Invalid env_id: {env_id}. Must start with 'myenv/'",
            )

        if base_url is None:
            raise ValueError("base_url (miner endpoint) is required")

        seed = seed if seed is not None else task_id
        start_time = time.time()

        try:
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
            await self._call_miner(
                base_url=base_url,
                endpoint="reset",
                payload={"task_config": task_config_dict},
                timeout=5.0,
            )

            # Reset simulation environment
            obs = env.reset(task_config)

            total_reward = 0.0
            timesteps = 0

            # Main evaluation loop
            for t in range(max_timesteps):
                if time.time() - start_time > timeout:
                    return self._build_error_result(
                        env_id=env_id,
                        task_id=task_id,
                        seed=seed,
                        start_time=start_time,
                        error=f"Evaluation timeout after {timeout}s",
                    )

                # Get action from miner
                payload = {"obs": obs.to_payload(include_images=use_images)}
                response = await self._call_miner(
                    base_url=base_url,
                    endpoint="act",
                    payload=payload,
                    timeout=action_timeout,
                )

                # TODO: Parse action from response - adjust for your action space
                action_data = response.get("action")
                if action_data is None:
                    action = Action(
                        continuous={
                            ActionKeys.EE_TWIST: [0.0] * 6,
                            ActionKeys.GRIPPER: [0.0],
                        }
                    )
                elif isinstance(action_data, dict) and "continuous" in action_data:
                    action = Action.model_validate(action_data)
                else:
                    arr = np.asarray(action_data).flatten()
                    action = Action(
                        continuous={
                            ActionKeys.EE_TWIST: arr[:6].tolist() if len(arr) >= 6 else [0.0] * 6,
                            ActionKeys.GRIPPER: [float(arr[6])] if len(arr) >= 7 else [0.0],
                        }
                    )

                # Step environment
                obs, reward, done, info = env.step(action)
                total_reward += reward
                timesteps = t + 1

                if done:
                    break

            success = env.get_success()
            return {
                "task_name": f"robotics:{env_id}",
                "score": 1.0 if success else 0.0,
                "success": success,
                "time_taken": time.time() - start_time,
                "extra": {
                    "task_id": task_id,
                    "seed": seed,
                    "env_id": env_id,
                    "timesteps": timesteps,
                    "total_reward": float(total_reward),
                },
            }

        except Exception as e:
            return self._build_error_result(
                env_id=env_id,
                task_id=task_id,
                seed=seed,
                start_time=start_time,
                error=f"{type(e).__name__}: {e}\n{traceback.format_exc()}",
            )

    def _build_error_result(
        self,
        env_id: str,
        task_id: int,
        seed: int,
        start_time: float,
        error: str,
    ) -> dict:
        """Build error result dict."""
        return {
            "task_name": f"robotics:{env_id}",
            "score": 0.0,
            "success": False,
            "error": error,
            "time_taken": time.time() - start_time,
            "extra": {"task_id": task_id, "seed": seed, "env_id": env_id},
        }

    async def cleanup(self):
        """Cleanup resources."""
        for env in self._env_cache.values():
            try:
                env.close()
            except Exception:
                pass
        self._env_cache.clear()
