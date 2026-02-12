"""Shared utilities for loading affinetes environments and running evaluations."""

import asyncio
import subprocess
from typing import Any

import affinetes as af_env
import docker.types
import structlog

from kinitro.backend.models import Task, TaskResult

logger = structlog.get_logger()


def build_load_kwargs(
    image: str,
    eval_mode: str,
    mem_limit: str,
    executor_id: str,
    family: str,
    hosts: list[str],
    eval_timeout: int,
    gpu_enabled: bool = False,
) -> dict[str, Any]:
    """Build kwargs dict for af_env.load_env().

    Args:
        image: Docker image tag for the environment.
        eval_mode: 'docker' or 'basilica'.
        mem_limit: Memory limit string (e.g. '8g').
        executor_id: Unique executor identifier.
        family: Environment family name (e.g. 'metaworld').
        hosts: Docker hosts list.
        eval_timeout: Evaluation timeout in seconds.
        gpu_enabled: Whether to enable GPU passthrough.

    Returns:
        Dict of keyword arguments for af_env.load_env().
    """
    load_kwargs: dict[str, Any] = {
        "image": image,
        "mode": eval_mode,
        "mem_limit": mem_limit,
        "pull": True,
    }

    if eval_mode == "docker":
        load_kwargs.update(
            {
                "hosts": hosts,
                "container_name": f"kinitro-eval-{executor_id}-{family}",
                "force_recreate": True,
            }
        )
        if gpu_enabled:
            load_kwargs["device_requests"] = [
                docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
            ]
    elif eval_mode == "basilica":
        load_kwargs.update(
            {
                "cpu_limit": "2000m",
                "ttl_buffer": eval_timeout + 60,
            }
        )

    return load_kwargs


async def load_and_warmup_env(family: str, image: str, load_kwargs: dict[str, Any]) -> Any:
    """Load an affinetes environment and perform a warmup call.

    Args:
        family: Environment family name (for logging).
        image: Docker image tag (for logging).
        load_kwargs: Keyword arguments for af_env.load_env().

    Returns:
        The loaded affinetes environment instance.
    """
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
    return env


async def run_evaluation(
    env: Any,
    task: Task,
    max_timesteps: int,
    action_timeout: float,
    use_images: bool,
    eval_timeout: int,
) -> TaskResult:
    """Run an evaluation and build a TaskResult from the response.

    Args:
        env: The affinetes environment instance.
        task: The task to evaluate.
        max_timesteps: Maximum timesteps per episode.
        action_timeout: Timeout for miner action responses.
        use_images: Whether to include camera images in observations.
        eval_timeout: Timeout for the evaluation call.

    Returns:
        TaskResult with the evaluation outcome.
    """
    result = await env.evaluate(
        task_id=task.seed,
        model=f"miner-{task.miner_uid}",
        base_url=task.miner_endpoint,
        env_id=task.env_id,
        max_timesteps=max_timesteps,
        action_timeout=action_timeout,
        use_images=use_images,
        _timeout=eval_timeout,
    )

    success = result.get("success", False)
    score = result.get("score", 0.0)
    extra = result.get("extra", {})
    error = result.get("error")

    return TaskResult(
        task_uuid=task.task_uuid,
        success=success,
        score=score,
        total_reward=extra.get("total_reward", 0.0),
        timesteps=extra.get("timesteps", 0),
        error=error,
    )


def force_remove_container(container_name: str) -> None:
    """Force-remove a Docker container by name, ignoring errors.

    Args:
        container_name: Name of the Docker container to remove.
    """
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_name],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception as e:
        logger.debug("force_remove_container_failed", container=container_name, error=str(e))
