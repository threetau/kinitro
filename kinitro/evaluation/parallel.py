"""Parallel evaluation orchestration for multiple miners and environments.

.. deprecated::
    This module is deprecated in favor of the split architecture:
    - kinitro.scheduler: Task generation and scoring
    - kinitro.executor: MuJoCo evaluations via affinetes
    - kinitro.api: REST API and task pool management

    This module remains for backwards compatibility and local testing only.
    Production deployments should use the split architecture with Chutes.
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from typing import Any

import structlog

from kinitro.chain.commitments import MinerCommitment
from kinitro.environments import get_environment
from kinitro.environments.base import EpisodeResult
from kinitro.evaluation.rollout import PolicyInterface, RolloutConfig, run_episode
from kinitro.scheduler.task_generator import generate_seed

logger = structlog.get_logger()


@dataclass
class MinerResult:
    """Aggregated results for one miner on one environment."""

    uid: int
    env_id: str
    success_rate: float
    mean_reward: float
    episodes_completed: int
    episodes_failed: int
    episode_results: list[EpisodeResult] = field(default_factory=list)


@dataclass
class EvaluationResult:
    """Complete evaluation results for all miners across all environments."""

    block_number: int
    miner_results: dict[int, dict[str, MinerResult]]  # uid -> env_id -> result
    evaluation_time_seconds: float


class BasilicaPolicyAdapter:
    """
    Adapter to make Basilica container work with PolicyInterface.

    This wraps the af_env interface to match our PolicyInterface protocol.

    The observation format is:
        - end_effector_pos: [x, y, z] - Robot end-effector position
        - gripper_state: float - Gripper open/close state
        - camera_images: dict[str, str] - Camera views as base64 PNGs
    """

    def __init__(self, basilica_env: Any):
        """
        Initialize adapter.

        Args:
            basilica_env: Loaded af_env environment
        """
        self._env = basilica_env

    async def reset(self, task_config: dict[str, Any]) -> None:
        """Reset policy for new episode."""
        await self._env.reset(task_config=task_config)

    async def act(self, observation: dict[str, Any]) -> list[float]:
        """
        Get action for observation.

        Args:
            observation: Dict with end_effector_pos, gripper_state, camera_images

        Returns:
            Action as list of floats
        """
        return await self._env.act(observation=observation)


class MockPolicy:
    """Mock policy for testing without Basilica."""

    def __init__(self, action_dim: int = 4):
        self.action_dim = action_dim

    async def reset(self, task_config: dict[str, Any]) -> None:
        pass

    async def act(self, observation: dict[str, Any]) -> list[float]:
        """
        Get action for observation (random for mock policy).

        Args:
            observation: Dict with end_effector_pos, gripper_state, camera_images

        Returns:
            Random action as list of floats
        """
        import random

        return [random.uniform(-1, 1) for _ in range(self.action_dim)]


async def load_miner_policy(
    miner: MinerCommitment,
    use_basilica: bool = True,
    basilica_api_token: str | None = None,
) -> PolicyInterface:
    """
    Load a miner's policy from their container.

    Args:
        miner: Miner commitment info
        use_basilica: Whether to use Basilica (True) or mock (False)
        basilica_api_token: Basilica API token

    Returns:
        PolicyInterface for querying the miner's policy
    """
    if not use_basilica:
        logger.info("using_mock_policy", uid=miner.uid)
        return MockPolicy()

    try:
        import af_env

        env = af_env.load_env(
            mode="basilica",
            image=miner.docker_image,
            cpu_limit="2000m",
            mem_limit="4Gi",
            env_vars={"BASILICA_API_TOKEN": basilica_api_token or ""},
        )
        return BasilicaPolicyAdapter(env)
    except ImportError:
        logger.warning("af_env_not_available", uid=miner.uid)
        return MockPolicy()
    except Exception as e:
        logger.error("policy_load_failed", uid=miner.uid, error=str(e))
        raise


async def evaluate_miner_on_environment(
    miner: MinerCommitment,
    env_id: str,
    seeds: list[int],
    policy: PolicyInterface | None = None,
    rollout_config: RolloutConfig | None = None,
    max_concurrent: int = 10,
) -> MinerResult:
    """
    Evaluate one miner on one environment across multiple seeds.

    Args:
        miner: Miner commitment info
        env_id: Environment identifier
        seeds: List of seeds for procedural task generation
        policy: Pre-loaded policy (optional, will load if not provided)
        rollout_config: Configuration for rollouts
        max_concurrent: Maximum concurrent episodes

    Returns:
        MinerResult with aggregated statistics
    """
    if rollout_config is None:
        rollout_config = RolloutConfig()

    # Load environment
    env = get_environment(env_id)

    # Load policy if not provided
    if policy is None:
        policy = await load_miner_policy(miner, use_basilica=False)

    semaphore = asyncio.Semaphore(max_concurrent)
    episode_results: list[EpisodeResult] = []

    async def run_single_episode(seed: int) -> EpisodeResult:
        async with semaphore:
            task_config = env.generate_task(seed)
            return await run_episode(env, task_config, policy, rollout_config)

    # Run all episodes concurrently
    try:
        results = await asyncio.gather(
            *[run_single_episode(seed) for seed in seeds],
            return_exceptions=True,
        )

        # Filter successful evaluations
        for r in results:
            if isinstance(r, EpisodeResult):
                episode_results.append(r)
            else:
                logger.warning("episode_exception", error=str(r))

    finally:
        env.close()

    # Compute aggregated metrics
    if episode_results:
        success_count = sum(1 for r in episode_results if r.success)
        total_reward = sum(r.total_reward for r in episode_results)
        success_rate = success_count / len(episode_results)
        mean_reward = total_reward / len(episode_results)
    else:
        success_rate = 0.0
        mean_reward = 0.0

    failed_count = len(seeds) - len(episode_results)

    return MinerResult(
        uid=miner.uid,
        env_id=env_id,
        success_rate=success_rate,
        mean_reward=mean_reward,
        episodes_completed=len(episode_results),
        episodes_failed=failed_count,
        episode_results=episode_results,
    )


async def evaluate_all_miners(
    miners: list[MinerCommitment],
    environment_ids: list[str],
    block_number: int,
    validator_hotkey: str,
    episodes_per_env: int = 50,
    rollout_config: RolloutConfig | None = None,
    use_basilica: bool = True,
    basilica_api_token: str | None = None,
) -> EvaluationResult:
    """
    Evaluate all miners on all environments.

    This is the main evaluation entry point called by validators.

    Args:
        miners: List of miner commitments
        environment_ids: List of environment IDs to evaluate on
        block_number: Current block number (for seed generation)
        validator_hotkey: Validator's hotkey (for seed uniqueness)
        episodes_per_env: Number of episodes per environment
        rollout_config: Configuration for rollouts
        use_basilica: Whether to use Basilica containers
        basilica_api_token: Basilica API token

    Returns:
        EvaluationResult with all miner results
    """
    import time

    start_time = time.time()

    results: dict[int, dict[str, MinerResult]] = {}

    for miner in miners:
        logger.info("evaluating_miner", uid=miner.uid, hotkey=miner.hotkey[:16])
        results[miner.uid] = {}

        # Load miner's policy once
        try:
            policy = await load_miner_policy(
                miner,
                use_basilica=use_basilica,
                basilica_api_token=basilica_api_token,
            )
        except Exception as e:
            logger.error("miner_policy_load_failed", uid=miner.uid, error=str(e))
            # Mark all environments as failed for this miner
            for env_id in environment_ids:
                results[miner.uid][env_id] = MinerResult(
                    uid=miner.uid,
                    env_id=env_id,
                    success_rate=0.0,
                    mean_reward=0.0,
                    episodes_completed=0,
                    episodes_failed=episodes_per_env,
                )
            continue

        # Evaluate on each environment
        for env_id in environment_ids:
            # Generate random UUIDs and derive seeds from them
            # This ensures unpredictable seeds that miners can't pre-compute
            seeds = [generate_seed(str(uuid.uuid4())) for _ in range(episodes_per_env)]

            try:
                result = await evaluate_miner_on_environment(
                    miner=miner,
                    env_id=env_id,
                    seeds=seeds,
                    policy=policy,
                    rollout_config=rollout_config,
                )
                results[miner.uid][env_id] = result
                logger.info(
                    "env_evaluation_complete",
                    uid=miner.uid,
                    env_id=env_id,
                    success_rate=result.success_rate,
                    mean_reward=result.mean_reward,
                )
            except Exception as e:
                logger.error(
                    "env_evaluation_failed",
                    uid=miner.uid,
                    env_id=env_id,
                    error=str(e),
                )
                results[miner.uid][env_id] = MinerResult(
                    uid=miner.uid,
                    env_id=env_id,
                    success_rate=0.0,
                    mean_reward=0.0,
                    episodes_completed=0,
                    episodes_failed=episodes_per_env,
                )

    elapsed = time.time() - start_time
    logger.info("evaluation_complete", miners=len(miners), time_seconds=elapsed)

    return EvaluationResult(
        block_number=block_number,
        miner_results=results,
        evaluation_time_seconds=elapsed,
    )
