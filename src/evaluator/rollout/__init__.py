import logging
import time
from typing import Any, Dict, List

import ray
from ray.actor import ActorProxy

from core.db.models import SnowflakeId

from ..rpc.client import AgentClient
from .envs import BenchmarkSpec, EnvManager, EnvResult, EnvSpec, EpisodeResult

logger = logging.getLogger(__name__)


class RolloutCluster:
    def __init__(self, cluster_name: str):
        self.name = cluster_name  # Ray namespace
        self.workers: list[ActorProxy] = []

    def create_worker(self, rollout_worker_id: SnowflakeId) -> ActorProxy:
        worker = RolloutWorker.remote(self, rollout_worker_id)
        self.workers.append(worker)
        return worker

    def delete_worker(self, worker: ActorProxy):
        self.workers.remove(worker)
        ray.kill(worker)
        pass


@ray.remote
class RolloutWorker:
    def __init__(
        self,
        cluster: RolloutCluster,
        rollout_worker_id: SnowflakeId,
        benchmark_specs: list[BenchmarkSpec],
    ) -> None:
        self.cluster = cluster
        self.rollout_worker_id = rollout_worker_id  # Name of the ray actor
        self.submission_container_host = (
            "localhost"  # Address of the submission container
        )
        self.benchmark_specs = benchmark_specs
        self.submission_container_port = 8000
        self.submission_container_address = None
        self.eval_start = None  # Start time of the evaluation
        self.eval_end = None  # End time of the evaluation

        self.agent = AgentClient(
            host=self.submission_container_host, port=self.submission_container_port
        )

        # Environment manager
        self.env_manager = EnvManager()

        # Evaluation configuration
        self.episodes_per_task = 1  # Number of episodes to run per task
        self.max_steps_per_episode = 200  # Maximum steps per episode

    def set_config(
        self,
        submission_container_address: str,
        episodes_per_task: int = 1,
        max_steps_per_episode: int = 200,
    ):
        """Configure the worker with submission address and evaluation parameters."""
        self.submission_container_address = submission_container_address
        self.episodes_per_task = episodes_per_task
        self.max_steps_per_episode = max_steps_per_episode

        logger.info(
            f"Worker {self.rollout_worker_id} configured for benchmarks {self.benchmark_specs}"
        )

    async def connect_to_agent(self):
        """Connect to the agent server."""
        if self.submission_container_address is None:
            raise ValueError("Submission container address not set")

        try:
            await self.agent.connect()
            logger.info(
                f"Worker {self.rollout_worker_id} connected to agent at {self.submission_container_address}"
            )
        except Exception as e:
            logger.error(f"Failed to connect to agent: {e}")
            raise

    def run_all_benchmark_tasks(self) -> List[EnvResult]:
        """Run the agent through every test task in every test environment in each benchmark."""
        if not self.benchmark_specs:
            raise ValueError("Benchmark specifications not set")

        # Start evaluation timer
        self.eval_start = time.time()

        # Collect all tasks from all benchmarks
        all_task_specs = []
        for benchmark_spec in self.benchmark_specs:
            print(f"Running all tasks for benchmark: {benchmark_spec.benchmark_name}")

            # Get all environment and task combinations for this benchmark
            task_specs = self.env_manager.get_benchmark_envs(
                benchmark_spec
            )  # This now returns all tasks
            all_task_specs.extend(task_specs)
            logger.info(
                f"Worker {self.rollout_worker_id} discovered {len(task_specs)} tasks for {benchmark_spec}"
            )

        logger.info(
            f"Worker {self.rollout_worker_id} will evaluate {len(all_task_specs)} total tasks"
        )

        # Run all tasks
        task_results = []
        for i, task_spec in enumerate(all_task_specs):
            logger.info(f"Running task {i + 1}/{len(all_task_specs)}: {task_spec}")
            try:
                task_result = self.run_env(task_spec)  # This now runs a specific task
                task_results.append(task_result)

                # Log task completion
                logger.info(
                    f"Task {task_spec} completed: "
                    f"success_rate={task_result.success_rate:.3f}, "
                    f"mean_reward={task_result.mean_reward:.3f}, "
                    f"mean_steps={task_result.mean_steps:.1f}"
                )
            except Exception as e:
                logger.error(f"Failed to run task {task_spec}: {e}")
                # Continue with other tasks
                continue

        # End evaluation timer
        self.eval_end = time.time()

        # Log summary
        self._log_evaluation_summary(task_results)

        return task_results

    def run_env(self, env_spec: EnvSpec) -> EnvResult:
        """Run all episodes for a single environment."""
        # Create environment for this spec
        env = self.env_manager.make_env(env_spec)

        # Run multiple episodes for this environment
        episodes = []
        for episode_id in range(self.episodes_per_task):
            try:
                episode_result = self.run_episode(
                    env, self.agent, env_spec, episode_id, self.max_steps_per_episode
                )
                episodes.append(episode_result)
            except Exception as e:
                logger.error(
                    f"Failed to run episode {episode_id} for environment {env_spec}: {e}"
                )
                continue

        # Clean up environment
        env.close()

        # Create environment result
        env_result = EnvResult.from_episodes(env_spec, episodes)

        return env_result

    def run_episode(self, env, agent, env_spec, episode_id, max_steps):
        """Run a single episode with the given environment and agent."""
        logger.info(f"Starting episode {episode_id} for environment {env_spec}")

        observation, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        episode_start = time.time()
        episode_steps = []

        while not done and step_count < max_steps:
            try:
                action = agent.act(observation)
                observation, reward, terminated, truncated, info = env.step(action)

                total_reward += reward
                step_count += 1
                done = terminated or truncated

                # Record step
                episode_steps.append(
                    {"step": step_count, "reward": reward, "done": done, "info": info}
                )

            except Exception as e:
                logger.error(
                    f"Error during step {step_count} in episode {episode_id}: {e}"
                )
                break

        episode_duration = time.time() - episode_start

        episode_result = EpisodeResult(
            episode_id=episode_id,
            env_spec=env_spec,
            reward=total_reward,
            steps=step_count,
            success=info.get("success", False) if info else False,
            info={"duration": episode_duration, "episode_steps": episode_steps},
        )

        logger.info(
            f"Episode {episode_id} completed: {step_count} steps, "
            f"reward={total_reward:.2f}, success={episode_result.success}"
        )

        return episode_result

    def run_episodes(self, env_specs: List[EnvSpec]) -> List[EnvResult]:
        """Run episodes for a list of specific environments."""
        # Start evaluation timer
        self.eval_start = time.time()

        # Run specified environments
        env_results = []
        for i, env_spec in enumerate(env_specs):
            logger.info(f"Running environment {i + 1}/{len(env_specs)}: {env_spec}")
            try:
                env_result = self.run_env(env_spec)
                env_results.append(env_result)

                # Log environment completion
                logger.info(
                    f"Environment {env_spec} completed: "
                    f"success_rate={env_result.success_rate:.3f}, "
                    f"mean_reward={env_result.mean_reward:.3f}, "
                    f"mean_steps={env_result.mean_steps:.1f}"
                )
            except Exception as e:
                logger.error(f"Failed to run environment {env_spec}: {e}")
                continue

        # End evaluation timer
        self.eval_end = time.time()

        # Log summary
        self._log_evaluation_summary(env_results)

        return env_results

    def _log_evaluation_summary(self, env_results: List[EnvResult]):
        """Log summary statistics for the evaluation."""
        if not env_results:
            logger.warning("No environment results to summarize")
            return

        # Calculate aggregate statistics
        total_episodes = sum(len(er.episodes) for er in env_results)
        mean_success_rate = sum(er.success_rate for er in env_results) / len(
            env_results
        )
        mean_reward = sum(er.mean_reward for er in env_results) / len(env_results)
        mean_steps = sum(er.mean_steps for er in env_results) / len(env_results)

        eval_duration = (
            self.eval_end - self.eval_start if self.eval_start and self.eval_end else 0
        )

        logger.info("=== Evaluation Summary ===")
        logger.info(f"Worker: {self.rollout_worker_id}")
        logger.info(f"Benchmarks: {self.benchmark_specs}")
        logger.info(f"Environments completed: {len(env_results)}")
        logger.info(f"Total episodes: {total_episodes}")
        logger.info(f"Mean success rate: {mean_success_rate:.3f}")
        logger.info(f"Mean reward: {mean_reward:.3f}")
        logger.info(f"Mean steps: {mean_steps:.1f}")
        logger.info(f"Evaluation duration: {eval_duration:.1f}s")
        logger.info("========================")

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the worker."""
        return {
            "worker_id": str(self.rollout_worker_id),
            "submission_address": self.submission_container_address,
            "benchmark_specs": [str(spec) for spec in self.benchmark_specs],
            "eval_start": self.eval_start,
            "eval_end": self.eval_end,
            "episodes_per_task": self.episodes_per_task,
            "max_steps_per_episode": self.max_steps_per_episode,
        }
