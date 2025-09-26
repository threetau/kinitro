import asyncio
import gc
import time
from typing import Any, Dict, List, Optional

import capnp
import numpy as np
import ray
import torch
from ray.util.queue import Queue

from core.db.models import SnowflakeId
from core.log import get_logger
from core.storage import R2Config

from ..rpc.rpc_process import RPCRequest
from .envs import BenchmarkSpec, EnvManager, EnvResult, EnvSpec, EpisodeResult
from .episode_logger import EpisodeLogger, LoggingConfig

logger = get_logger(__name__)

LOGGING_INTERVAL = 10


class RolloutCluster:
    def __init__(self, cluster_name: str):
        self.name = cluster_name  # Ray namespace
        self.workers: list[ray.actor.ActorHandle] = []

    def create_worker(
        self,
        rollout_worker_id: SnowflakeId,
        benchmark_specs: List[BenchmarkSpec],
        submission_container_host: str,
        submission_container_port: int,
        submission_id: SnowflakeId | None = None,
        r2_config: Optional[R2Config] = None,
        episode_log_interval: int = 1,
        step_log_interval: int = 1,
        database_url: Optional[str] = None,
    ) -> ray.actor.ActorHandle:
        logger.info(
            f"Creating worker: {rollout_worker_id}, {benchmark_specs}, {submission_container_host}, {submission_container_port}, {submission_id}"
        )
        worker = RolloutWorker.remote(
            self.name,
            rollout_worker_id,
            benchmark_specs,
            submission_container_host,
            submission_container_port,
            submission_id,
            r2_config,
            episode_log_interval,
            step_log_interval,
            database_url,
        )
        self.workers.append(worker)
        return worker

    def delete_worker(self, worker: ray.actor.ActorHandle):
        if worker in self.workers:
            self.workers.remove(worker)
        try:
            # Try to call cleanup on the worker before killing it
            try:
                ray.get(worker.cleanup.remote(), timeout=2)
                logger.debug("Worker cleanup completed before deletion")
            except Exception as e:
                logger.debug(f"Worker cleanup before deletion failed: {e}")

            # Kill the worker actor
            ray.kill(worker)
        except Exception as e:
            logger.warning(f"Failed to kill worker: {e}")

    def cleanup_all_workers(self):
        """Clean up all workers in the cluster."""
        logger.info(f"Cleaning up {len(self.workers)} workers in cluster {self.name}")
        for worker in list(self.workers):
            self.delete_worker(worker)
        self.workers.clear()
        logger.info(f"Completed cleanup of all workers in cluster {self.name}")


@ray.remote(
    max_restarts=1, max_task_retries=0, memory=2 * 1024 * 1024 * 1024
)  # 2GB memory limit per worker
class RolloutWorker:
    def __init__(
        self,
        cluster_name: str,
        rollout_worker_id: SnowflakeId,
        benchmark_specs: list[BenchmarkSpec],
        submission_container_host: str,
        submission_container_port: int,
        submission_id: SnowflakeId | None = None,
        r2_config: Optional[R2Config] = None,
        episode_log_interval: int = 1,
        step_log_interval: int = 1,
        database_url: Optional[str] = None,
    ) -> None:
        logger.info(
            f"RolloutWorker init: {cluster_name}, {rollout_worker_id}, {benchmark_specs}, "
            f"{submission_container_host}:{submission_container_port}, {submission_id}"
        )
        self.cluster_name = cluster_name
        self.rollout_worker_id = rollout_worker_id
        self.submission_container_host = submission_container_host
        self.submission_container_port = submission_container_port
        self.benchmark_specs = benchmark_specs
        self.submission_id = submission_id or rollout_worker_id
        self.submission_container_address = (
            f"{submission_container_host}:{submission_container_port}"
        )
        self.env_manager = EnvManager()
        self.eval_start = None
        self.eval_end = None

        # Episode logging configuration
        self.episode_log_interval = episode_log_interval
        self.step_log_interval = step_log_interval
        self.r2_config = r2_config
        self.database_url = database_url
        self.episode_loggers: Dict[str, EpisodeLogger] = {}

    def cleanup(self):
        """Clean up resources held by this worker."""
        logger.info(f"Worker {self.rollout_worker_id} starting cleanup")

        # Close all environment instances
        if hasattr(self, "env_manager") and self.env_manager:
            try:
                # Close any open environments properly
                if hasattr(self.env_manager, "envs"):
                    for env in list(self.env_manager.envs.values()):
                        try:
                            env.close()
                        except Exception:
                            pass
                    self.env_manager.envs.clear() if hasattr(
                        self.env_manager, "envs"
                    ) else None
                self.env_manager = None
            except Exception as e:
                logger.warning(f"Failed to cleanup env_manager: {e}")

        # Clean up episode loggers and their database connections
        if hasattr(self, "episode_loggers"):
            for logger_instance in self.episode_loggers.values():
                try:
                    if hasattr(logger_instance, "cleanup"):
                        logger_instance.cleanup()
                    elif hasattr(logger_instance, "close"):
                        logger_instance.close()
                except Exception as e:
                    logger.warning(f"Failed to cleanup episode logger: {e}")
            self.episode_loggers.clear()

        # Clear any large data structures
        self.benchmark_specs = None
        self.submission_container_address = None
        self.submission_container_host = None
        self.submission_container_port = None

        # Force garbage collection
        gc.collect()

        logger.info(f"Worker {self.rollout_worker_id} cleanup completed")

    async def test_rpc(self, send_queue: Queue, recv_queue: Queue):
        rpc_msg = RPCRequest.create_ping("ray-ping")
        await send_queue.put_async(rpc_msg)
        while True:
            try:
                resp = await recv_queue.get_async()
                print(
                    f"[Worker {self.rollout_worker_id}] RPC test ping response={resp}"
                )
                return resp
            except asyncio.CancelledError:
                print(f"[Worker {self.rollout_worker_id}] RPC test cancelled")
                break
            except Exception as e:
                print(f"[Worker {self.rollout_worker_id}] Error getting response: {e}")
                break

    async def run_all_benchmark_tasks(
        self, send_queue: Queue, recv_queue: Queue
    ) -> List[EnvResult]:
        if not self.benchmark_specs:
            raise ValueError("Benchmark specifications not set")

        async def _run_all_tasks_inner():
            if self.submission_container_address is None:
                raise ValueError("Submission container address not set")

            self.eval_start = time.time()

            all_task_specs = []
            for benchmark_spec in self.benchmark_specs:
                logger.info(
                    "Running all tasks for benchmark: %s", benchmark_spec.benchmark_name
                )
                task_specs = self.env_manager.get_benchmark_envs(benchmark_spec)
                all_task_specs.extend(task_specs)
                logger.info(
                    "Discovered %d tasks for %s", len(task_specs), benchmark_spec
                )

            logger.info("Will evaluate %d total tasks", len(all_task_specs))

            task_results = []
            for i, task_spec in enumerate(all_task_specs):
                logger.info(
                    "Starting task %d/%d: %s", i + 1, len(all_task_specs), task_spec
                )
                try:
                    task_result = await self.run_env(task_spec, send_queue, recv_queue)
                    task_results.append(task_result)
                    logger.info(
                        "Completed task %s: success_rate=%f mean_reward=%f mean_steps=%f",
                        task_spec,
                        task_result.success_rate,
                        task_result.mean_reward,
                        task_result.mean_steps,
                    )
                except Exception:
                    logger.exception("Failed to run task %s", task_spec)
                    continue

            self.eval_end = time.time()
            self._log_evaluation_summary(task_results)

            # Clean up episode loggers before general cleanup
            for logger_instance in self.episode_loggers.values():
                try:
                    if hasattr(logger_instance, "cleanup"):
                        logger_instance.cleanup()
                except Exception as e:
                    logger.warning(f"Failed to cleanup episode logger: {e}")
            self.episode_loggers.clear()

            # Clean up after evaluation
            self.cleanup()

            # Additional cleanup for task results
            for result in task_results:
                try:
                    if hasattr(result, "episodes"):
                        for episode in result.episodes:
                            if hasattr(episode, "clear"):
                                episode.clear()
                except Exception:
                    pass

            return task_results

        try:
            async with capnp.kj_loop():
                return await _run_all_tasks_inner()
        finally:
            # Ensure cleanup happens even on exception
            self.cleanup()

    async def run_env(
        self, env_spec: EnvSpec, send_queue: Queue, recv_queue: Queue
    ) -> EnvResult:
        env = self.env_manager.make_env(
            env_spec, save_images=False, submission_id=str(self.submission_id)
        )

        # Get parameters from env_spec (which now contains them)
        episodes_per_task = env_spec.episodes_per_task
        max_steps_per_episode = env_spec.max_episode_steps

        # Create episode logger for this environment
        logging_config = LoggingConfig(
            episode_log_interval=self.episode_log_interval,
            step_log_interval=self.step_log_interval,
            enable_r2_upload=self.r2_config is not None,
            r2_config=self.r2_config,
            database_url=self.database_url,
        )

        # Generate a unique task ID combining benchmark and env names
        task_uid = getattr(env_spec, "task_idx", int(time.time()))
        task_id = f"{env_spec.benchmark_name}_{env_spec.env_name}_{task_uid}"

        episode_logger = EpisodeLogger(
            config=logging_config,
            submission_id=str(self.submission_id),
            # TODO: rename rollout_worker_id to job_id in RolloutWorker
            job_id=self.rollout_worker_id,  # Using worker ID as job ID for now
            task_id=task_id,
            env_name=env_spec.env_name,
            benchmark_name=env_spec.benchmark_name,
        )

        # Store logger for this environment
        env_key = f"{env_spec.benchmark_name}_{env_spec.env_name}_{task_uid}"
        self.episode_loggers[env_key] = episode_logger

        episodes = []
        for episode_id in range(episodes_per_task):
            try:
                episode_result = await self.run_episode(
                    env,
                    env_spec,
                    episode_id,
                    max_steps_per_episode,
                    send_queue,
                    recv_queue,
                    episode_logger,
                )
                episodes.append(episode_result)
            except Exception:
                logger.exception("Failed episode %d for %s", episode_id, env_spec)
                continue

        env.close()

        # Clean up environment resources
        del env
        gc.collect()

        # Clean up episode logger after use
        if env_key in self.episode_loggers:
            try:
                self.episode_loggers[env_key].cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup episode logger for {env_key}: {e}")
            del self.episode_loggers[env_key]

        return EnvResult.from_episodes(env_spec, episodes)

    async def run_episode(
        self,
        env,
        env_spec,
        episode_id,
        max_steps,
        send_queue: Queue,
        recv_queue: Queue,
        episode_logger: Optional[EpisodeLogger] = None,
    ):
        logger.info("Starting episode %d for env %s", episode_id, env_spec)

        # Start episode logging if logger is available
        if episode_logger:
            episode_logger.start_episode(episode_id)

        # Reset may return (obs, info) or just obs depending on gym
        res = env.reset()
        if isinstance(res, tuple) and len(res) == 2:
            observation, info = res
        else:
            observation, info = res, {}

        done = False
        step_count = 0
        total_reward = 0.0
        episode_steps = []
        episode_start = time.time()

        while not done and step_count < max_steps:
            try:
                if step_count == 0:
                    logger.debug(
                        "Episode first obs type=%s shape=%s",
                        type(observation),
                        getattr(observation, "shape", None),
                    )

                if not hasattr(observation, "tobytes"):
                    # if observation is a dict (gym.Dict), flatten to concatenated array or handle appropriately
                    # simple fallback: try to convert dict of arrays to concatenated vector
                    if isinstance(observation, dict):
                        obs_arr = np.concatenate(
                            [
                                np.asarray(v).ravel()
                                for _, v in sorted(observation.items())
                            ]
                        )
                    else:
                        obs_arr = np.asarray(observation)
                else:
                    obs_arr = np.asarray(observation)

                rpc_msg = RPCRequest.create_act(obs_arr)
                await send_queue.put_async(rpc_msg)
                resp = await recv_queue.get_async()
                action_tensor = resp.result
                if isinstance(action_tensor, torch.Tensor):
                    action_arr = action_tensor.detach().cpu().numpy()
                else:
                    action_arr = np.asarray(action_tensor)

                # Ensure action_arr is a proper numpy array
                if not isinstance(action_arr, np.ndarray):
                    action_arr = np.asarray(action_arr)

                # Ensure it's not a 0-d array that might cause issues
                if action_arr.ndim == 0:
                    action_arr = np.array([action_arr.item()])

                # Step env
                step_result = env.step(action_arr)
                # compat with gym >=0.26 (terminated, truncated) vs older (done)
                if len(step_result) == 5:
                    observation, reward, terminated, truncated, info = step_result
                    done = bool(terminated or truncated)
                else:
                    observation, reward, done, info = step_result
                    done = bool(done)

                total_reward += float(reward)
                step_count += 1

                # Only keep minimal step info to reduce memory usage
                episode_steps.append(
                    {"step": step_count, "reward": float(reward), "done": bool(done)}
                )

                # Log step data if logger is available
                if episode_logger:
                    # Capture observations if the environment wrapper has the method
                    observations = None
                    logger.debug(
                        f"Environment type: {type(env)}, has capture method: {hasattr(env, 'capture_and_save_images')}"
                    )

                    # Check if the wrapped environment has the method
                    if hasattr(env, "env") and hasattr(
                        env.env, "capture_and_save_images"
                    ):
                        logger.debug(
                            f"Wrapped environment type: {type(env.env)}, has capture method: True"
                        )
                        try:
                            images_hwc, camera_names = env.env.capture_and_save_images()
                            observations = list(zip(images_hwc, camera_names))
                            logger.info(
                                f"Captured {len(images_hwc)} images from cameras: {camera_names}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not capture images from wrapped env: {e}"
                            )
                    elif hasattr(env, "capture_and_save_images"):
                        try:
                            images_hwc, camera_names = env.capture_and_save_images()
                            observations = list(zip(images_hwc, camera_names))
                            logger.info(
                                f"Captured {len(images_hwc)} images from cameras: {camera_names}"
                            )
                        except Exception as e:
                            logger.warning(f"Could not capture images: {e}")
                    else:
                        logger.warning(
                            "Environment does not have capture_and_save_images method"
                        )

                    await episode_logger.log_step(
                        step=step_count,
                        action=action_arr,
                        reward=float(reward),
                        done=done,
                        truncated=False,  # Add truncated detection if available
                        observations=observations,
                        info=info,
                    )

                if step_count % LOGGING_INTERVAL == 0:
                    logger.info(
                        "Episode %d progress: %d/%d steps, total_reward=%.3f",
                        episode_id,
                        step_count,
                        max_steps,
                        total_reward,
                    )

            except Exception:
                logger.exception(
                    "Error during step %d in episode %d", step_count, episode_id
                )
                break

        episode_duration = time.time() - episode_start
        success = info.get("success", False) if info else False

        # Only keep essential episode info to reduce memory
        episode_result = EpisodeResult(
            episode_id=episode_id,
            env_spec=env_spec,
            reward=total_reward,
            steps=step_count,
            success=success,
            info={"duration": episode_duration},  # Removed episode_steps to save memory
        )

        # Clear episode steps data
        episode_steps.clear()
        del episode_steps
        gc.collect()

        # End episode logging if logger is available
        if episode_logger:
            await episode_logger.end_episode(
                total_reward=total_reward,
                success=success,
                extra_metrics={"duration": episode_duration, "final_info": info},
            )

        logger.info(
            "Episode %d completed: steps=%d reward=%.2f success=%s",
            episode_id,
            step_count,
            total_reward,
            episode_result.success,
        )
        return episode_result

    async def run_episodes(self, env_specs: List[EnvSpec]) -> List[EnvResult]:
        self.eval_start = time.time()
        env_results = []
        for i, env_spec in enumerate(env_specs):
            try:
                env_result = await self.run_env(env_spec)
                env_results.append(env_result)
                logger.info("Environment %s completed", env_spec)
            except Exception:
                logger.exception("Failed env %s", env_spec)
                continue
        self.eval_end = time.time()
        self._log_evaluation_summary(env_results)
        return env_results

    def _log_evaluation_summary(self, env_results: List[EnvResult]):
        if not env_results:
            logger.warning("No environment results to summarize")
            return

        total_episodes = sum(len(er.episodes) for er in env_results)
        mean_success_rate = sum(er.success_rate for er in env_results) / len(
            env_results
        )
        mean_reward = sum(er.mean_reward for er in env_results) / len(env_results)
        mean_steps = sum(er.mean_steps for er in env_results) / len(env_results)
        eval_duration = (
            (self.eval_end - self.eval_start)
            if (self.eval_start and self.eval_end)
            else 0.0
        )

        logger.info("=== Evaluation Summary ===")
        logger.info("Worker: %s", self.rollout_worker_id)
        logger.info("Benchmarks: %s", self.benchmark_specs)
        logger.info("Environments completed: %d", len(env_results))
        logger.info("Total episodes: %d", total_episodes)
        logger.info("Mean success rate: %.3f", mean_success_rate)
        logger.info("Mean reward: %.3f", mean_reward)
        logger.info("Mean steps: %.1f", mean_steps)
        logger.info("Evaluation duration: %.1fs", eval_duration)
        logger.info("========================")

    def get_status(self) -> Dict[str, Any]:
        return {
            "worker_id": str(self.rollout_worker_id),
            "submission_address": self.submission_container_address,
            "benchmark_specs": [str(spec) for spec in self.benchmark_specs],
            "eval_start": self.eval_start,
            "eval_end": self.eval_end,
        }
