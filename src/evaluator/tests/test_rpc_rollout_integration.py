#!/usr/bin/env python3
"""
Full integration test with real RPC server and worker.
Tests the complete pipeline: Agent -> RPC Server -> RPC Client -> Worker -> Environment
"""

import asyncio
import logging
import multiprocessing
import time
import traceback
from typing import List

import capnp
import pytest
from kinitro_eval.roullout.envs import BenchmarkSpec, EnvManager, EnvSpec
from kinitro_eval.rpc.client import AgentClient
from kinitro_eval.rpc.server import start_server
from test_scripts.random_agent import RandomActionAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def run_agent_server(port=8001):
    """Run agent server in a separate process."""
    try:
        # Create random action agent
        agent = RandomActionAgent(seed=42)

        # Start the server (this blocks)
        logger.info(f"Starting agent server on port {port}")
        start_server(agent, address="127.0.0.1", port=port)

    except KeyboardInterrupt:
        logger.info("Agent server stopped")
    except Exception as e:
        logger.error(f"Agent server error: {e}")


class RealRolloutWorker:
    """Worker that uses real RPC client to connect to agent server."""

    def __init__(
        self,
        rollout_worker_id: int,
        rollout_worker_address: str,
        submission_container_host: str,
        submission_container_port: int,
        benchmark_specs: List[BenchmarkSpec],
    ):
        self.rollout_worker_id = rollout_worker_id
        self.rollout_worker_address = rollout_worker_address
        self.submission_container_host = submission_container_host
        self.submission_container_port = submission_container_port
        self.benchmark_specs = benchmark_specs

        # Create real agent client
        self.agent = AgentClient(
            host=self.submission_container_host, port=self.submission_container_port
        )

        # Environment manager
        self.env_manager = EnvManager()

        # Evaluation configuration
        self.episodes_per_task = 1
        self.max_steps_per_episode = 25  # Reduced for faster testing

        # Timing
        self.eval_start = None

    async def connect_to_agent(self):
        """Connect to the agent server."""
        await self.agent.connect()
        logger.info(f"Worker {self.rollout_worker_id} connected to agent")

    async def disconnect_from_agent(self):
        """Disconnect from the agent server."""
        await self.agent.close()
        logger.info(f"Worker {self.rollout_worker_id} disconnected from agent")

    async def run_episode(self, env, agent, env_spec, episode_id, max_steps):
        """Run a single episode with real RPC communication."""
        logger.info(f"Starting episode {episode_id} for {env_spec}")

        observation, info = env.reset()
        done = False
        step_count = 0
        total_reward = 0.0

        episode_start = time.time()

        while not done and step_count < max_steps:
            try:
                # Get action from agent via RPC (async)
                action = await agent.act(observation)

                # Convert tensor to numpy array if needed
                if hasattr(action, "numpy"):
                    action = action.numpy()
                elif hasattr(action, "detach"):
                    action = action.detach().cpu().numpy()

                # Step environment
                observation, reward, terminated, truncated, info = env.step(action)

                total_reward += float(reward)
                step_count += 1
                done = terminated or truncated

            except Exception as e:
                logger.error(
                    f"Error during step {step_count} in episode {episode_id}: {e}"
                )
                break

        episode_duration = time.time() - episode_start

        logger.info(
            f"Episode {episode_id} completed: {step_count} steps, "
            f"reward={total_reward:.2f}, duration={episode_duration:.2f}s"
        )

        return {
            "episode_id": episode_id,
            "env_spec": env_spec,
            "steps": step_count,
            "reward": total_reward,
            "duration": episode_duration,
            "success": info.get("success", 0.0) if info else 0.0,
        }

    async def run_environment_episodes(self, env_spec: EnvSpec):
        """Run episodes for a specific environment using real RPC."""
        logger.info(f"Running episodes for environment: {env_spec}")

        # Create environment
        env = self.env_manager.make_env(env_spec)

        # Run episodes
        episodes = []
        for episode_id in range(self.episodes_per_task):
            try:
                episode_result = await self.run_episode(
                    env, self.agent, env_spec, episode_id, self.max_steps_per_episode
                )
                episodes.append(episode_result)
            except Exception as e:
                logger.error(f"Failed to run episode {episode_id} for {env_spec}: {e}")
                continue

        # Clean up environment
        env.close()

        return {
            "env_spec": env_spec,
            "episodes": episodes,
            "avg_reward": sum(e["reward"] for e in episodes) / len(episodes)
            if episodes
            else 0,
            "avg_steps": sum(e["steps"] for e in episodes) / len(episodes)
            if episodes
            else 0,
            "success_rate": sum(e["success"] for e in episodes) / len(episodes)
            if episodes
            else 0,
        }

    async def run_benchmark_tasks(self, max_tasks=None):
        """Run agent through benchmark tasks using real RPC."""
        if not self.benchmark_specs:
            raise ValueError("Benchmark specifications not set")

        self.eval_start = time.time()

        # Collect tasks from benchmarks
        all_task_specs = []
        for benchmark_spec in self.benchmark_specs:
            logger.info(f"Getting tasks for benchmark: {benchmark_spec.benchmark_name}")
            task_specs = self.env_manager.get_benchmark_envs(benchmark_spec)

            # Limit tasks for testing
            if max_tasks:
                task_specs = task_specs[:max_tasks]

            all_task_specs.extend(task_specs)
            logger.info(
                f"Added {len(task_specs)} tasks from {benchmark_spec.benchmark_name}"
            )

        logger.info(f"Total tasks to run: {len(all_task_specs)}")

        # Run tasks
        all_results = []
        for i, env_spec in enumerate(all_task_specs):
            logger.info(f"Processing task {i + 1}/{len(all_task_specs)}: {env_spec}")
            try:
                result = await self.run_environment_episodes(env_spec)
                all_results.append(result)
                logger.info(f"Task completed: avg_reward={result['avg_reward']:.2f}")
            except Exception as e:
                logger.error(f"Failed to run task {env_spec}: {e}")
                continue

        # Summary
        total_time = time.time() - self.eval_start
        total_episodes = sum(len(r["episodes"]) for r in all_results)
        avg_reward = (
            sum(r["avg_reward"] for r in all_results) / len(all_results)
            if all_results
            else 0
        )

        logger.info(
            f"Evaluation completed: {len(all_results)} tasks, {total_episodes} episodes, "
            f"avg_reward={avg_reward:.2f}, time={total_time:.2f}s"
        )

        return all_results


class TestRealRPCIntegration:
    """Test real RPC integration between agent server and worker."""

    @pytest.fixture
    def agent_server_process(self):
        """Start agent server in separate process."""
        port = 8001
        process = multiprocessing.Process(target=run_agent_server, args=(port,))
        process.start()

        # Wait for server to start
        time.sleep(2)

        yield port

        # Cleanup
        process.terminate()
        process.join(timeout=5)
        if process.is_alive():
            process.kill()

    @pytest.fixture
    def mt1_benchmark(self):
        """Create MT1 benchmark spec."""
        return BenchmarkSpec(
            provider="metaworld", benchmark_name="MT1", config={"env_name": "reach-v3"}
        )

    @pytest.fixture
    def real_worker(self, mt1_benchmark, agent_server_process):
        """Create worker that connects to real agent server."""
        port = agent_server_process
        return RealRolloutWorker(
            rollout_worker_id=1,
            rollout_worker_address="127.0.0.1:8000",
            submission_container_host="127.0.0.1",
            submission_container_port=port,
            benchmark_specs=[mt1_benchmark],
        )

    @pytest.mark.asyncio
    async def test_agent_server_connection(self, real_worker):
        """Test that worker can connect to real agent server."""
        async with capnp.kj_loop():
            await real_worker.connect_to_agent()

            # Test basic RPC call
            mock_obs = {"base": [0.0] * 39}
            action = await real_worker.agent.act(mock_obs)

            assert action is not None, "Should receive action from agent"
            assert len(action) == 4, f"Action should have 4 elements, got {len(action)}"

            await real_worker.disconnect_from_agent()

    @pytest.mark.asyncio
    async def test_single_task_with_rpc(self, real_worker, mt1_benchmark):
        """Test running a single task with real RPC communication."""
        async with capnp.kj_loop():
            await real_worker.connect_to_agent()

            try:
                # Get first task
                task_specs = real_worker.env_manager.get_benchmark_envs(mt1_benchmark)
                env_spec = task_specs[0]

                # Run task
                result = await real_worker.run_environment_episodes(env_spec)

                assert result is not None, "Should return result"
                assert len(result["episodes"]) == 1, "Should run 1 episode"
                assert result["episodes"][0]["steps"] > 0, "Episode should have steps"
                assert isinstance(result["avg_reward"], float), (
                    "Should have average reward"
                )

            finally:
                await real_worker.disconnect_from_agent()

    @pytest.mark.asyncio
    async def test_multiple_tasks_with_rpc(self, real_worker):
        """Test running multiple tasks with real RPC communication."""
        async with capnp.kj_loop():
            await real_worker.connect_to_agent()

            try:
                # Run first 3 tasks from MT1
                results = await real_worker.run_benchmark_tasks(max_tasks=3)

                assert len(results) == 3, f"Should complete 3 tasks, got {len(results)}"
                assert all(len(r["episodes"]) > 0 for r in results), (
                    "All tasks should have episodes"
                )

                # Check results structure
                for result in results:
                    assert "env_spec" in result, "Should have env_spec"
                    assert "episodes" in result, "Should have episodes"
                    assert "avg_reward" in result, "Should have avg_reward"
                    assert result["avg_reward"] > 0, "Should have positive reward"

            finally:
                await real_worker.disconnect_from_agent()

    @pytest.mark.asyncio
    async def test_worker_runs_all_mt1_tasks(self, real_worker):
        """Test that worker runs through all MT1 tasks with real RPC agent."""
        async with capnp.kj_loop():
            await real_worker.connect_to_agent()

            try:
                # Run all MT1 tasks (limited to first 5 for speed)
                results = await real_worker.run_benchmark_tasks(max_tasks=5)

                assert len(results) > 0, "Should return results"
                assert all(isinstance(r, dict) for r in results), (
                    "All results should be dictionaries"
                )

                # Check that episodes were run
                total_episodes = sum(len(result["episodes"]) for result in results)
                assert total_episodes > 0, "Should have run some episodes"

                # Check result structure
                for result in results:
                    assert "env_spec" in result, "Should have env_spec"
                    assert "episodes" in result, "Should have episodes"
                    assert "avg_reward" in result, "Should have avg_reward"
                    assert len(result["episodes"]) > 0, "Should have episodes"

                    # Check episode structure
                    for episode in result["episodes"]:
                        assert "episode_id" in episode, "Episode should have ID"
                        assert "steps" in episode, "Episode should have steps"
                        assert "reward" in episode, "Episode should have reward"
                        assert isinstance(episode["reward"], float), (
                            "Reward should be float"
                        )

                logger.info(
                    f"Successfully ran {len(results)} environments with {total_episodes} total episodes"
                )

            finally:
                await real_worker.disconnect_from_agent()


async def run_standalone_test():
    """Standalone test function for manual testing."""
    print("🚀 Starting Real RPC Integration Test")
    print("=" * 60)

    # Start agent server in background
    port = 8001
    server_process = multiprocessing.Process(target=run_agent_server, args=(port,))
    server_process.start()

    try:
        # Wait for server to start
        print("⏳ Waiting for agent server to start...")
        time.sleep(3)

        # Run test within KJ loop
        async with capnp.kj_loop():
            # Create benchmark and worker
            mt1_benchmark = BenchmarkSpec(
                provider="metaworld",
                benchmark_name="MT1",
                config={"env_name": "reach-v3"},
            )

            worker = RealRolloutWorker(
                rollout_worker_id=1,
                rollout_worker_address="127.0.0.1:8000",
                submission_container_host="127.0.0.1",
                submission_container_port=port,
                benchmark_specs=[mt1_benchmark],
            )

            # Connect to agent
            print("🔌 Connecting to agent server...")
            await worker.connect_to_agent()

            # Test single RPC call
            print("🎯 Testing single RPC call...")
            mock_obs = {"base": [0.0] * 39}
            action = await worker.agent.act(mock_obs)
            print(f"   Received action: {action}")

            # Run a few tasks
            print("🏃 Running 3 tasks from MT1...")
            results = await worker.run_benchmark_tasks(max_tasks=3)

            # Summary
            print("\n📊 Results Summary:")
            print(f"   Tasks completed: {len(results)}")
            total_episodes = sum(len(r["episodes"]) for r in results)
            print(f"   Total episodes: {total_episodes}")
            avg_reward = sum(r["avg_reward"] for r in results) / len(results)
            print(f"   Average reward: {avg_reward:.2f}")

            # Disconnect
            await worker.disconnect_from_agent()
            print("✅ Real RPC integration test completed successfully!")

    except Exception as e:
        print(f"❌ Test failed: {e}")
        traceback.print_exc()

    finally:
        # Cleanup server
        print("🛑 Stopping agent server...")
        server_process.terminate()
        server_process.join(timeout=5)
        if server_process.is_alive():
            server_process.kill()


if __name__ == "__main__":
    # Run standalone test
    asyncio.run(run_standalone_test())
