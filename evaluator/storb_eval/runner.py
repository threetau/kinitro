"""
Ray-powered evaluation runner.

This module defines a small, self-contained evaluator that can:
 - instantiate a MetaWorld environment
 - load a miner-provided agent from disk (or create a default lightweight agent)
 - run rollouts and compute simple scores (average return and success rate)
 - scale out rollouts across Ray workers for faster evaluation

It intentionally avoids heavy model dependencies to support local development.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import ray  # type: ignore

from .agent import SimpleVLAPolicy, load_agent_from_path
from .envs import EnvSpec, action_size, make_env, obs_size


@dataclass
class EvalConfig:
    task_name: str = "push-v2"
    max_episode_steps: int = 200
    num_episodes: int = 10
    num_workers: int = 1
    agent_path: Optional[str] = None
    goal_text: str = "push the block to the goal"
    seed: int = 0


@ray.remote
class RolloutWorker:
    def __init__(self, spec: EnvSpec, agent_blob: Optional[bytes], goal_text: str, seed: int):
        env = make_env(spec)
        self.env = env
        self.goal_text = goal_text

        # If an agent blob is provided, save to temp and load; otherwise create default
        observation_dim = obs_size(env)
        action_dim = action_size(env)
        if agent_blob is not None:
            import tempfile
            from pathlib import Path

            tmp = tempfile.NamedTemporaryFile(suffix=".npz", delete=False)
            tmp.write(agent_blob)
            tmp.flush()
            tmp.close()
            self.agent = SimpleVLAPolicy.load(tmp.name, observation_size=observation_dim, action_size=action_dim)
            Path(tmp.name).unlink(missing_ok=True)
        else:
            self.agent = load_agent_from_path(None, observation_size=observation_dim, action_size=action_dim, seed=seed)

    def run_episodes(self, num_episodes: int, max_steps: int) -> Dict[str, float]:
        returns: List[float] = []
        successes = 0
        for _ in range(num_episodes):
            obs, _ = self.env.reset()
            total_reward = 0.0
            for _step in range(max_steps):
                action = self.agent.act(np.asarray(obs, dtype=np.float32), goal_text=self.goal_text)
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    # MetaWorld envs report success via info.get('success', 0.0)
                    successes += int(info.get("success", 0.0) > 0.0)
                    break
            returns.append(total_reward)

        avg_return = float(np.mean(returns)) if returns else 0.0
        success_rate = float(successes) / float(num_episodes) if num_episodes > 0 else 0.0
        return {"avg_return": avg_return, "success_rate": success_rate}


def maybe_init_ray() -> None:
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False, log_to_driver=False)


def evaluate(config: EvalConfig) -> Dict[str, float]:
    maybe_init_ray()

    spec = EnvSpec(task_name=config.task_name, max_episode_steps=config.max_episode_steps)

    # Prepare agent payload if provided
    agent_blob: Optional[bytes] = None
    if config.agent_path is not None:
        with open(config.agent_path, "rb") as f:
            agent_blob = f.read()

    # Distribute episodes across workers
    episodes_per_worker = max(1, config.num_episodes // max(1, config.num_workers))
    remainder = config.num_episodes - episodes_per_worker * max(1, config.num_workers)

    workers = [
        RolloutWorker.remote(spec, agent_blob, config.goal_text, config.seed + i) for i in range(config.num_workers)
    ]
    episode_counts = [episodes_per_worker] * config.num_workers
    for i in range(remainder):
        episode_counts[i % config.num_workers] += 1

    results = ray.get([w.run_episodes.remote(n, config.max_episode_steps) for w, n in zip(workers, episode_counts)])
    avg_return = float(np.mean([r["avg_return"] for r in results])) if results else 0.0
    success_rate = float(np.mean([r["success_rate"] for r in results])) if results else 0.0
    return {"avg_return": avg_return, "success_rate": success_rate, "episodes": float(config.num_episodes)}


