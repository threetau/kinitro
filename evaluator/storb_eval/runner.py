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

from .agent_interface import AgentInterface
from .agent_loader import AgentLoader
from .envs import EnvSpec, action_size, make_env, obs_size


@dataclass
class EvalConfig:
    task_name: str = "push-v3"
    max_episode_steps: int = 200
    num_episodes: int = 10
    num_workers: int = 1
    submission_path: Optional[str] = None  # Path to miner submission directory
    goal_text: str = "push the block to the goal"
    seed: int = 0
    render: bool = False
    render_mode: Optional[str] = None
    fps: int = 30


@ray.remote
class RolloutWorker:
    def __init__(
        self,
        spec: EnvSpec,
        submission_path: Optional[str],
        goal_text: str,
        seed: int,
        render: bool,
        fps: int,
    ):
        env = make_env(spec)
        self.env = env
        self.goal_text = goal_text
        self.render = render
        self.fps = max(1, int(fps))

        # Load agent from submission directory
        observation_dim = obs_size(env)
        action_dim = action_size(env)
        
        if submission_path is not None:
            # Load miner submission directory
            self.agent = AgentLoader.load_agent(
                submission_path,
                observation_size=observation_dim,
                action_size=action_dim,
                seed=seed,
            )
        else:
            # Use default SimpleVLA submission for testing
            from pathlib import Path
            default_submission = Path(__file__).parent.parent / "default_submission"
            self.agent = AgentLoader.load_agent(
                default_submission,
                observation_size=observation_dim,
                action_size=action_dim,
                seed=seed,
            )

    def run_episodes(self, num_episodes: int, max_steps: int) -> Dict[str, float]:
        returns: List[float] = []
        successes = 0
        for ep_idx in range(num_episodes):
            print(f"[Worker] Episode {ep_idx+1}/{num_episodes} start", flush=True)
            obs, _ = self.env.reset()
            self.agent.reset()  # Reset agent state for new episode
            total_reward = 0.0
            for step_idx in range(max_steps):
                action = self.agent.act(
                    np.asarray(obs, dtype=np.float32), goal_text=self.goal_text
                )
                obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += float(reward)
                if self.render:
                    try:
                        self.env.render()
                    except Exception:
                        pass
                    # Add small delay for smoother rendering

                if (step_idx + 1) % 50 == 0:
                    print(
                        f"[Worker] Episode {ep_idx+1} step {step_idx+1}/{max_steps} reward_so_far={total_reward:.3f}",
                        flush=True,
                    )
                if terminated or truncated:
                    # MetaWorld envs report success via info.get('success', 0.0)
                    successes += int(info.get("success", 0.0) > 0.0)
                    break
            returns.append(total_reward)
            print(
                f"[Worker] Episode {ep_idx+1} done: return={total_reward:.3f}",
                flush=True,
            )

        avg_return = float(np.mean(returns)) if returns else 0.0
        success_rate = (
            float(successes) / float(num_episodes) if num_episodes > 0 else 0.0
        )
        return {"avg_return": avg_return, "success_rate": success_rate}


def maybe_init_ray() -> None:
    if not ray.is_initialized():
        ray.init(
            ignore_reinit_error=True, 
            include_dashboard=False, 
            log_to_driver=True,
            _system_config={"worker_register_timeout_seconds": 30}
        )


def evaluate(config: EvalConfig) -> Dict[str, float]:
    maybe_init_ray()

    spec = EnvSpec(
        task_name=config.task_name,
        max_episode_steps=config.max_episode_steps,
        render_mode=config.render_mode,
    )

    # Pre-install requirements before creating Ray workers to show output
    if config.submission_path is not None:
        from pathlib import Path
        from .agent_loader import AgentLoader
        
        print(f"🔧 Pre-installing requirements for submission: {config.submission_path}")
        submission_dir = Path(config.submission_path)
        requirements_file = submission_dir / "requirements.txt"
        
        if requirements_file.exists():
            # Call the installation logic directly (same as in agent_loader)
            AgentLoader._install_requirements(submission_dir)
            # Prefetch large model assets to prevent long downloads inside Ray actors
            AgentLoader.prefetch_models_if_configured(submission_dir)
        else:
            print("ℹ️  No requirements.txt found in submission")
    else:
        print("ℹ️  Using default submission (no additional requirements)")
    
    print("🚀 Starting evaluation with Ray workers...")
    print("=" * 60)

    # Distribute episodes across workers
    episodes_per_worker = max(1, config.num_episodes // max(1, config.num_workers))
    remainder = config.num_episodes - episodes_per_worker * max(1, config.num_workers)

    workers = [
        RolloutWorker.remote(
            spec,
            config.submission_path,
            config.goal_text,
            config.seed + i,
            config.render,
            config.fps,
        )
        for i in range(config.num_workers)
    ]
    episode_counts = [episodes_per_worker] * config.num_workers
    for i in range(remainder):
        episode_counts[i % config.num_workers] += 1

    results = ray.get(
        [
            w.run_episodes.remote(n, config.max_episode_steps)
            for w, n in zip(workers, episode_counts)
        ]
    )
    avg_return = float(np.mean([r["avg_return"] for r in results])) if results else 0.0
    success_rate = (
        float(np.mean([r["success_rate"] for r in results])) if results else 0.0
    )
    return {
        "avg_return": avg_return,
        "success_rate": success_rate,
        "episodes": float(config.num_episodes),
    }
