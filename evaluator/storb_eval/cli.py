from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agent import create_default_agent
from .runner import EvalConfig, evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Storb RL Eval Runner")
    sub = p.add_subparsers(dest="command")

    # eval subcommand
    pe = sub.add_parser("eval", help="Run evaluation")
    pe.add_argument("--task", default="push-v3", help="MetaWorld task name, e.g. push-v3, reach-v3, door-open-v3")
    pe.add_argument("--episodes", type=int, default=10, help="Number of episodes to evaluate")
    pe.add_argument("--max-steps", type=int, default=200, help="Maximum steps per episode")
    pe.add_argument("--workers", type=int, default=1, help="Number of Ray workers to parallelize rollouts")
    pe.add_argument("--agent", type=str, default=None, help="Path to miner agent weights (.npz). Optional")
    pe.add_argument("--goal", type=str, default="push the block to the goal", help="Text goal for the planner")
    pe.add_argument("--seed", type=int, default=0, help="Random seed")
    pe.add_argument("--json", action="store_true", help="Print result as JSON only")

    # make-agent subcommand
    pm = sub.add_parser("make-agent", help="Create a tiny random agent and save to disk")
    pm.add_argument("--obs", type=int, required=True, help="Observation size expected by the env")
    pm.add_argument("--act", type=int, required=True, help="Action size expected by the env")
    pm.add_argument("--out", type=str, required=True, help="Output path for the .npz agent file")
    pm.add_argument("--seed", type=int, default=0, help="Random seed")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "make-agent":
        agent = create_default_agent(args.obs, args.act, seed=args.seed)
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        agent.save(out)
        print(str(out))
        return

    # Default to eval if no subcommand provided
    if args.command is None or args.command == "eval":
        # Reparse with eval defaults if no subcommand
        if args.command is None:
            args = parser.parse_args(["eval", *([] if argv is None else argv)])
        cfg = EvalConfig(
            task_name=args.task,
            max_episode_steps=args.max_steps,
            num_episodes=args.episodes,
            num_workers=args.workers,
            agent_path=args.agent,
            goal_text=args.goal,
            seed=args.seed,
        )
        result = evaluate(cfg)
        if args.json:
            print(json.dumps(result))
        else:
            print("Storb RL Eval Result")
            print(f"  task: {cfg.task_name}")
            if args.agent:
                print(f"  agent: {Path(args.agent).name}")
            print(f"  episodes: {int(result['episodes'])}")
            print(f"  avg_return: {result['avg_return']:.3f}")
            print(f"  success_rate: {result['success_rate']:.3f}")


if __name__ == "__main__":
    main()


