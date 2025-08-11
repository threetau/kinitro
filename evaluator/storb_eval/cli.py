from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Agent imports removed - now using submission format
from .runner import EvalConfig, evaluate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Storb RL Eval Runner")
    sub = p.add_subparsers(dest="command")

    # eval subcommand
    pe = sub.add_parser("eval", help="Run evaluation")
    pe.add_argument(
        "--task",
        default="push-v3",
        help="MetaWorld task name, e.g. push-v3, reach-v3, door-open-v3",
    )
    pe.add_argument(
        "--episodes", type=int, default=10, help="Number of episodes to evaluate"
    )
    pe.add_argument(
        "--max-steps", type=int, default=200, help="Maximum steps per episode"
    )
    pe.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of Ray workers to parallelize rollouts",
    )
    pe.add_argument(
        "--submission",
        type=str,
        default=None,
        help="Path to miner submission directory (contains agent.py + model files). If not provided, uses default SimpleVLA agent.",
    )
    pe.add_argument(
        "--goal",
        type=str,
        default="push the block to the goal",
        help="Text goal for the planner",
    )
    pe.add_argument("--seed", type=int, default=0, help="Random seed")
    pe.add_argument(
        "--render", action="store_true", help="Render the environment in real time"
    )
    pe.add_argument(
        "--render-mode",
        type=str,
        default=None,
        help="Render mode, e.g., human or rgb_array",
    )
    pe.add_argument(
        "--fps", type=int, default=30, help="Frames per second when rendering"
    )
    pe.add_argument("--json", action="store_true", help="Print result as JSON only")

    # make-submission subcommand
    pm = sub.add_parser(
        "make-submission", help="Create a default SimpleVLA submission directory"
    )
    pm.add_argument(
        "--out", type=str, required=True, help="Output directory for the submission"
    )
    pm.add_argument("--seed", type=int, default=0, help="Random seed")
    pm.add_argument("--save-weights", action="store_true", help="Save random weights to model.npz")

    return p


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "make-submission":
        from shutil import copytree
        import tempfile
        
        out_dir = Path(args.out)
        default_submission = Path(__file__).parent.parent / "default_submission"
        
        # Copy the default submission to output directory
        if out_dir.exists():
            raise ValueError(f"Output directory already exists: {out_dir}")
        
        copytree(default_submission, out_dir)
        
        if args.save_weights:
            # Create a temporary agent with random weights and save them
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                # Load the agent to initialize random weights, then save them
                from .agent_loader import AgentLoader
                temp_agent = AgentLoader.load_agent(
                    out_dir,
                    observation_size=100,  # Placeholder - will be overridden during actual eval
                    action_size=4,         # Placeholder
                    seed=args.seed,
                )
                # Save the weights if the agent supports it
                if hasattr(temp_agent, 'save_weights'):
                    temp_agent.save_weights(out_dir / "model.npz")
        
        print(f"Created submission directory: {out_dir}")
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
            submission_path=args.submission,
            goal_text=args.goal,
            seed=args.seed,
            render=args.render,
            render_mode=args.render_mode,
            fps=args.fps,
        )
        result = evaluate(cfg)
        if args.json:
            print(json.dumps(result))
        else:
            print("Storb RL Eval Result")
            print(f"  task: {cfg.task_name}")
            if args.submission:
                print(f"  submission: {Path(args.submission).name}")
            else:
                print("  agent: default (SimpleVLA)")
            print(f"  episodes: {int(result['episodes'])}")
            print(f"  avg_return: {result['avg_return']:.3f}")
            print(f"  success_rate: {result['success_rate']:.3f}")


if __name__ == "__main__":
    main()
