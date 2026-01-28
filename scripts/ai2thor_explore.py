from __future__ import annotations

import argparse
import time

import numpy as np

from kinitro.environments import get_environment
from kinitro.environments.ai2thor_env import AI2ThorManipulationEnvironment
from kinitro.rl_interface import CanonicalAction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AI2-THOR manipulation exploration script")
    parser.add_argument("--env-id", default="ai2thor/manip-v0")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--scene", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env: AI2ThorManipulationEnvironment = get_environment(args.env_id)
    if args.scene and hasattr(env, "_scene"):
        env._scene = args.scene

    task_config = env.generate_task(seed=args.seed)
    obs = env.reset(task_config)
    print("Initial obs keys:", obs.model_dump().keys())
    print("RGB cameras:", list(obs.rgb.keys()))

    for step_idx in range(args.steps):
        twist = np.zeros(6, dtype=np.float32)
        if step_idx % 4 == 0:
            twist[2] = 1.0
        elif step_idx % 4 == 1:
            twist[0] = 1.0
        elif step_idx % 4 == 2:
            twist[1] = 0.5
        else:
            twist[3] = 0.5

        action = CanonicalAction(twist_ee_norm=twist.tolist(), gripper_01=0.0)
        obs, _reward, done, info = env.step(action)
        print(
            f"step={step_idx:03d} action={info.get('last_action')} "
            f"success={info.get('last_action_success')} rgb={bool(obs.rgb)}"
        )
        if done:
            break
        if args.sleep > 0:
            time.sleep(args.sleep)

    env.close()


if __name__ == "__main__":
    main()
