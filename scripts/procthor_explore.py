#!/usr/bin/env python3
"""ProcTHOR environment exploration script with visualization."""

from __future__ import annotations

import argparse
import time

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from kinitro.environments.procthor import ProcTHOREnvironment
from kinitro.rl_interface import CanonicalAction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ProcTHOR environment exploration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for task generation")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to run")
    parser.add_argument("--sleep", type=float, default=0.1, help="Sleep between steps (seconds)")
    parser.add_argument("--no-display", action="store_true", help="Disable visual display")
    parser.add_argument(
        "--random-actions", action="store_true", help="Use random actions instead of scripted"
    )
    parser.add_argument(
        "--save-video", type=str, default=None, help="Save video to file (e.g., output.mp4)"
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Run headless by default; only show display window if explicitly requested
    # (i.e., when --no-display is NOT set and we have cv2)
    show_window = HAS_CV2 and not args.no_display
    headless = not show_window

    if show_window:
        print("Running with display window (use --no-display for headless)")

    print("Creating ProcTHOR environment...")
    env = ProcTHOREnvironment(
        image_size=(400, 400),
        field_of_view=90,
        headless=headless,
    )

    print(f"Generating task with seed={args.seed}...")
    task_config = env.generate_task(seed=args.seed)

    # Extract task info
    task_spec = task_config.domain_randomization.get("task_spec", {})
    task_prompt = task_spec.get("task_prompt", "No task")
    task_type = task_spec.get("task_type", "unknown")
    target_object = task_spec.get("target_object_type", "unknown")

    print(f"\n{'=' * 60}")
    print(f"Task: {task_prompt}")
    print(f"Type: {task_type}")
    print(f"Target: {target_object}")
    print(f"{'=' * 60}\n")

    print("Resetting environment...")
    obs = env.reset(task_config)
    print(f"Initial position: {obs.ee_pos_m}")
    print(f"RGB cameras: {list(obs.rgb.keys())}")

    # Setup video writer if requested
    video_writer = None
    if args.save_video and HAS_CV2:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(args.save_video, fourcc, 10.0, (400, 400))
        print(f"Recording video to: {args.save_video}")

    # Setup display window (reuse show_window from above)
    if show_window:
        cv2.namedWindow("ProcTHOR", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ProcTHOR", 600, 600)

    print(f"\nRunning {args.steps} steps...")
    print("Actions: Move forward, strafe, rotate, look up/down")
    print("-" * 60)

    total_reward = 0.0

    for step_idx in range(args.steps):
        # Generate action
        if args.random_actions:
            twist = np.random.uniform(-0.5, 0.5, 6).astype(np.float32)
            gripper = float(np.random.random() > 0.8)  # Occasionally try to grab
        else:
            # Scripted exploration pattern
            twist = np.zeros(6, dtype=np.float32)
            phase = step_idx % 20

            if phase < 5:
                # Move forward
                twist[2] = 0.8
            elif phase < 8:
                # Strafe right
                twist[0] = 0.6
            elif phase < 12:
                # Rotate right
                twist[5] = 0.8
            elif phase < 15:
                # Look down
                twist[3] = 0.5
            elif phase < 18:
                # Move forward
                twist[2] = 0.6
            else:
                # Rotate left
                twist[5] = -0.8

            # Try to grab periodically
            gripper = 1.0 if step_idx % 15 == 14 else 0.0

        action = CanonicalAction(twist_ee_norm=twist.tolist(), gripper_01=gripper)
        obs, reward, done, info = env.step(action)
        total_reward += reward

        # Display frame
        if obs.rgb and "ego" in obs.rgb:
            frame = np.array(obs.rgb["ego"], dtype=np.uint8)

            if show_window:
                # Add text overlay
                display_frame = frame.copy()
                cv2.putText(
                    display_frame,
                    f"Step: {step_idx}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    display_frame,
                    f"Task: {task_prompt[:40]}",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    1,
                )
                cv2.putText(
                    display_frame,
                    f"Reward: {total_reward:.1f}",
                    (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )

                # Convert RGB to BGR for OpenCV
                display_frame = cv2.cvtColor(display_frame, cv2.COLOR_RGB2BGR)
                cv2.imshow("ProcTHOR", display_frame)

                key = cv2.waitKey(1)
                if key == ord("q"):
                    print("\nQuitting...")
                    break

            if video_writer:
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Print status
        action_name = info.get("last_action", "unknown")
        action_success = info.get("last_action_success", False)
        status = "OK" if action_success else "FAIL"

        if step_idx % 10 == 0 or done or reward > 0:
            print(
                f"Step {step_idx:3d}: action={action_name:15s} [{status:4s}] reward={reward:.1f} pos={obs.ee_pos_m[0]:.2f},{obs.ee_pos_m[2]:.2f}"
            )

        if done:
            success = env.get_success()
            print(f"\n{'=' * 60}")
            print(f"Episode finished! Success: {success}")
            print(f"Total reward: {total_reward:.1f}")
            print(f"{'=' * 60}")
            break

        if args.sleep > 0:
            time.sleep(args.sleep)

    # Cleanup
    if video_writer:
        video_writer.release()
        print(f"Video saved to: {args.save_video}")

    if show_window:
        cv2.destroyAllWindows()

    env.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
