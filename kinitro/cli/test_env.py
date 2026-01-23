"""Test environment command."""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import typer


def test_env(
    env_id: str = typer.Argument(..., help="Environment ID to test"),
    episodes: int = typer.Option(5, help="Number of episodes to run"),
    record_dir: str | None = typer.Option(
        None, "--record-dir", "-r", help="Directory to save recordings (enables recording)"
    ),
    save_images: bool = typer.Option(
        False, "--save-images", help="Save camera images (can be large)"
    ),
    max_steps: int = typer.Option(500, "--max-steps", help="Maximum steps per episode"),
):
    """
    Test an environment with random actions.

    Useful for verifying environment setup.

    Examples:
        # Basic test
        kinitro test-env metaworld/pick-place-v3

        # Record trajectories to disk
        kinitro test-env metaworld/push-v3 --record-dir ./recordings

        # Record with camera images
        kinitro test-env metaworld/push-v3 --record-dir ./recordings --save-images
    """
    from kinitro.environments import get_environment

    typer.echo(f"Testing environment: {env_id}")

    env = get_environment(env_id)
    typer.echo(f"  Proprioceptive observation shape: {env.observation_shape}")
    typer.echo(f"  Action shape: {env.action_shape}")

    # Check for camera support
    has_cameras = hasattr(env, "num_cameras") and env.num_cameras > 0
    if has_cameras:
        typer.echo(f"  Number of cameras: {env.num_cameras}")
        if hasattr(env, "image_shape"):
            typer.echo(f"  Image shape: {env.image_shape}")

    # Setup recording directory
    recording = record_dir is not None
    if recording:
        record_path = Path(record_dir)
        record_path.mkdir(parents=True, exist_ok=True)

        # Save run metadata
        metadata = {
            "env_id": env_id,
            "episodes": episodes,
            "max_steps": max_steps,
            "save_images": save_images,
            "timestamp": datetime.now().isoformat(),
            "observation_shape": list(env.observation_shape),
            "action_shape": list(env.action_shape),
        }
        if has_cameras and hasattr(env, "image_shape"):
            metadata["image_shape"] = list(env.image_shape)
            metadata["num_cameras"] = env.num_cameras

        with open(record_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        typer.echo(f"  Recording to: {record_path}")

    successes = 0
    total_reward = 0.0

    for ep in range(episodes):
        task_config = env.generate_task(seed=ep)
        obs = env.reset(task_config)

        typer.echo(f"  Episode {ep + 1} initial obs: {obs}")

        # Storage for trajectory
        observations = [obs.copy()]
        actions = []
        rewards = []
        dones = []
        infos = []
        images: dict[str, list[np.ndarray]] = {}

        # Capture initial images if recording
        if recording and save_images and has_cameras and hasattr(env, "get_observation"):
            typer.echo("    Capturing initial images...")
            full_obs = env.get_observation()
            for cam_name, img in full_obs.camera_views.items():
                if cam_name not in images:
                    images[cam_name] = []
                images[cam_name].append(img.copy())
            typer.echo("    Done.")

        ep_reward = 0.0
        steps = 0

        for step_idx in range(max_steps):
            # Random action
            low, high = env.action_bounds
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1

            # Record trajectory data
            if recording:
                observations.append(obs.copy())
                actions.append(action.copy())
                rewards.append(reward)
                dones.append(done)
                # Convert info to serializable format
                infos.append(
                    {k: v for k, v in info.items() if isinstance(v, (int, float, bool, str))}
                )

                # Capture images
                if save_images and has_cameras and hasattr(env, "get_observation"):
                    if step_idx % 100 == 0:
                        typer.echo(f"    Step {step_idx}...")
                    full_obs = env.get_observation()
                    for cam_name, img in full_obs.camera_views.items():
                        if cam_name not in images:
                            images[cam_name] = []
                        images[cam_name].append(img.copy())

            if done:
                break

        success = env.get_success()
        successes += int(success)
        total_reward += ep_reward

        status = "SUCCESS" if success else "FAIL"
        typer.echo(f"  Episode {ep + 1}: {status} | Reward: {ep_reward:.2f} | Steps: {steps}")

        # Save episode data
        if recording:
            ep_dir = record_path / f"episode_{ep:03d}"
            ep_dir.mkdir(exist_ok=True)

            # Save task config
            with open(ep_dir / "task_config.json", "w") as f:
                json.dump(task_config.to_dict(), f, indent=2)

            # Save episode result
            result = {
                "success": success,
                "total_reward": ep_reward,
                "timesteps": steps,
            }
            with open(ep_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)

            # Save trajectory as numpy archive
            np.savez_compressed(
                ep_dir / "trajectory.npz",
                observations=np.array(observations),
                actions=np.array(actions),
                rewards=np.array(rewards),
                dones=np.array(dones),
            )

            # Save images
            if save_images and images:
                from PIL import Image

                images_dir = ep_dir / "images"
                images_dir.mkdir(exist_ok=True)

                for cam_name, cam_images in images.items():
                    for i, img in enumerate(cam_images):
                        img_path = images_dir / f"{cam_name}_{i:04d}.png"
                        Image.fromarray(img).save(img_path)

                typer.echo(f"    Saved {sum(len(v) for v in images.values())} images")

    env.close()

    typer.echo(f"\nResults: {successes}/{episodes} successful")
    typer.echo(f"Average reward: {total_reward / episodes:.2f}")

    if recording:
        typer.echo(f"\nRecordings saved to: {record_path}")
