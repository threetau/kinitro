"""Environment commands for building, listing, and testing environments."""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

import affinetes
import numpy as np
import typer
from PIL import Image

from kinitro.environments import get_environment
from kinitro.environments.registry import (
    get_all_environment_ids,
    get_available_families,
    get_environments_by_family,
    get_family_metadata,
)
from kinitro.rl_interface import CanonicalObservation, coerce_action

# Available environment families for build command
AVAILABLE_ENV_FAMILIES = ["metaworld", "procthor"]


@runtime_checkable
class CameraCapable(Protocol):
    num_cameras: int
    image_shape: tuple[int, ...]

    def get_observation(self) -> CanonicalObservation: ...


def build_env(
    family: str = typer.Argument(
        ...,
        help=f"Environment family to build: {', '.join(AVAILABLE_ENV_FAMILIES)}",
    ),
    tag: str | None = typer.Option(
        None,
        help="Docker tag (default: kinitro/<family>:v1)",
    ),
    push: bool = typer.Option(False, help="Push to registry after building"),
    registry: str | None = typer.Option(
        None, help="Registry URL for pushing (e.g., docker.io/myuser)"
    ),
    no_cache: bool = typer.Option(False, help="Build without using cache"),
    quiet: bool = typer.Option(False, help="Suppress build output"),
):
    """
    Build an environment-specific Docker image using affinetes.

    This creates a focused Docker image containing only the dependencies
    for a specific environment family.

    Environment families:
      - metaworld: MuJoCo-based manipulation tasks (~400MB image)
      - procthor: AI2-THOR procedural house tasks (~1.5GB image, x86_64 Linux only)

    Examples:
        # Build MetaWorld environment
        kinitro env build metaworld --tag kinitro/metaworld:v1

        # Build ProcTHOR environment
        kinitro env build procthor --tag kinitro/procthor:v1

        # Build and push to registry
        kinitro env build metaworld --push --registry docker.io/myuser
    """
    # Validate family
    if family not in AVAILABLE_ENV_FAMILIES:
        typer.echo(
            f"Unknown environment family: {family}. Available: {', '.join(AVAILABLE_ENV_FAMILIES)}",
            err=True,
        )
        raise typer.Exit(1)

    # Default tag if not provided
    if tag is None:
        tag = f"kinitro/{family}:v1"

    # Find the environment directory and kinitro package
    kinitro_package_dir = Path(__file__).parent.parent.parent
    root_dir = kinitro_package_dir.parent
    env_path = root_dir / "environments" / family
    environments_src = kinitro_package_dir / "environments"

    if not env_path.exists():
        typer.echo(f"Environment directory not found at {env_path}", err=True)
        raise typer.Exit(1)

    if not (env_path / "env.py").exists():
        typer.echo(f"env.py not found at {env_path}", err=True)
        raise typer.Exit(1)

    if not (env_path / "Dockerfile").exists():
        typer.echo(f"Dockerfile not found at {env_path}", err=True)
        raise typer.Exit(1)

    if not environments_src.exists():
        typer.echo(f"environments module not found at {environments_src}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Building {family} environment image: {tag}")
    typer.echo(f"  Environment path: {env_path}")
    if push:
        typer.echo(f"  Push: True (registry: {registry or 'from tag'})")

    # Copy kinitro/environments to env/kinitro/environments for the build
    env_kinitro = env_path / "kinitro"
    env_environments = env_kinitro / "environments"
    created_kinitro_dir = False

    if env_kinitro.exists():
        typer.echo(f"Refusing to overwrite existing {env_kinitro}", err=True)
        raise typer.Exit(1)
    env_kinitro.mkdir()
    created_kinitro_dir = True

    # Also need rl_interface.py for the Actor
    rl_interface_src = kinitro_package_dir / "rl_interface.py"

    try:
        # Create kinitro package structure
        env_kinitro.mkdir(exist_ok=True)

        # Create __init__.py for kinitro package
        (env_kinitro / "__init__.py").write_text(
            '"""Kinitro package subset for eval environment."""\n'
        )

        # Copy environments module (filtered based on family)
        if env_environments.exists():
            shutil.rmtree(env_environments)

        # Copy the full environments module
        shutil.copytree(
            environments_src,
            env_environments,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        typer.echo("  Copied environments module to build context")

        # Copy rl_interface.py
        if rl_interface_src.exists():
            shutil.copy(rl_interface_src, env_kinitro / "rl_interface.py")
            typer.echo("  Copied rl_interface module to build context")

        # Build the image
        result_tag = affinetes.build_image_from_env(
            env_path=str(env_path),
            image_tag=tag,
            nocache=no_cache,
            quiet=quiet,
            push=push,
            registry=registry,
        )
        typer.echo(f"\nBuild successful: {result_tag}")

        if push:
            typer.echo(f"Pushed to: {result_tag}")

    except Exception as e:
        typer.echo(f"Build failed: {e}", err=True)
        raise typer.Exit(1)

    finally:
        # Clean up the copied kinitro directory
        if created_kinitro_dir and env_kinitro.exists():
            shutil.rmtree(env_kinitro)
            typer.echo("  Cleaned up temporary build files")

    typer.echo(f"\nTo use this image in the backend for {family}/* environments:")
    typer.echo("  eval_images:")
    typer.echo(f"    {family}: {result_tag}")


def list_envs():
    """List all available robotics environments."""
    all_envs = get_all_environment_ids()

    typer.echo("Available Robotics Environments:\n")

    for family in get_available_families():
        envs = get_environments_by_family(family)
        if envs:
            metadata = get_family_metadata(family)
            label = (
                f"{metadata['name']} ({metadata['description']})" if metadata else family.upper()
            )
            typer.echo(f"  {label}:")
            for env_id in envs:
                typer.echo(f"    - {env_id}")
            typer.echo()

    typer.echo(f"Total: {len(all_envs)} environments available")


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
        kinitro env test metaworld/pick-place-v3

        # Record trajectories to disk
        kinitro env test metaworld/push-v3 --record-dir ./recordings

        # Record with camera images
        kinitro env test metaworld/push-v3 --record-dir ./recordings --save-images
    """
    typer.echo(f"Testing environment: {env_id}")
    if episodes < 1:
        typer.echo("Error: --episodes must be >= 1", err=True)
        raise typer.Exit(1)

    env = get_environment(env_id)
    try:
        typer.echo(f"  Canonical observation shape: {env.observation_shape}")
        typer.echo(f"  Canonical action shape: {env.action_shape}")

        # Check for camera support
        camera_env = env if isinstance(env, CameraCapable) else None
        has_cameras = camera_env is not None and camera_env.num_cameras > 0
        if has_cameras and camera_env is not None:
            typer.echo(f"  Number of cameras: {camera_env.num_cameras}")
            typer.echo(f"  Image shape: {camera_env.image_shape}")

        # Setup recording directory
        recording = record_dir is not None
        record_path: Path | None = None
        if record_dir is not None:
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
            if has_cameras and camera_env is not None:
                metadata["image_shape"] = list(camera_env.image_shape)
                metadata["num_cameras"] = camera_env.num_cameras
            with open(record_path / "metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)
            typer.echo(f"  Recording to: {record_path}")

        if recording and record_path is None:
            typer.echo("Error: record path was not initialized", err=True)
            raise typer.Exit(1)

        successes = 0
        total_reward = 0.0
        for ep in range(episodes):
            task_config = env.generate_task(seed=ep)
            obs = env.reset(task_config)
            typer.echo(f"  Episode {ep + 1} initial obs: {obs}")

            # Storage for trajectory
            observations = []
            if recording:
                observations.append(obs.to_payload(include_images=save_images))
            actions = []
            rewards = []
            dones = []
            infos = []
            images: dict[str, list[np.ndarray]] = {}

            # Capture initial images if recording
            if recording and save_images and has_cameras and camera_env is not None:
                typer.echo("    Capturing initial images...")
                full_obs = camera_env.get_observation()
                for cam_name, img in full_obs.rgb.items():
                    if cam_name not in images:
                        images[cam_name] = []
                    images[cam_name].append(np.array(img))
                if images:
                    typer.echo("    Done.")
                else:
                    typer.secho(
                        "    Warning: No images captured. Camera rendering may have failed.",
                        fg="yellow",
                    )
                    typer.secho(
                        "    Hint: Try running with MUJOCO_GL=egl or MUJOCO_GL=osmesa",
                        fg="yellow",
                    )

            ep_reward = 0.0
            steps = 0
            for step_idx in range(max_steps):
                # Random action
                low, high = env.action_bounds
                action_array = np.random.uniform(low, high)
                action = coerce_action(action_array)
                obs, reward, done, info = env.step(action)
                ep_reward += reward
                steps += 1

                # Record trajectory data
                if recording:
                    observations.append(obs.to_payload(include_images=save_images))
                    actions.append(action_array)
                    rewards.append(reward)
                    dones.append(done)

                    # Convert info to serializable format
                    infos.append(
                        {k: v for k, v in info.items() if isinstance(v, (int, float, bool, str))}
                    )

                    # Capture images
                    if save_images and has_cameras and camera_env is not None:
                        if step_idx % 100 == 0:
                            typer.echo(f"    Step {step_idx}...")
                        full_obs = camera_env.get_observation()
                        for cam_name, img in full_obs.rgb.items():
                            if cam_name not in images:
                                images[cam_name] = []
                            images[cam_name].append(np.array(img))
                if done:
                    break

            success = env.get_success()
            successes += int(success)
            total_reward += ep_reward
            status = "SUCCESS" if success else "FAIL"
            typer.echo(f"  Episode {ep + 1}: {status} | Reward: {ep_reward:.2f} | Steps: {steps}")

            # Save episode data
            if recording:
                if record_path is None:
                    raise typer.Exit(1)
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
                    observations=np.array(observations, dtype=object),
                    actions=np.array(actions, dtype=object),
                    rewards=np.array(rewards),
                    dones=np.array(dones),
                )

                # Save images
                if save_images and images:
                    images_dir = ep_dir / "images"
                    images_dir.mkdir(exist_ok=True)

                    for cam_name, cam_images in images.items():
                        for i, img in enumerate(cam_images):
                            img_path = images_dir / f"{cam_name}_{i:04d}.png"
                            Image.fromarray(img.astype(np.uint8)).save(img_path)
                    typer.echo(f"    Saved {sum(len(v) for v in images.values())} images")
                elif save_images and not images:
                    typer.secho(
                        "    Warning: --save-images was set but no images were captured. "
                        "Camera rendering may have failed (try MUJOCO_GL=egl or osmesa).",
                        fg="yellow",
                    )

        typer.echo(f"\nResults: {successes}/{episodes} successful")
        typer.echo(f"Average reward: {total_reward / episodes:.2f}")
    finally:
        env.close()

    if recording and record_path is not None:
        typer.echo(f"\nRecordings saved to: {record_path}")
