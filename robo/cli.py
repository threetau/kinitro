"""Command-line interface for the robotics subnet."""

import asyncio
from typing import Optional

import typer

app = typer.Typer(
    name="robo",
    help="Robotics Generalization Subnet CLI",
    add_completion=False,
)


# =============================================================================
# VALIDATOR COMMANDS
# =============================================================================


@app.command()
def validate(
    network: str = typer.Option("finney", help="Network: finney, test, or local"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
    episodes_per_env: int = typer.Option(50, help="Episodes per environment"),
    eval_interval: int = typer.Option(3600, help="Seconds between evaluation cycles"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the validator.

    Evaluates miner policies across robotics environments and sets weights.
    """
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    from robo.config import ValidatorConfig
    from robo.validator.main import run_validator

    config = ValidatorConfig(
        network=network,
        netuid=netuid,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        episodes_per_env=episodes_per_env,
        eval_interval_seconds=eval_interval,
        log_level=log_level,
    )

    typer.echo(f"Starting validator on {network} (netuid={netuid})")
    asyncio.run(run_validator(config))


@app.command()
def list_envs():
    """List all available robotics environments."""
    from robo.environments.registry import (
        get_all_environment_ids,
        get_available_families,
        get_environments_by_family,
        is_family_available,
    )

    typer.echo("Available Robotics Environments:\n")

    all_families = ["metaworld", "dm_control", "maniskill"]
    available_families = get_available_families()

    for family in all_families:
        if is_family_available(family):
            envs = get_environments_by_family(family)
            if envs:
                typer.echo(f"  {family.upper()} (installed):")
                for env_id in envs:
                    typer.echo(f"    - {env_id}")
                typer.echo()
        else:
            typer.echo(f"  {family.upper()} (not installed):")
            if family == "dm_control":
                typer.echo("    Install with: pip install robo-subnet[dm-control]")
            elif family == "maniskill":
                typer.echo("    Install with: pip install robo-subnet[maniskill]")
            typer.echo()

    total = len(get_all_environment_ids())
    typer.echo(f"Total: {total} environments available")
    typer.echo(f"Families: {', '.join(available_families)}")


# =============================================================================
# MINER COMMANDS
# =============================================================================


@app.command()
def commit(
    repo: str = typer.Option(..., help="HuggingFace repo (user/model)"),
    revision: str = typer.Option(..., help="Commit SHA"),
    chute_id: str = typer.Option(..., help="Chutes deployment ID"),
    network: str = typer.Option("finney", help="Network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
):
    """
    Commit model to chain (for miners).

    Registers your policy so validators can evaluate it.
    """
    import bittensor as bt

    from robo.chain.commitments import commit_model

    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

    typer.echo(f"Committing model to {network} (netuid={netuid})")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision}")
    typer.echo(f"  Chute ID: {chute_id}")

    success = commit_model(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        repo=repo,
        revision=revision,
        chute_id=chute_id,
    )

    if success:
        typer.echo("Commitment successful!")
    else:
        typer.echo("Commitment failed!", err=True)
        raise typer.Exit(1)


@app.command()
def build(
    env_path: str = typer.Argument(..., help="Path to env directory"),
    tag: str = typer.Option(..., help="Docker tag (e.g., user/repo:v1)"),
    push: bool = typer.Option(False, help="Push to registry after building"),
):
    """
    Build miner Docker image.

    Builds a Docker image from your policy directory.
    """
    import subprocess

    typer.echo(f"Building Docker image: {tag}")

    # Build
    result = subprocess.run(
        ["docker", "build", "-t", tag, env_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        typer.echo(f"Build failed:\n{result.stderr}", err=True)
        raise typer.Exit(1)

    typer.echo("Build successful!")

    # Push if requested
    if push:
        typer.echo(f"Pushing to registry: {tag}")
        result = subprocess.run(
            ["docker", "push", tag],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            typer.echo(f"Push failed:\n{result.stderr}", err=True)
            raise typer.Exit(1)

        typer.echo("Push successful!")


@app.command()
def init_miner(
    output_dir: str = typer.Argument(".", help="Directory to create template in"),
):
    """
    Initialize a new miner policy from template.

    Creates the necessary files for building a policy container.
    """
    import shutil
    from pathlib import Path

    template_dir = Path(__file__).parent / "miner" / "template"
    output_path = Path(output_dir)

    if not template_dir.exists():
        typer.echo("Template directory not found!", err=True)
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy template files
    for file in template_dir.iterdir():
        dest = output_path / file.name
        if dest.exists():
            typer.echo(f"Skipping {file.name} (already exists)")
        else:
            shutil.copy(file, dest)
            typer.echo(f"Created {file.name}")

    typer.echo("\nMiner template initialized!")
    typer.echo("Next steps:")
    typer.echo("  1. Edit env.py to implement your policy")
    typer.echo("  2. Add your model weights")
    typer.echo("  3. Run: robo build . --tag your-user/robo-policy:v1 --push")
    typer.echo("  4. Run: robo commit --repo ... --revision ... --chute-id ...")


# =============================================================================
# TESTING COMMANDS
# =============================================================================


@app.command()
def test_env(
    env_id: str = typer.Argument(..., help="Environment ID to test"),
    episodes: int = typer.Option(5, help="Number of episodes to run"),
    render: bool = typer.Option(False, help="Render environment (if supported)"),
):
    """
    Test an environment with random actions.

    Useful for verifying environment setup.
    """
    import numpy as np

    from robo.environments import get_environment

    typer.echo(f"Testing environment: {env_id}")

    env = get_environment(env_id)
    typer.echo(f"  Proprioceptive observation shape: {env.observation_shape}")
    typer.echo(f"  Action shape: {env.action_shape}")

    # Check for camera support
    if hasattr(env, "num_cameras"):
        typer.echo(f"  Number of cameras: {env.num_cameras}")
        if hasattr(env, "image_shape"):
            typer.echo(f"  Image shape: {env.image_shape}")

    successes = 0
    total_reward = 0.0

    for ep in range(episodes):
        task_config = env.generate_task(seed=ep)
        obs = env.reset(task_config)

        typer.echo(f"  Episode {ep + 1} initial obs: {obs}")

        ep_reward = 0.0
        steps = 0

        for _ in range(500):
            # Random action
            low, high = env.action_bounds
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1

            if done:
                break

        success = env.get_success()
        successes += int(success)
        total_reward += ep_reward

        status = "SUCCESS" if success else "FAIL"
        typer.echo(f"  Episode {ep + 1}: {status} | Reward: {ep_reward:.2f} | Steps: {steps}")

    env.close()

    typer.echo(f"\nResults: {successes}/{episodes} successful")
    typer.echo(f"Average reward: {total_reward / episodes:.2f}")


@app.command()
def test_scoring(
    n_miners: int = typer.Option(5, help="Number of simulated miners"),
    n_envs: int = typer.Option(3, help="Number of environments"),
):
    """
    Test the Pareto scoring mechanism with simulated data.
    """
    import numpy as np

    from robo.scoring.pareto import compute_pareto_frontier
    from robo.scoring.winners_take_all import compute_full_scoring

    typer.echo(f"Testing Pareto scoring with {n_miners} miners, {n_envs} environments\n")

    # Generate random scores
    env_ids = [f"env_{i}" for i in range(n_envs)]
    miner_scores = {}

    for uid in range(n_miners):
        miner_scores[uid] = {env_id: float(np.random.uniform(0.3, 0.9)) for env_id in env_ids}

    # Display scores
    typer.echo("Miner scores:")
    for uid, scores in miner_scores.items():
        scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())
        typer.echo(f"  UID {uid}: {scores_str}")

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(miner_scores, env_ids, n_samples_per_env=50)
    typer.echo(f"\nPareto frontier: {pareto.frontier_uids}")

    # Compute weights
    weights = compute_full_scoring(miner_scores, env_ids)
    typer.echo("\nFinal weights:")
    for uid, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"  UID {uid}: {weight:.4f}")


if __name__ == "__main__":
    app()
