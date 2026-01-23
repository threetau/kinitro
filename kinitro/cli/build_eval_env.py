"""Build evaluation environment Docker image command."""

import typer


def build_eval_env(
    tag: str = typer.Option(
        "kinitro/eval-env:v1",
        help="Docker tag for eval environment image",
    ),
    push: bool = typer.Option(False, help="Push to registry after building"),
    registry: str | None = typer.Option(
        None, help="Registry URL for pushing (e.g., docker.io/myuser)"
    ),
    no_cache: bool = typer.Option(False, help="Build without using cache"),
    quiet: bool = typer.Option(False, help="Suppress build output"),
):
    """
    Build the evaluation environment Docker image using affinetes.

    This image is used by affinetes to run evaluations. It contains:
    - MuJoCo + MetaWorld simulation environment
    - HTTP client for calling miner policy endpoints
    - The kinitro environments module

    The built image is used by the backend scheduler when running evaluations.

    Examples:
        # Build locally
        kinitro build-eval-env --tag kinitro/eval-env:v1

        # Build and push to Docker Hub
        kinitro build-eval-env --tag eval-env:v1 --push --registry docker.io/myuser
    """
    import shutil
    from pathlib import Path

    import affinetes

    # Find the eval-env directory and kinitro package
    kinitro_package_dir = Path(__file__).parent.parent
    root_dir = kinitro_package_dir.parent
    eval_env_path = root_dir / "eval-env"
    environments_src = kinitro_package_dir / "environments"

    if not (eval_env_path / "env.py").exists():
        typer.echo(f"env.py not found at {eval_env_path}", err=True)
        typer.echo("Make sure you're running from within the kinitro package.")
        raise typer.Exit(1)

    if not (eval_env_path / "Dockerfile").exists():
        typer.echo(f"Dockerfile not found at {eval_env_path}", err=True)
        raise typer.Exit(1)

    if not environments_src.exists():
        typer.echo(f"environments module not found at {environments_src}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Building eval environment image: {tag}")
    typer.echo(f"  Environment path: {eval_env_path}")
    if push:
        typer.echo(f"  Push: True (registry: {registry or 'from tag'})")

    # Copy kinitro/environments to eval-env/kinitro/environments for the build
    # This avoids duplicating the code in the repo
    eval_env_kinitro = eval_env_path / "kinitro"
    eval_env_environments = eval_env_kinitro / "environments"

    try:
        # Create kinitro package structure in eval-env
        eval_env_kinitro.mkdir(exist_ok=True)

        # Create __init__.py for kinitro package
        (eval_env_kinitro / "__init__.py").write_text(
            '"""Kinitro package subset for eval environment."""\n'
        )

        # Copy environments module
        if eval_env_environments.exists():
            shutil.rmtree(eval_env_environments)
        shutil.copytree(
            environments_src,
            eval_env_environments,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        typer.echo("  Copied environments module to build context")

        # Build the image
        result_tag = affinetes.build_image_from_env(
            env_path=str(eval_env_path),
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
        if eval_env_kinitro.exists():
            shutil.rmtree(eval_env_kinitro)
            typer.echo("  Cleaned up temporary build files")

    typer.echo("\nTo use this image in the backend, ensure your config has:")
    typer.echo(f"  eval_image: {result_tag}")
