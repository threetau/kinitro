"""Deploy policy to Chutes command."""

import typer


def chutes_push(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    revision: str = typer.Option(..., "--revision", help="HuggingFace commit SHA"),
    chutes_api_key: str | None = typer.Option(
        None, "--api-key", envvar="CHUTES_API_KEY", help="Chutes API key"
    ),
    chute_user: str | None = typer.Option(
        None, "--user", envvar="CHUTE_USER", help="Chutes username"
    ),
    gpu_count: int = typer.Option(1, "--gpu-count", help="Number of GPUs"),
    min_vram_gb: int = typer.Option(8, "--min-vram", help="Minimum VRAM in GB"),
):
    """
    Deploy policy to Chutes.

    Generates a Chute configuration and deploys your policy server.
    Requires CHUTES_API_KEY and CHUTE_USER environment variables.

    Example:
        kinitro chutes-push --repo user/robo-policy --revision abc123
    """
    import os
    import subprocess
    import textwrap
    from pathlib import Path

    # Validate required credentials
    api_key = chutes_api_key or os.environ.get("CHUTES_API_KEY")
    user = chute_user or os.environ.get("CHUTE_USER")

    if not api_key:
        typer.echo("Error: CHUTES_API_KEY not configured", err=True)
        typer.echo("Set it via --api-key or CHUTES_API_KEY environment variable")
        raise typer.Exit(1)

    if not user:
        typer.echo("Error: CHUTE_USER not configured", err=True)
        typer.echo("Set it via --user or CHUTE_USER environment variable")
        raise typer.Exit(1)

    typer.echo("Deploying to Chutes:")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision[:12]}...")
    typer.echo(f"  User: {user}")
    typer.echo(f"  GPU: {gpu_count}x (min {min_vram_gb}GB VRAM)")

    # Generate Chute configuration for robotics policy server
    chute_config = textwrap.dedent(f'''
import os
from chutes import Chute
from chutes.chute import NodeSelector

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = Chute(
    name="{repo.replace("/", "-")}",
    readme="{repo}",
    node_selector=NodeSelector(
        gpu_count={gpu_count},
        min_vram_gb={min_vram_gb},
    ),
)

# The policy server files are loaded from HuggingFace
chute.from_huggingface("{repo}", revision="{revision}")

# Install dependencies
chute.add_pip_requirements("requirements.txt")

# Set the entrypoint for the FastAPI policy server
chute.entrypoint = "uvicorn server:app --host 0.0.0.0 --port 8000"
''')

    tmp_file = Path("tmp_kinitro_chute.py")
    tmp_file.write_text(chute_config)

    try:
        # Deploy to Chutes
        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
        env = {**os.environ, "CHUTES_API_KEY": api_key}

        typer.echo("\nDeploying...")
        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            typer.echo("Chutes deployment failed!", err=True)
            raise typer.Exit(1)

        typer.echo("\nDeployment submitted!")
        typer.echo("Check the Chutes dashboard for your chute_id")
        typer.echo("\nNext step:")
        typer.echo(
            f"  kinitro commit --repo {repo} --revision {revision} --chute-id YOUR_CHUTE_ID --netuid ..."
        )

    finally:
        tmp_file.unlink(missing_ok=True)
