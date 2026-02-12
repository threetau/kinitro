"""Deployment commands for Basilica and one-command deployment."""

import asyncio
import os

import typer
from basilica import BasilicaClient
from bittensor import AsyncSubtensor
from bittensor_wallet import Wallet
from huggingface_hub import HfApi

from kinitro.chain.commitments import commit_model_async

# Shared deployment configuration
PIP_PACKAGES = [
    "fastapi",
    "uvicorn",
    "numpy",
    "huggingface-hub",
    "pydantic",
    "pillow",
]


def _get_deployment_source(repo: str, revision: str) -> str:
    """Generate the deployment source code template."""
    return f"""
import os
import sys
import subprocess

print("Starting Kinitro Policy Server...")
print(f"HF_REPO: {{os.environ.get('HF_REPO', 'not set')}}")
print(f"HF_REVISION: {{os.environ.get('HF_REVISION', 'not set')}}")

# Download model from HuggingFace
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN") or None
print("Downloading model from HuggingFace...")
snapshot_download(
    "{repo}",
    revision="{revision}",
    local_dir="/app",
    token=hf_token,
)
print("Model downloaded successfully!")

# Change to /app directory and add to Python path
os.chdir("/app")
sys.path.insert(0, "/app")

# Start the FastAPI server from /app directory
print("Starting uvicorn server on port 8000...")
subprocess.run(
    [
        sys.executable, "-m", "uvicorn",
        "server:app",
        "--host", "0.0.0.0",
        "--port", "8000",
    ],
    cwd="/app",
    check=True,
)
"""


def _get_docker_image(gpu_count: int) -> str:
    """Choose Docker image based on GPU requirement."""
    if gpu_count > 0:
        return "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    return "python:3.11-slim"


def _extract_deployment_id(deployment) -> str:
    """Extract deployment ID from deployment URL or fall back to name."""
    if deployment.url and (
        deployment.url.startswith("http://") or deployment.url.startswith("https://")
    ):
        try:
            url_parts = deployment.url.split("//")[1].split(".")
            if len(url_parts) > 0:
                return url_parts[0]
        except (IndexError, AttributeError):
            pass
    return deployment.name


def basilica_push(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    revision: str = typer.Option(..., "--revision", help="HuggingFace commit SHA"),
    deployment_name: str | None = typer.Option(
        None, "--name", "-n", help="Deployment name (default: derived from repo)"
    ),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Number of GPUs (0 for CPU-only)"),
    min_gpu_memory_gb: int | None = typer.Option(None, "--min-vram", help="Minimum GPU VRAM in GB"),
    cpu: str = typer.Option("1", "--cpu", help="CPU allocation (e.g., '1', '2', '500m', '4000m')"),
    memory: str = typer.Option(
        "512Mi", "--memory", help="Memory allocation (e.g., '512Mi', '16Gi')"
    ),
    basilica_api_token: str | None = typer.Option(
        None, "--api-token", envvar="BASILICA_API_TOKEN", help="Basilica API token"
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace token for private repos"
    ),
    timeout: int = typer.Option(600, "--timeout", help="Deployment timeout in seconds"),
):
    """
    Deploy policy to Basilica.

    Deploys your robotics policy server to Basilica's GPU serverless platform.
    The policy is downloaded from HuggingFace and served via FastAPI.

    Requires BASILICA_API_TOKEN environment variable or --api-token.

    Example:
        kinitro miner push --repo user/policy --revision abc123

        # With custom name
        kinitro miner push --repo user/policy --revision abc123 --name my-policy

        # With more GPU memory
        kinitro miner push --repo user/policy --revision abc123 --min-vram 24
    """
    # Validate required credentials
    api_token = basilica_api_token or os.environ.get("BASILICA_API_TOKEN")

    if not api_token:
        typer.echo("Error: BASILICA_API_TOKEN not configured", err=True)
        typer.echo("Set it via --api-token or BASILICA_API_TOKEN environment variable")
        typer.echo("\nTo get a token, run: basilica tokens create")
        raise typer.Exit(1)

    # Derive deployment name from repo
    name = deployment_name or repo.replace("/", "-").lower()
    # Ensure name is DNS-safe (lowercase, alphanumeric and hyphens only)
    name = "".join(c if c.isalnum() or c == "-" else "-" for c in name).strip("-")[:63]

    typer.echo("Deploying to Basilica:")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision[:12]}...")
    typer.echo(f"  Deployment Name: {name}")
    vram_str = f" (min {min_gpu_memory_gb}GB VRAM)" if min_gpu_memory_gb else ""
    typer.echo(f"  GPU: {gpu_count}x{vram_str}")
    typer.echo(f"  Memory: {memory}")

    # Create client and build deployment configuration
    client = BasilicaClient(api_key=api_token)
    source_code = _get_deployment_source(repo, revision)

    # Note: kinitro is NOT included in pip_packages because the miner template
    # is self-contained (includes rl_interface.py locally). This avoids
    # dependency conflicts and keeps the deployment lightweight.
    env_vars: dict[str, str] = {"HF_REPO": repo, "HF_REVISION": revision}
    if hf_token:
        env_vars["HF_TOKEN"] = hf_token

    deploy_kwargs: dict = {
        "name": name,
        "source": source_code,
        "image": _get_docker_image(gpu_count),
        "port": 8000,
        "env": env_vars,
        "cpu": cpu,
        "memory": memory,
        "pip_packages": PIP_PACKAGES,
        "timeout": timeout,
    }

    if gpu_count > 0:
        deploy_kwargs["gpu_count"] = gpu_count
        if min_gpu_memory_gb is not None:
            deploy_kwargs["min_gpu_memory_gb"] = min_gpu_memory_gb

    # Deploy
    typer.echo("\nDeploying to Basilica (this may take several minutes)...")
    try:
        deployment = client.deploy(**deploy_kwargs)
    except Exception as e:
        typer.echo(f"\nDeployment failed: {e}", err=True)
        raise typer.Exit(1)

    deploy_id = _extract_deployment_id(deployment)

    typer.echo("\n" + "=" * 60)
    typer.echo("DEPLOYMENT SUCCESSFUL")
    typer.echo("=" * 60)
    typer.echo(f"  Name: {deployment.name}")
    typer.echo(f"  URL: {deployment.url}")
    typer.echo(f"  State: {deployment.state}")
    typer.echo("=" * 60)
    typer.echo("\nNext step - commit on-chain:")
    typer.echo(f"  kinitro miner commit --repo {repo} --revision {revision} \\")
    typer.echo(f"    --deployment-id {deploy_id} --netuid YOUR_NETUID")


def miner_deploy(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    policy_path: str | None = typer.Option(
        None, "--path", "-p", help="Path to local policy directory"
    ),
    revision: str | None = typer.Option(
        None, "--revision", help="HuggingFace commit SHA (required if --skip-upload)"
    ),
    deployment_id: str | None = typer.Option(
        None, "--deployment-id", "-d", help="Basilica deployment ID (required if --skip-deploy)"
    ),
    deployment_name: str | None = typer.Option(
        None, "--name", "-n", help="Deployment name (default: derived from repo)"
    ),
    message: str = typer.Option(
        "Model update", "--message", "-m", help="Commit message for HuggingFace upload"
    ),
    skip_upload: bool = typer.Option(False, "--skip-upload", help="Skip HuggingFace upload"),
    skip_deploy: bool = typer.Option(False, "--skip-deploy", help="Skip Basilica deployment"),
    skip_commit: bool = typer.Option(False, "--skip-commit", help="Skip on-chain commit"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without executing"
    ),
    network: str = typer.Option("finney", "--network", help="Bittensor network"),
    netuid: int | None = typer.Option(
        None, "--netuid", help="Subnet UID (required unless --skip-commit)"
    ),
    wallet_name: str = typer.Option("default", "--wallet-name", help="Wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey-name", help="Hotkey name"),
    basilica_api_token: str | None = typer.Option(
        None, "--api-token", envvar="BASILICA_API_TOKEN", help="Basilica API token"
    ),
    hf_token: str | None = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Number of GPUs (0 for CPU-only)"),
    min_gpu_memory_gb: int | None = typer.Option(None, "--min-vram", help="Minimum GPU VRAM in GB"),
    cpu: str = typer.Option("1", "--cpu", help="CPU allocation (e.g., '1', '2', '500m', '4000m')"),
    memory: str = typer.Option(
        "512Mi", "--memory", help="Memory allocation (e.g., '512Mi', '16Gi')"
    ),
    timeout: int = typer.Option(600, "--timeout", help="Deployment timeout in seconds"),
):
    """
    One-command deployment: Upload -> Deploy -> Commit.

    Combines the miner deployment workflow into a single command:
    1. Upload policy to HuggingFace (skip with --skip-upload)
    2. Deploy to Basilica (skip with --skip-deploy)
    3. Commit on-chain (skip with --skip-commit)

    Examples:
        # Full deployment
        kinitro miner deploy -r user/policy -p ./my-policy --netuid 123

        # Skip upload (already on HuggingFace)
        kinitro miner deploy -r user/policy --skip-upload --revision abc123 --netuid 123

        # Skip deployment (already deployed)
        kinitro miner deploy -r user/policy --skip-upload --revision abc123 \
            --skip-deploy --deployment-id https://my-policy.basilica.ai --netuid 123

        # Dry run to see what would happen
        kinitro miner deploy -r user/policy -p ./my-policy --netuid 123 --dry-run
    """
    # Validate arguments
    if not skip_upload and not policy_path:
        typer.echo("Error: --path is required unless --skip-upload is set", err=True)
        raise typer.Exit(1)

    if skip_upload and not revision:
        typer.echo("Error: --revision is required when --skip-upload is set", err=True)
        raise typer.Exit(1)

    if skip_deploy and not deployment_id:
        typer.echo("Error: --deployment-id is required when --skip-deploy is set", err=True)
        raise typer.Exit(1)

    if not skip_commit and netuid is None:
        typer.echo("Error: --netuid is required unless --skip-commit is set", err=True)
        raise typer.Exit(1)

    # Get credentials from env if not provided
    api_token = basilica_api_token or os.environ.get("BASILICA_API_TOKEN")
    hf = hf_token or os.environ.get("HF_TOKEN")

    # Validate credentials
    if not dry_run:
        if not skip_upload and not hf:
            typer.echo("Error: HF_TOKEN not configured", err=True)
            raise typer.Exit(1)
        if not skip_deploy and not api_token:
            typer.echo("Error: BASILICA_API_TOKEN not configured", err=True)
            typer.echo("Set it via --api-token or BASILICA_API_TOKEN environment variable")
            typer.echo("\nTo get a token, run: basilica tokens create")
            raise typer.Exit(1)

    revision_value = revision

    # Determine steps
    steps = []
    if not skip_upload:
        steps.append("upload")
    if not skip_deploy:
        steps.append("deploy")
    if not skip_commit:
        steps.append("commit")

    typer.echo("=" * 60)
    typer.echo("KINITRO DEPLOYMENT")
    typer.echo("=" * 60)
    typer.echo(f"  Repository: {repo}")
    if policy_path:
        typer.echo(f"  Policy Path: {policy_path}")
    if revision_value:
        typer.echo(f"  Revision: {revision_value}")
    if deployment_id:
        typer.echo(f"  Deployment ID: {deployment_id}")
    typer.echo(f"  Steps: {' -> '.join(steps) if steps else 'none'}")
    if dry_run:
        typer.echo("  Mode: DRY RUN")
    typer.echo("=" * 60)

    # Maximum allowed repo size (same as verification limit, configurable via env var)
    max_repo_size_gb = float(os.environ.get("KINITRO_MAX_REPO_SIZE_GB", "5.0"))
    max_repo_size_bytes = int(max_repo_size_gb * 1024 * 1024 * 1024)

    # Step 1: Upload to HuggingFace
    if not skip_upload:
        typer.echo(f"\n[1/{len(steps)}] Uploading to HuggingFace ({repo})...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would upload {policy_path} to {repo}")
            revision_value = "dry-run-revision"
        else:
            try:
                if policy_path is None:
                    raise typer.Exit(1)
                policy_path_value = policy_path

                # Check local folder size before uploading
                total_size = 0
                for dirpath, dirnames, filenames in os.walk(policy_path_value):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.isfile(filepath):
                            total_size += os.path.getsize(filepath)

                if total_size > max_repo_size_bytes:
                    typer.echo(
                        f"\nError: Folder size {total_size / (1024 * 1024):.2f}MB exceeds limit of {max_repo_size_gb}GB",
                        err=True,
                    )
                    typer.echo("Please reduce your policy folder size to stay within the limit.")
                    raise typer.Exit(1)

                typer.echo(
                    f"  Folder size: {total_size / (1024 * 1024):.2f}MB (max: {max_repo_size_gb}GB)"
                )

                api = HfApi(token=hf)

                # Create repo if it doesn't exist
                try:
                    api.create_repo(repo_id=repo, exist_ok=True, private=False)
                    typer.echo(f"  Repository ready: {repo}")
                except Exception:
                    typer.echo(f"  Repository already exists or created: {repo}")

                # Upload folder
                typer.echo(f"  Uploading from {policy_path_value}...")
                result = api.upload_folder(
                    folder_path=policy_path_value,
                    repo_id=repo,
                    commit_message=message,
                )
                revision_value = result.commit_url.split("/")[-1]
                typer.echo("  Upload successful!")
                typer.echo(f"  Revision: {revision_value}")

            except typer.Exit:
                raise
            except Exception as e:
                typer.echo(f"\nUpload failed: {e}", err=True)
                raise typer.Exit(1)
    else:
        if revision_value is None:
            typer.echo("Error: --revision is required when --skip-upload is set", err=True)
            raise typer.Exit(1)
        typer.echo(f"\nSkipping upload, using revision: {revision_value[:12]}...")

    # Step 2: Deploy to Basilica
    if not skip_deploy:
        step_num = 2 if not skip_upload else 1
        typer.echo(f"\n[{step_num}/{len(steps)}] Deploying to Basilica...")

        if revision_value is None:
            typer.echo("Error: revision is required for deployment", err=True)
            raise typer.Exit(1)

        if dry_run:
            typer.echo(f"  [DRY RUN] Would deploy {repo}@{revision_value[:12]}...")
            deployment_id = "dry-run-deployment-id"
        else:
            # Derive deployment name
            name = deployment_name or repo.replace("/", "-").lower()
            name = "".join(c if c.isalnum() or c == "-" else "-" for c in name).strip("-")[:63]

            typer.echo(f"  Deployment Name: {name}")

            # Create client and build deployment configuration
            client = BasilicaClient(api_key=api_token)
            source_code = _get_deployment_source(repo, revision_value)

            # Note: kinitro is NOT included in pip_packages because the miner template
            # is self-contained (includes rl_interface.py locally).
            env_vars: dict[str, str] = {"HF_REPO": repo, "HF_REVISION": revision_value}
            if hf:
                env_vars["HF_TOKEN"] = hf

            deploy_kwargs: dict = {
                "name": name,
                "source": source_code,
                "image": _get_docker_image(gpu_count),
                "port": 8000,
                "env": env_vars,
                "cpu": cpu,
                "memory": memory,
                "pip_packages": PIP_PACKAGES,
                "timeout": timeout,
            }

            if gpu_count > 0:
                deploy_kwargs["gpu_count"] = gpu_count
                if min_gpu_memory_gb is not None:
                    deploy_kwargs["min_gpu_memory_gb"] = min_gpu_memory_gb

            # Deploy
            typer.echo("  Deploying (this may take several minutes)...")
            try:
                deployment = client.deploy(**deploy_kwargs)
                typer.echo("  Deployment successful!")
                typer.echo(f"    Name: {deployment.name}")
                typer.echo(f"    URL: {deployment.url}")
                typer.echo(f"    State: {deployment.state}")

                deployment_id = _extract_deployment_id(deployment)
                typer.echo(f"    Deployment ID: {deployment_id}")
            except Exception as e:
                typer.echo(f"\nDeployment failed: {e}", err=True)
                raise typer.Exit(1)
    else:
        typer.echo(f"\nSkipping deployment, using deployment ID: {deployment_id}")

    # Step 3: Commit on-chain
    if not skip_commit and deployment_id:
        step_num = len(steps)
        typer.echo(f"\n[{step_num}/{len(steps)}] Committing on-chain...")

        if revision_value is None:
            typer.echo("Error: revision is required for on-chain commit", err=True)
            raise typer.Exit(1)
        if netuid is None:
            typer.echo("Error: netuid is required for on-chain commit", err=True)
            raise typer.Exit(1)

        if dry_run:
            typer.echo(
                f"  [DRY RUN] Would commit {repo}@{revision_value[:12]}... with deployment_id {deployment_id}"
            )
        else:
            wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
            typer.echo(f"  Wallet: {wallet.hotkey.ss58_address[:16]}...")

            async def _commit_on_chain() -> bool:
                async with AsyncSubtensor(network=network) as subtensor:
                    return await commit_model_async(
                        subtensor=subtensor,
                        wallet=wallet,
                        netuid=netuid,
                        repo=repo,
                        revision=revision_value,
                        deployment_id=deployment_id,
                    )

            success = asyncio.run(_commit_on_chain())

            if success:
                typer.echo("  Commitment successful!")
            else:
                typer.echo("  Commitment failed!", err=True)
                raise typer.Exit(1)

    # Summary
    typer.echo("\n" + "=" * 60)
    if dry_run:
        typer.echo("DRY RUN COMPLETE - No changes were made")
    else:
        typer.echo("DEPLOYMENT COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"  Repository: {repo}")
    revision_summary = revision_value[:12] if revision_value else "N/A"
    typer.echo(f"  Revision: {revision_summary}...")
    typer.echo(f"  Deployment ID: {deployment_id or 'N/A'}")
    typer.echo("=" * 60)
