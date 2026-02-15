"""Deployment commands for Basilica and one-command deployment."""

import asyncio
import os

import typer
from basilica import BasilicaClient
from bittensor import AsyncSubtensor
from bittensor_wallet import Wallet

from kinitro.chain.commitments import commit_model_async


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
    image: str = typer.Option(
        ...,
        "--image",
        "-i",
        help="Docker image to deploy (e.g., user/policy:v1)",
    ),
    deployment_name: str = typer.Option(..., "--name", "-n", help="Deployment name"),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Number of GPUs (0 for CPU-only)"),
    min_gpu_memory_gb: int | None = typer.Option(None, "--min-vram", help="Minimum GPU VRAM in GB"),
    cpu: str = typer.Option("1", "--cpu", help="CPU allocation (e.g., '1', '2', '500m', '4000m')"),
    memory: str = typer.Option(
        "512Mi", "--memory", help="Memory allocation (e.g., '512Mi', '16Gi')"
    ),
    basilica_api_token: str | None = typer.Option(
        None, "--api-token", envvar="BASILICA_API_TOKEN", help="Basilica API token"
    ),
    timeout: int = typer.Option(600, "--timeout", help="Deployment timeout in seconds"),
):
    """
    Deploy a Docker image to Basilica.

    Requires --image and --name.
    Requires BASILICA_API_TOKEN environment variable or --api-token.

    Example:
        kinitro miner push --image user/policy:v1 --name my-policy

        # With GPU
        kinitro miner push --image user/policy:v1 --name my-policy --gpu-count 1 --min-vram 16
    """
    # Validate required credentials
    api_token = basilica_api_token or os.environ.get("BASILICA_API_TOKEN")

    if not api_token:
        typer.echo("Error: BASILICA_API_TOKEN not configured", err=True)
        typer.echo("Set it via --api-token or BASILICA_API_TOKEN environment variable")
        typer.echo("\nTo get a token, run: basilica tokens create")
        raise typer.Exit(1)

    client = BasilicaClient(api_key=api_token)

    name = "".join(c if c.isalnum() or c == "-" else "-" for c in deployment_name).strip("-")[:63]

    typer.echo("Deploying to Basilica:")
    typer.echo(f"  Image: {image}")
    typer.echo(f"  Deployment Name: {name}")
    vram_str = f" (min {min_gpu_memory_gb}GB VRAM)" if min_gpu_memory_gb else ""
    typer.echo(f"  GPU: {gpu_count}x{vram_str}")
    typer.echo(f"  Memory: {memory}")

    typer.echo("\nDeploying to Basilica (this may take several minutes)...")
    try:
        gpu_kwargs: dict = {}
        if gpu_count > 0:
            gpu_kwargs["gpu_count"] = gpu_count
            if min_gpu_memory_gb is not None:
                gpu_kwargs["min_gpu_memory_gb"] = min_gpu_memory_gb
        deployment = client.deploy(
            name=name,
            image=image,
            port=8000,
            cpu=cpu,
            memory=memory,
            timeout=timeout,
            **gpu_kwargs,
        )
    except Exception as e:
        typer.echo(f"\nDeployment failed: {e}", err=True)
        raise typer.Exit(1)

    # Enroll for public metadata so validators can verify the deployment
    try:
        client.enroll_metadata(deployment.name, enabled=True)
        typer.echo("  Public metadata enrolled for validator verification.")
    except Exception as e:
        typer.echo(f"  Warning: Could not enroll public metadata: {e}", err=True)

    deploy_id = _extract_deployment_id(deployment)

    typer.echo("\n" + "=" * 60)
    typer.echo("DEPLOYMENT SUCCESSFUL")
    typer.echo("=" * 60)
    typer.echo(f"  Name: {deployment.name}")
    typer.echo(f"  URL: {deployment.url}")
    typer.echo(f"  State: {deployment.state}")
    typer.echo("=" * 60)

    typer.echo("\nNext step - commit on-chain:")
    typer.echo(f"  kinitro miner commit --deployment-id {deploy_id} --netuid YOUR_NETUID")


def miner_deploy(
    image: str = typer.Option(..., "--image", "-i", help="Docker image to deploy"),
    deployment_id: str | None = typer.Option(
        None, "--deployment-id", "-d", help="Basilica deployment ID (skip deploy step)"
    ),
    deployment_name: str | None = typer.Option(
        None, "--name", "-n", help="Deployment name (default: derived from image)"
    ),
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
    gpu_count: int = typer.Option(0, "--gpu-count", help="Number of GPUs (0 for CPU-only)"),
    min_gpu_memory_gb: int | None = typer.Option(None, "--min-vram", help="Minimum GPU VRAM in GB"),
    cpu: str = typer.Option("1", "--cpu", help="CPU allocation (e.g., '1', '2', '500m', '4000m')"),
    memory: str = typer.Option(
        "512Mi", "--memory", help="Memory allocation (e.g., '512Mi', '16Gi')"
    ),
    timeout: int = typer.Option(600, "--timeout", help="Deployment timeout in seconds"),
):
    """
    One-command deployment: Deploy -> Commit.

    Combines the miner deployment workflow into a single command:
    1. Deploy Docker image to Basilica (skip with --skip-deploy)
    2. Commit on-chain (skip with --skip-commit)

    Examples:
        # Full deployment
        kinitro miner deploy --image user/policy:v1 --netuid 123

        # Skip deployment (already deployed)
        kinitro miner deploy --image user/policy:v1 --skip-deploy \\
            --deployment-id my-deploy-id --netuid 123

        # Dry run to see what would happen
        kinitro miner deploy --image user/policy:v1 --netuid 123 --dry-run
    """
    if skip_deploy and not deployment_id:
        typer.echo("Error: --deployment-id is required when --skip-deploy is set", err=True)
        raise typer.Exit(1)

    if not skip_commit and netuid is None:
        typer.echo("Error: --netuid is required unless --skip-commit is set", err=True)
        raise typer.Exit(1)

    # Get credentials from env if not provided
    api_token = basilica_api_token or os.environ.get("BASILICA_API_TOKEN")

    # Validate credentials
    if not dry_run and not skip_deploy and not api_token:
        typer.echo("Error: BASILICA_API_TOKEN not configured", err=True)
        typer.echo("Set it via --api-token or BASILICA_API_TOKEN environment variable")
        typer.echo("\nTo get a token, run: basilica tokens create")
        raise typer.Exit(1)

    # Determine steps
    steps = []
    if not skip_deploy:
        steps.append("deploy")
    if not skip_commit:
        steps.append("commit")

    typer.echo("=" * 60)
    typer.echo("KINITRO DEPLOYMENT")
    typer.echo("=" * 60)
    typer.echo(f"  Image: {image}")
    if deployment_id:
        typer.echo(f"  Deployment ID: {deployment_id}")
    typer.echo(f"  Steps: {' -> '.join(steps) if steps else 'none'}")
    if dry_run:
        typer.echo("  Mode: DRY RUN")
    typer.echo("=" * 60)

    # Step 1: Deploy to Basilica
    if not skip_deploy:
        step_num = 1
        typer.echo(f"\n[{step_num}/{len(steps)}] Deploying to Basilica...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would deploy {image}")
            deployment_id = "dry-run-deployment-id"
        else:
            # Derive deployment name from image
            name = deployment_name or image.split(":", maxsplit=1)[0].replace("/", "-").lower()
            name = "".join(c if c.isalnum() or c == "-" else "-" for c in name).strip("-")[:63]

            typer.echo(f"  Deployment Name: {name}")

            client = BasilicaClient(api_key=api_token)

            deploy_kwargs: dict = {
                "name": name,
                "image": image,
                "port": 8000,
                "cpu": cpu,
                "memory": memory,
                "timeout": timeout,
            }

            if gpu_count > 0:
                deploy_kwargs["gpu_count"] = gpu_count
                if min_gpu_memory_gb is not None:
                    deploy_kwargs["min_gpu_memory_gb"] = min_gpu_memory_gb

            typer.echo("  Deploying (this may take several minutes)...")
            try:
                deployment = client.deploy(**deploy_kwargs)
                typer.echo("  Deployment successful!")
                typer.echo(f"    Name: {deployment.name}")
                typer.echo(f"    URL: {deployment.url}")
                typer.echo(f"    State: {deployment.state}")

                deployment_id = _extract_deployment_id(deployment)
                typer.echo(f"    Deployment ID: {deployment_id}")

                # Enroll for public metadata so validators can verify the deployment
                try:
                    client.enroll_metadata(deployment.name, enabled=True)
                    typer.echo("    Public metadata enrolled for validator verification.")
                except Exception as e:
                    typer.echo(f"    Warning: Could not enroll public metadata: {e}", err=True)
            except Exception as e:
                typer.echo(f"\nDeployment failed: {e}", err=True)
                raise typer.Exit(1)
    else:
        typer.echo(f"\nSkipping deployment, using deployment ID: {deployment_id}")

    # Step 2: Commit on-chain
    if not skip_commit and deployment_id:
        step_num = len(steps)
        typer.echo(f"\n[{step_num}/{len(steps)}] Committing on-chain...")

        if netuid is None:
            typer.echo("Error: netuid is required for on-chain commit", err=True)
            raise typer.Exit(1)

        if dry_run:
            typer.echo(f"  [DRY RUN] Would commit deployment_id {deployment_id}")
        else:
            wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
            typer.echo(f"  Wallet: {wallet.hotkey.ss58_address[:16]}...")

            async def _commit_on_chain() -> bool:
                async with AsyncSubtensor(network=network) as subtensor:
                    return await commit_model_async(
                        subtensor=subtensor,
                        wallet=wallet,
                        netuid=netuid,
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
    typer.echo(f"  Image: {image}")
    typer.echo(f"  Deployment ID: {deployment_id or 'N/A'}")
    typer.echo("=" * 60)
