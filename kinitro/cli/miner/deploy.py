"""Deployment commands for uploading policies to HuggingFace."""

import os

import typer
from huggingface_hub import HfApi

from kinitro.chain.commitments import commit_model


def miner_deploy(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    policy_path: str | None = typer.Option(
        None, "--path", "-p", help="Path to local policy directory"
    ),
    revision: str | None = typer.Option(
        None, "--revision", help="HuggingFace commit SHA (required if --skip-upload)"
    ),
    message: str = typer.Option(
        "Model update", "--message", "-m", help="Commit message for HuggingFace upload"
    ),
    skip_upload: bool = typer.Option(False, "--skip-upload", help="Skip HuggingFace upload"),
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
    hf_token: str | None = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
):
    """
    One-command deployment: Upload -> Commit.

    Combines the miner deployment workflow into a single command:
    1. Upload policy to HuggingFace (skip with --skip-upload)
    2. Commit on-chain (skip with --skip-commit)

    The executor will download your model from HuggingFace and create
    deployments on-demand. You no longer need to deploy to Basilica yourself.

    Examples:
        # Full deployment
        kinitro miner deploy -r user/policy -p ./my-policy --netuid 123

        # Skip upload (already on HuggingFace)
        kinitro miner deploy -r user/policy --skip-upload --revision abc123 --netuid 123

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

    if not skip_commit and netuid is None:
        typer.echo("Error: --netuid is required unless --skip-commit is set", err=True)
        raise typer.Exit(1)

    # Get credentials from env if not provided
    hf = hf_token or os.environ.get("HF_TOKEN")

    # Validate credentials
    if not dry_run:
        if not skip_upload and not hf:
            typer.echo("Error: HF_TOKEN not configured", err=True)
            raise typer.Exit(1)

    revision_value = revision

    # Determine steps
    steps = []
    if not skip_upload:
        steps.append("upload")
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
    typer.echo(f"  Steps: {' -> '.join(steps) if steps else 'none'}")
    if dry_run:
        typer.echo("  Mode: DRY RUN")
    typer.echo("=" * 60)

    # Maximum allowed repo size (configurable via env var)
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

    # Step 2: Commit on-chain
    if not skip_commit:
        step_num = len(steps)
        typer.echo(f"\n[{step_num}/{len(steps)}] Committing on-chain...")

        if revision_value is None:
            typer.echo("Error: revision is required for on-chain commit", err=True)
            raise typer.Exit(1)
        if netuid is None:
            typer.echo("Error: netuid is required for on-chain commit", err=True)
            raise typer.Exit(1)

        if dry_run:
            typer.echo(f"  [DRY RUN] Would commit {repo}@{revision_value[:12]}...")
        else:
            import bittensor as bt  # noqa: PLC0415 - lazy import to avoid argparse hijacking

            subtensor = bt.Subtensor(network=network)
            wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

            typer.echo(f"  Wallet: {wallet.hotkey.ss58_address[:16]}...")

            try:
                success = commit_model(
                    subtensor=subtensor,
                    wallet=wallet,
                    netuid=netuid,
                    repo=repo,
                    revision=revision_value,
                )

                if success:
                    typer.echo("  Commitment successful!")
                else:
                    typer.echo("  Commitment failed!", err=True)
                    raise typer.Exit(1)
            finally:
                subtensor.close()

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
    typer.echo("=" * 60)
