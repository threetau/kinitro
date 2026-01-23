"""One-command miner deployment."""

import asyncio
import os
import subprocess
import textwrap
from pathlib import Path

import typer


def miner_deploy(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    policy_path: str | None = typer.Option(
        None, "--path", "-p", help="Path to local policy directory"
    ),
    revision: str | None = typer.Option(
        None, "--revision", help="HuggingFace commit SHA (required if --skip-upload)"
    ),
    chute_id: str | None = typer.Option(
        None, "--chute-id", help="Chutes deployment ID (required if --skip-chutes)"
    ),
    message: str = typer.Option(
        "Model update", "--message", "-m", help="Commit message for HuggingFace upload"
    ),
    skip_upload: bool = typer.Option(False, "--skip-upload", help="Skip HuggingFace upload"),
    skip_chutes: bool = typer.Option(False, "--skip-chutes", help="Skip Chutes deployment"),
    skip_commit: bool = typer.Option(False, "--skip-commit", help="Skip on-chain commit"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without executing"
    ),
    network: str = typer.Option("finney", "--network", help="Bittensor network"),
    netuid: int = typer.Option(..., "--netuid", help="Subnet UID"),
    wallet_name: str = typer.Option("default", "--wallet-name", help="Wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey-name", help="Hotkey name"),
    chutes_api_key: str | None = typer.Option(None, "--chutes-api-key", envvar="CHUTES_API_KEY"),
    chute_user: str | None = typer.Option(None, "--chute-user", envvar="CHUTE_USER"),
    hf_token: str | None = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
):
    """
    One-command deployment: Upload -> Deploy -> Commit.

    Combines the miner deployment workflow into a single command:
    1. Upload policy to HuggingFace (skip with --skip-upload)
    2. Deploy to Chutes (skip with --skip-chutes)
    3. Commit on-chain (skip with --skip-commit)

    Examples:
        # Full deployment
        kinitro miner-deploy -r user/policy -p ./my-policy --netuid 123

        # Skip upload (already on HuggingFace)
        kinitro miner-deploy -r user/policy --skip-upload --revision abc123 --netuid 123

        # Dry run to see what would happen
        kinitro miner-deploy -r user/policy -p ./my-policy --netuid 123 --dry-run
    """
    # Validate arguments
    if not skip_upload and not policy_path:
        typer.echo("Error: --path is required unless --skip-upload is set", err=True)
        raise typer.Exit(1)

    if skip_upload and not revision:
        typer.echo("Error: --revision is required when --skip-upload is set", err=True)
        raise typer.Exit(1)

    if skip_chutes and not chute_id:
        typer.echo("Error: --chute-id is required when --skip-chutes is set", err=True)
        raise typer.Exit(1)

    # Get credentials from env if not provided
    api_key = chutes_api_key or os.environ.get("CHUTES_API_KEY")
    user = chute_user or os.environ.get("CHUTE_USER")
    hf = hf_token or os.environ.get("HF_TOKEN")

    # Validate credentials
    if not dry_run:
        if not skip_upload and not hf:
            typer.echo("Error: HF_TOKEN not configured", err=True)
            raise typer.Exit(1)
        if not skip_chutes and not api_key:
            typer.echo("Error: CHUTES_API_KEY not configured", err=True)
            raise typer.Exit(1)
        if not skip_chutes and not user:
            typer.echo("Error: CHUTE_USER not configured", err=True)
            raise typer.Exit(1)

    # Determine steps
    steps = []
    if not skip_upload:
        steps.append("upload")
    if not skip_chutes:
        steps.append("chutes")
    if not skip_commit:
        steps.append("commit")

    typer.echo("=" * 60)
    typer.echo("KINITRO DEPLOYMENT")
    typer.echo("=" * 60)
    typer.echo(f"  Repository: {repo}")
    if policy_path:
        typer.echo(f"  Policy Path: {policy_path}")
    if revision:
        typer.echo(f"  Revision: {revision}")
    if chute_id:
        typer.echo(f"  Chute ID: {chute_id}")
    typer.echo(f"  Steps: {' -> '.join(steps) if steps else 'none'}")
    if dry_run:
        typer.echo("  Mode: DRY RUN")
    typer.echo("=" * 60)

    # Step 1: Upload to HuggingFace
    if not skip_upload:
        typer.echo(f"\n[1/{len(steps)}] Uploading to HuggingFace ({repo})...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would upload {policy_path} to {repo}")
            revision = "dry-run-revision"
        else:
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=hf)

                # Create repo if it doesn't exist
                try:
                    api.create_repo(repo, exist_ok=True, repo_type="model")
                except Exception:
                    pass  # Repo may already exist

                # Upload folder
                typer.echo(f"  Uploading {policy_path}...")
                api.upload_folder(
                    folder_path=policy_path,
                    repo_id=repo,
                    commit_message=message,
                )

                # Get latest commit SHA
                info = api.repo_info(repo, repo_type="model")
                revision = info.sha

                typer.echo(f"  Upload complete. Revision: {revision[:12]}...")

            except Exception as e:
                typer.echo(f"HuggingFace upload failed: {e}", err=True)
                raise typer.Exit(1)
    else:
        typer.echo(f"\nSkipping upload, using revision: {revision[:12]}...")

    # Step 2: Deploy to Chutes
    if not skip_chutes:
        step_num = 2 if not skip_upload else 1
        typer.echo(f"\n[{step_num}/{len(steps)}] Deploying to Chutes...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would deploy {repo}@{revision[:12]}...")
            chute_id = "dry-run-chute-id"
        else:
            # Generate Chute config
            chute_config = textwrap.dedent(f'''
import os
from chutes import Chute
from chutes.chute import NodeSelector

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = Chute(
    name="{repo.replace("/", "-")}",
    readme="{repo}",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb=8),
)

chute.from_huggingface("{repo}", revision="{revision}")
chute.add_pip_requirements("requirements.txt")
chute.entrypoint = "uvicorn server:app --host 0.0.0.0 --port 8000"
''')

            tmp_file = Path("tmp_kinitro_chute.py")
            tmp_file.write_text(chute_config)

            try:
                cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
                env = {**os.environ, "CHUTES_API_KEY": api_key}

                result = subprocess.run(cmd, env=env, capture_output=True, text=True)

                if result.returncode != 0:
                    typer.echo(f"Chutes deployment failed: {result.stderr}", err=True)
                    raise typer.Exit(1)

                # Try to get chute_id from Chutes API
                try:
                    import aiohttp

                    async def get_chute():
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                "https://api.chutes.ai/chutes/", headers={"Authorization": api_key}
                            ) as r:
                                if r.status == 200:
                                    data = await r.json()
                                    chutes = (
                                        data.get("items", data) if isinstance(data, dict) else data
                                    )
                                    for c in reversed(chutes):
                                        if c.get("readme") == repo or c.get("model_name") == repo:
                                            return c.get("chute_id")
                        return None

                    chute_id = asyncio.run(get_chute())
                except Exception:
                    chute_id = None

                if chute_id:
                    typer.echo(f"  Chute ID: {chute_id}")
                else:
                    typer.echo("  Deployment submitted. Check Chutes dashboard for chute_id.")
                    typer.echo("  You may need to run commit separately with --chute-id")
                    if not skip_commit:
                        typer.echo("  Skipping commit step (no chute_id)")
                        skip_commit = True

            finally:
                tmp_file.unlink(missing_ok=True)
    else:
        typer.echo(f"\nSkipping Chutes deployment, using chute_id: {chute_id}")

    # Step 3: Commit on-chain
    if not skip_commit and chute_id:
        step_num = len(steps)
        typer.echo(f"\n[{step_num}/{len(steps)}] Committing on-chain...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would commit {repo}@{revision[:12]}... with chute {chute_id}")
        else:
            import bittensor as bt

            from kinitro.chain.commitments import commit_model

            subtensor = bt.Subtensor(network=network)
            wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

            typer.echo(f"  Wallet: {wallet.hotkey.ss58_address[:16]}...")

            success = commit_model(
                subtensor=subtensor,
                wallet=wallet,
                netuid=netuid,
                repo=repo,
                revision=revision,
                chute_id=chute_id,
            )

            if success:
                typer.echo("  Commit successful!")
            else:
                typer.echo("  Commit failed!", err=True)
                raise typer.Exit(1)

    # Summary
    typer.echo("\n" + "=" * 60)
    if dry_run:
        typer.echo("DRY RUN COMPLETE - No changes were made")
    else:
        typer.echo("DEPLOYMENT COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"  Repository: {repo}")
    typer.echo(f"  Revision: {revision[:12] if revision else 'N/A'}...")
    typer.echo(f"  Chute ID: {chute_id or 'N/A'}")
    typer.echo("=" * 60)
