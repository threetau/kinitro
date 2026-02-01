"""Chain commitment commands for miners."""

import bittensor as bt
import typer

from kinitro.chain.commitments import (
    _query_commitment_by_hotkey,
    commit_model,
    parse_commitment,
)
from kinitro.cli.crypto_commands import fetch_backend_public_key


def commit(
    repo: str = typer.Option(..., help="HuggingFace repo (user/model)"),
    revision: str = typer.Option(..., help="Commit SHA"),
    deployment_id: str = typer.Option(
        ..., "--deployment-id", "-d", help="Basilica deployment ID (UUID only)"
    ),
    network: str = typer.Option("finney", help="Network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
    encrypt: bool = typer.Option(
        False,
        "--encrypt",
        "-e",
        help="Encrypt the endpoint using the backend operator's public key",
    ),
    backend_hotkey: str | None = typer.Option(
        None,
        "--backend-hotkey",
        "-b",
        envvar="KINITRO_BACKEND_HOTKEY",
        help="Backend operator's hotkey (fetches public key from chain)",
    ),
    backend_public_key: str | None = typer.Option(
        None,
        "--backend-public-key",
        envvar="KINITRO_BACKEND_PUBLIC_KEY",
        help="Backend operator's X25519 public key (hex, 64 chars) - alternative to --backend-hotkey",
    ),
):
    """
    Commit model to chain.

    Registers your policy so validators can evaluate it.

    If --encrypt is specified, the deployment endpoint will be encrypted using
    the backend operator's public key. This protects your endpoint from being
    visible on-chain, so only the backend operator can discover and evaluate it.

    You can provide the backend's public key in two ways:
    1. --backend-hotkey: Fetches the public key from the chain (recommended)
    2. --backend-public-key: Provide the hex-encoded key directly

    Example:
        # Plain commitment (endpoint visible on-chain)
        kinitro miner commit --repo user/policy --revision abc123 --deployment-id UUID --netuid 1

        # Encrypted commitment using backend hotkey (recommended)
        kinitro miner commit --repo user/policy --revision abc123 --deployment-id UUID \\
            --netuid 1 --encrypt --backend-hotkey 5Dxxx...

        # Encrypted commitment using explicit public key
        kinitro miner commit --repo user/policy --revision abc123 --deployment-id UUID \\
            --netuid 1 --encrypt --backend-public-key <hex>
    """
    # Validate encryption options
    if encrypt and not backend_hotkey and not backend_public_key:
        typer.echo(
            "Error: --encrypt requires --backend-hotkey or --backend-public-key",
            err=True,
        )
        raise typer.Exit(1)

    # Fetch public key from chain if backend_hotkey is provided
    if encrypt and backend_hotkey and not backend_public_key:
        typer.echo("Fetching backend public key from chain...")
        typer.echo(f"  Backend hotkey: {backend_hotkey}")

        backend_public_key = fetch_backend_public_key(network, netuid, backend_hotkey)

        if not backend_public_key:
            typer.echo(
                f"Error: Could not find public key for backend hotkey {backend_hotkey}",
                err=True,
            )
            typer.echo("The backend operator may not have published their key yet.")
            typer.echo("Ask them to run: kinitro crypto publish-public-key --netuid ...")
            raise typer.Exit(1)

        typer.echo(f"  Found public key: {backend_public_key[:16]}...")

    backend_public_key_value = backend_public_key
    if encrypt and backend_public_key_value:
        is_valid_hex = all(c in "0123456789abcdefABCDEF" for c in backend_public_key_value)
        if len(backend_public_key_value) != 64 or not is_valid_hex:
            typer.echo(
                "Error: Invalid public key. Expected 64 hex characters.",
                err=True,
            )
            raise typer.Exit(1)

    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

    typer.echo(f"Committing model to {network} (netuid={netuid})")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision}")
    typer.echo(f"  Deployment ID: {deployment_id}")
    if encrypt:
        typer.echo("  Encryption: ENABLED")
    else:
        typer.echo("  Encryption: disabled (endpoint visible on-chain)")

    try:
        success = commit_model(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            repo=repo,
            revision=revision,
            deployment_id=deployment_id,
            backend_public_key=backend_public_key if encrypt else None,
        )

        if success:
            typer.echo("Commitment successful!")
        else:
            typer.echo("Commitment failed!", err=True)
            raise typer.Exit(1)
    finally:
        subtensor.close()


def show_commitment(
    network: str = typer.Option("finney", help="Network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
    uid: int = typer.Option(None, help="UID to query (if not using wallet)"),
    hotkey: str = typer.Option(None, help="Hotkey address to query (if not using wallet)"),
):
    """
    Show commitment for a miner.

    Query by wallet (default), UID, or hotkey address.
    """
    subtensor = bt.Subtensor(network=network)

    try:
        # Determine which hotkey to query
        if hotkey:
            query_hotkey = hotkey
            typer.echo(f"Querying commitment for hotkey: {hotkey[:16]}...")
        elif uid is not None:
            neurons = subtensor.neurons(netuid=netuid)
            if uid >= len(neurons):
                typer.echo(f"UID {uid} not found on subnet {netuid}", err=True)
                raise typer.Exit(1)
            query_hotkey = neurons[uid].hotkey
            typer.echo(f"Querying commitment for UID {uid} ({query_hotkey[:16]}...)")
        else:
            wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
            query_hotkey = wallet.hotkey.ss58_address
            typer.echo(f"Querying commitment for wallet {wallet_name}/{hotkey_name}")
            typer.echo(f"  Hotkey: {query_hotkey}")

        # Query the commitment
        raw, block = _query_commitment_by_hotkey(subtensor, netuid, query_hotkey)

        if not raw:
            typer.echo("\nNo commitment found.")
            raise typer.Exit(0)

        typer.echo(f"\nRaw commitment: {raw[:100]}{'...' if len(raw) > 100 else ''}")
        if block is not None:
            typer.echo(f"Committed at block: {block}")

        # Parse the commitment (supports both JSON and legacy formats)
        parsed = parse_commitment(raw)
        if parsed["huggingface_repo"]:
            typer.echo("\nParsed commitment:")
            typer.echo(f"  Repo: {parsed['huggingface_repo']}")
            typer.echo(f"  Revision: {parsed['revision_sha']}")
            if parsed.get("encrypted_deployment"):
                typer.echo("  Encrypted: YES")
                typer.echo(f"  Encrypted Blob: {parsed['encrypted_deployment'][:40]}...")
            else:
                typer.echo(f"  Deployment ID: {parsed['deployment_id']}")
            if parsed["docker_image"]:
                typer.echo(f"  Docker Image: {parsed['docker_image']}")
        else:
            typer.echo("\nCould not parse commitment format.")
    finally:
        subtensor.close()
