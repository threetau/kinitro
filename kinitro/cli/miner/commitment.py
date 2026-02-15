"""Chain commitment commands for miners."""

import asyncio

import typer
from bittensor import AsyncSubtensor
from bittensor_wallet import Wallet

from kinitro.chain.commitments import (
    _query_commitment_by_hotkey_async,
    commit_model_async,
    parse_commitment,
)
from kinitro.cli.crypto_commands import fetch_backend_public_key


async def _commit_async(
    network: str,
    wallet_name: str,
    hotkey_name: str,
    netuid: int,
    deployment_id: str,
    encrypt: bool,
    backend_public_key: str | None,
) -> bool:
    """Perform the on-chain commit using AsyncSubtensor."""
    async with AsyncSubtensor(network=network) as subtensor:
        wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
        return await commit_model_async(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            deployment_id=deployment_id,
            backend_public_key=backend_public_key if encrypt else None,
        )


async def _show_commitment_async(
    network: str,
    netuid: int,
    query_hotkey: str,
) -> tuple[str | None, int | None]:
    """Query a commitment from chain using AsyncSubtensor."""
    async with AsyncSubtensor(network=network) as subtensor:
        return await _query_commitment_by_hotkey_async(subtensor, netuid, query_hotkey)


async def _get_neurons_hotkey_async(
    network: str,
    netuid: int,
    uid: int,
) -> str | None:
    """Look up a hotkey by UID using AsyncSubtensor."""
    async with AsyncSubtensor(network=network) as subtensor:
        neurons = await subtensor.neurons(netuid=netuid)
        if uid < 0 or uid >= len(neurons):
            return None
        return neurons[uid].hotkey


def commit(
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
    Commit deployment to chain.

    Registers your policy so validators can evaluate it.

    If --encrypt is specified, the deployment endpoint will be encrypted using
    the backend operator's public key. This protects your endpoint from being
    visible on-chain, so only the backend operator can discover and evaluate it.

    You can provide the backend's public key in two ways:
    1. --backend-hotkey: Fetches the public key from the chain (recommended)
    2. --backend-public-key: Provide the hex-encoded key directly

    Example:
        # Plain commitment (endpoint visible on-chain)
        kinitro miner commit --deployment-id UUID --netuid 1

        # Encrypted commitment using backend hotkey (recommended)
        kinitro miner commit --deployment-id UUID \\
            --netuid 1 --encrypt --backend-hotkey 5Dxxx...

        # Encrypted commitment using explicit public key
        kinitro miner commit --deployment-id UUID \\
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

    typer.echo(f"Committing deployment to {network} (netuid={netuid})")
    typer.echo(f"  Deployment ID: {deployment_id}")
    if encrypt:
        typer.echo("  Encryption: ENABLED")
    else:
        typer.echo("  Encryption: disabled (endpoint visible on-chain)")

    try:
        success = asyncio.run(
            _commit_async(
                network=network,
                wallet_name=wallet_name,
                hotkey_name=hotkey_name,
                netuid=netuid,
                deployment_id=deployment_id,
                encrypt=encrypt,
                backend_public_key=backend_public_key,
            )
        )

        if success:
            typer.echo("Commitment successful!")
        else:
            typer.echo("Commitment failed!", err=True)
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Commitment failed: {e}", err=True)
        raise typer.Exit(1)


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
    # Determine which hotkey to query
    if hotkey:
        query_hotkey = hotkey
        typer.echo(f"Querying commitment for hotkey: {hotkey[:16]}...")
    elif uid is not None:
        result_hotkey = asyncio.run(_get_neurons_hotkey_async(network, netuid, uid))
        if result_hotkey is None:
            typer.echo(f"UID {uid} not found on subnet {netuid}", err=True)
            raise typer.Exit(1)
        query_hotkey = result_hotkey
        typer.echo(f"Querying commitment for UID {uid} ({query_hotkey[:16]}...)")
    else:
        wallet = Wallet(name=wallet_name, hotkey=hotkey_name)
        query_hotkey = wallet.hotkey.ss58_address
        typer.echo(f"Querying commitment for wallet {wallet_name}/{hotkey_name}")
        typer.echo(f"  Hotkey: {query_hotkey}")

    # Query the commitment
    raw, block = asyncio.run(_show_commitment_async(network, netuid, query_hotkey))

    if not raw:
        typer.echo("\nNo commitment found.", err=True)
        raise typer.Exit(1)

    typer.echo(f"\nRaw commitment: {raw[:100]}{'...' if len(raw) > 100 else ''}")
    if block is not None:
        typer.echo(f"Committed at block: {block}")

    # Parse the commitment
    parsed = parse_commitment(raw)
    if parsed["deployment_id"] or parsed.get("encrypted_deployment"):
        typer.echo("\nParsed commitment:")
        encrypted_blob = parsed.get("encrypted_deployment")
        if encrypted_blob:
            typer.echo("  Encrypted: YES")
            typer.echo(f"  Encrypted Blob: {encrypted_blob[:40]}...")
        else:
            typer.echo(f"  Deployment ID: {parsed['deployment_id']}")
    else:
        typer.echo("\nCould not parse commitment format.")
