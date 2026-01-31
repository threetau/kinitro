"""Chain commitment commands for miners."""

import bittensor as bt
import typer

from kinitro.chain.commitments import (
    _query_commitment_by_hotkey,
    commit_model,
    parse_commitment,
)


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
):
    """
    Commit model to chain.

    Registers your policy so validators can evaluate it.
    """
    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

    typer.echo(f"Committing model to {network} (netuid={netuid})")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision}")
    typer.echo(f"  Deployment ID: {deployment_id}")

    success = commit_model(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        repo=repo,
        revision=revision,
        deployment_id=deployment_id,
    )

    if success:
        typer.echo("Commitment successful!")
    else:
        typer.echo("Commitment failed!", err=True)
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
    subtensor = bt.Subtensor(network=network)

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
        typer.echo(f"  Deployment ID: {parsed['deployment_id']}")
        if parsed["docker_image"]:
            typer.echo(f"  Docker Image: {parsed['docker_image']}")
    else:
        typer.echo("\nCould not parse commitment format.")
