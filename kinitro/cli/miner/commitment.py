"""Chain commitment commands for miners."""

import typer

from kinitro.chain.commitments import (
    _query_commitment_by_hotkey,
    commit_model,
    parse_commitment,
)


def commit(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repo (user/model)"),
    revision: str = typer.Option(..., "--revision", help="Commit SHA"),
    network: str = typer.Option("finney", "--network", help="Network"),
    netuid: int = typer.Option(..., "--netuid", help="Subnet UID"),
    wallet_name: str = typer.Option("default", "--wallet-name", help="Wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey-name", help="Hotkey name"),
):
    """
    Commit model to chain.

    Registers your policy so validators can evaluate it. The executor will
    download your model from HuggingFace and create deployments on-demand.

    Example:
        kinitro miner commit --repo user/policy --revision abc123 --netuid 1
    """
    import bittensor as bt  # noqa: PLC0415 - lazy import to avoid argparse hijacking

    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

    typer.echo(f"Committing model to {network} (netuid={netuid})")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision}")

    try:
        success = commit_model(
            subtensor=subtensor,
            wallet=wallet,
            netuid=netuid,
            repo=repo,
            revision=revision,
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
    uid: int | None = typer.Option(None, help="UID to query (if not using wallet)"),
    hotkey: str | None = typer.Option(None, help="Hotkey address to query (if not using wallet)"),
):
    """
    Show commitment for a miner.

    Query by wallet (default), UID, or hotkey address.
    """
    import bittensor as bt  # noqa: PLC0415 - lazy import to avoid argparse hijacking

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
            typer.echo("\nNo commitment found.", err=True)
            raise typer.Exit(1)

        typer.echo(f"\nRaw commitment: {raw[:100]}{'...' if len(raw) > 100 else ''}")
        if block is not None:
            typer.echo(f"Committed at block: {block}")

        # Parse the commitment
        parsed = parse_commitment(raw)
        if parsed["huggingface_repo"]:
            typer.echo("\nParsed commitment:")
            typer.echo(f"  Repo: {parsed['huggingface_repo']}")
            typer.echo(f"  Revision: {parsed['revision_sha']}")
        else:
            typer.echo("\nCould not parse commitment format.")
    finally:
        subtensor.close()
