"""Commit model to chain command."""

import typer


def commit(
    repo: str = typer.Option(..., help="HuggingFace repo (user/model)"),
    revision: str = typer.Option(..., help="Commit SHA"),
    chute_id: str = typer.Option(..., help="Chutes deployment ID"),
    network: str = typer.Option("finney", help="Network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
):
    """
    Commit model to chain (for miners).

    Registers your policy so validators can evaluate it.
    """
    import bittensor as bt

    from kinitro.chain.commitments import commit_model

    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

    typer.echo(f"Committing model to {network} (netuid={netuid})")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision}")
    typer.echo(f"  Chute ID: {chute_id}")

    success = commit_model(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        repo=repo,
        revision=revision,
        chute_id=chute_id,
    )

    if success:
        typer.echo("Commitment successful!")
    else:
        typer.echo("Commitment failed!", err=True)
        raise typer.Exit(1)
