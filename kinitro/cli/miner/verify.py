"""Standalone metadata verification command for miners."""

import asyncio

import typer
from bittensor import AsyncSubtensor

from kinitro.chain.commitments import (
    MinerCommitment,
    _query_commitment_by_hotkey_async,
    parse_commitment,
)
from kinitro.executor.verification import MetadataVerifier
from kinitro.types import BlockNumber, Hotkey, MinerUID


async def _get_hotkey_for_uid(network: str, netuid: int, uid: int) -> str | None:
    """Look up a hotkey by UID using AsyncSubtensor."""
    async with AsyncSubtensor(network=network) as subtensor:
        neurons = await subtensor.neurons(netuid=netuid)
        if uid < 0 or uid >= len(neurons):
            return None
        return neurons[uid].hotkey


async def _read_commitment_from_chain(
    network: str, netuid: int, hotkey: str
) -> MinerCommitment | None:
    """Read and parse a miner's commitment from chain."""
    async with AsyncSubtensor(network=network) as subtensor:
        raw, block = await _query_commitment_by_hotkey_async(subtensor, netuid, hotkey)

    if not raw:
        return None

    parsed = parse_commitment(raw)
    if not parsed["deployment_id"] and not parsed.get("encrypted_deployment"):
        return None

    return MinerCommitment(
        uid=MinerUID(0),
        hotkey=Hotkey(hotkey),
        deployment_id=parsed["deployment_id"],
        committed_block=BlockNumber(block if block is not None else 0),
        encrypted_deployment=parsed.get("encrypted_deployment"),
    )


def _print_result(result) -> None:
    """Print a MetadataVerificationResult in a human-readable format."""
    status = "VERIFIED" if result.verified else "FAILED"
    typer.echo(f"\nVerification: {status}")
    typer.echo(f"  Deployment ID: {result.deployment_id}")
    if result.state is not None:
        typer.echo(f"  State: {result.state}")
    if result.image is not None:
        tag_str = f":{result.image_tag}" if result.image_tag else ""
        typer.echo(f"  Image: {result.image}{tag_str}")
    if result.image_public is not None:
        typer.echo(f"  Image Public: {result.image_public}")
    if result.uptime_seconds is not None:
        typer.echo(f"  Uptime: {result.uptime_seconds:.0f}s")
    if result.failure_reason:
        typer.echo(f"  Failure Reason: {result.failure_reason}")
    if result.error:
        typer.echo(f"  Error: {result.error}")


def verify(
    deployment_id: str | None = typer.Option(
        None, "--deployment-id", "-d", help="Basilica deployment name to verify directly"
    ),
    netuid: int | None = typer.Option(None, "--netuid", help="Subnet UID (for chain lookup)"),
    uid: int | None = typer.Option(None, "--uid", help="Miner UID (requires --netuid)"),
    hotkey: str | None = typer.Option(
        None, "--hotkey", help="Miner hotkey SS58 address (requires --netuid)"
    ),
    network: str = typer.Option("finney", "--network", help="Bittensor network"),
):
    """
    Verify a miner's Basilica deployment metadata.

    Two modes:

    1. Direct: Verify a specific deployment by its Basilica deployment name.
       kinitro miner verify --deployment-id my-deployment

    2. Chain: Read a miner's commitment from chain, then verify the deployment.
       kinitro miner verify --netuid 1 --uid 5
       kinitro miner verify --netuid 1 --hotkey 5Dxxx...
    """
    if deployment_id:
        # Direct mode: verify a specific deployment
        commitment = MinerCommitment(
            uid=MinerUID(0),
            hotkey=Hotkey(""),
            deployment_id=deployment_id,
            committed_block=BlockNumber(0),
        )
        typer.echo(f"Verifying deployment: {deployment_id}")

    elif netuid is not None and (uid is not None or hotkey is not None):
        # Chain mode: read commitment then verify
        if hotkey:
            query_hotkey = hotkey
            typer.echo(f"Looking up commitment for hotkey {hotkey[:16]}... on netuid {netuid}")
        elif uid is not None:
            typer.echo(f"Looking up hotkey for UID {uid} on netuid {netuid}...")
            query_hotkey = asyncio.run(_get_hotkey_for_uid(network, netuid, uid))
            if not query_hotkey:
                typer.echo(f"Error: UID {uid} not found on subnet {netuid}", err=True)
                raise typer.Exit(1)
            typer.echo(f"  Hotkey: {query_hotkey[:16]}...")
        else:
            typer.echo("Error: --uid or --hotkey required with --netuid", err=True)
            raise typer.Exit(1)

        typer.echo("Reading commitment from chain...")
        commitment = asyncio.run(_read_commitment_from_chain(network, netuid, query_hotkey))
        if not commitment:
            typer.echo("Error: No valid commitment found on chain", err=True)
            raise typer.Exit(1)

        typer.echo(f"  Deployment ID: {commitment.deployment_id}")

    else:
        typer.echo(
            "Error: Provide --deployment-id, or --netuid with --uid/--hotkey",
            err=True,
        )
        raise typer.Exit(1)

    # Run verification
    typer.echo("Running metadata verification...")
    verifier = MetadataVerifier()
    result = asyncio.run(verifier.verify_miner(commitment))
    _print_result(result)

    if not result.verified:
        raise typer.Exit(1)
