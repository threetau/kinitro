"""Cryptographic key management commands for backend operators."""

import json
from pathlib import Path

import typer

from kinitro.crypto import BackendKeypair

# Subcommand group for crypto operations
crypto_app = typer.Typer(
    help="Cryptographic key management for backend operators", 
    add_completion=False, 
    no_args_is_help=True
)


# Backend public key commitment format
# Stored as: {"type": "backend_pubkey", "key": "<hex>"}
BACKEND_PUBKEY_TYPE = "backend_pubkey"


def _parse_backend_pubkey_commitment(raw: str) -> str | None:
    """Parse a backend public key from a chain commitment."""
    if not raw:
        return None
    try:
        if raw.strip().startswith("{"):
            data = json.loads(raw)
            if data.get("type") == BACKEND_PUBKEY_TYPE or data.get("t") == "pk":
                return data.get("key") or data.get("k")
    except json.JSONDecodeError:
        pass
    return None


def fetch_backend_public_key(
    network: str,
    netuid: int,
    backend_hotkey: str,
) -> str | None:
    """
    Fetch the backend operator's public key from the chain.
    
    Args:
        network: Bittensor network
        netuid: Subnet UID
        backend_hotkey: Backend operator's hotkey SS58 address
        
    Returns:
        Public key hex string, or None if not found
    """
    import bittensor as bt
    from kinitro.chain.commitments import _query_commitment_by_hotkey
    
    subtensor = bt.Subtensor(network=network)
    raw, _ = _query_commitment_by_hotkey(subtensor, netuid, backend_hotkey)
    
    if raw:
        return _parse_backend_pubkey_commitment(raw)
    return None


@crypto_app.command("generate-keypair")
def generate_keypair(
    output_dir: str = typer.Option(
        ".",
        "--output",
        "-o",
        help="Directory to save keypair files",
    ),
    name: str = typer.Option(
        "backend",
        "--name",
        "-n",
        help="Name prefix for key files",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite existing key files",
    ),
):
    """
    Generate a new X25519 keypair for endpoint encryption.

    Creates two files:
    - {name}.key: Private key (hex-encoded) - KEEP THIS SECRET!
    - {name}.pub: Public key (hex-encoded) - Share with miners

    The public key should be distributed to miners so they can encrypt
    their deployment endpoints. The private key should be kept secure
    and configured for the scheduler to decrypt miner endpoints.

    Example:
        kinitro crypto generate-keypair --output ~/.kinitro --name backend

        # This creates:
        #   ~/.kinitro/backend.key (private, 0600 permissions)
        #   ~/.kinitro/backend.pub (public)
    """
    output_path = Path(output_dir)
    private_key_file = output_path / f"{name}.key"
    public_key_file = output_path / f"{name}.pub"

    # Check if files exist
    if not force:
        if private_key_file.exists():
            typer.echo(f"Error: {private_key_file} already exists. Use --force to overwrite.", err=True)
            raise typer.Exit(1)
        if public_key_file.exists():
            typer.echo(f"Error: {public_key_file} already exists. Use --force to overwrite.", err=True)
            raise typer.Exit(1)

    # Generate keypair
    keypair = BackendKeypair.generate()

    # Save keys
    keypair.save_private_key(private_key_file)
    keypair.save_public_key(public_key_file)

    typer.echo("Keypair generated successfully!")
    typer.echo("")
    typer.echo(f"Private key: {private_key_file}")
    typer.echo(f"  (permissions: 0600 - keep this secret!)")
    typer.echo("")
    typer.echo(f"Public key: {public_key_file}")
    typer.echo(f"  Value: {keypair.public_key_hex()}")
    typer.echo("")
    typer.echo("Next steps:")
    typer.echo("  1. Publish the public key to the chain:")
    typer.echo(f"     kinitro crypto publish-public-key --private-key-file {private_key_file} --netuid <NETUID>")
    typer.echo("")
    typer.echo("  2. Configure the scheduler with the private key:")
    typer.echo(f"     export KINITRO_SCHEDULER_BACKEND_PRIVATE_KEY_FILE={private_key_file}")
    typer.echo("     OR")
    typer.echo(f"     --backend-private-key-file {private_key_file}")


@crypto_app.command("publish-public-key")
def publish_public_key(
    private_key_file: str = typer.Option(
        None,
        "--private-key-file",
        "-f",
        help="Path to private key file",
    ),
    private_key: str = typer.Option(
        None,
        "--private-key",
        "-k",
        help="Private key as hex string",
    ),
    network: str = typer.Option("finney", "--network", help="Bittensor network"),
    netuid: int = typer.Option(..., "--netuid", help="Subnet UID"),
    wallet_name: str = typer.Option("default", "--wallet-name", help="Wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey-name", help="Hotkey name"),
):
    """
    Publish backend public key to the chain.

    This allows miners to fetch your public key by specifying your hotkey,
    rather than manually sharing the key.

    Example:
        kinitro crypto publish-public-key \\
            --private-key-file ~/.kinitro/backend.key \\
            --netuid 1

    Miners can then use:
        kinitro miner commit ... --encrypt --backend-hotkey <YOUR_HOTKEY>
    """
    import bittensor as bt

    if not private_key_file and not private_key:
        typer.echo("Error: Provide --private-key-file or --private-key", err=True)
        raise typer.Exit(1)

    # Load keypair
    try:
        if private_key_file:
            keypair = BackendKeypair.from_private_key_file(private_key_file)
        else:
            keypair = BackendKeypair.from_private_key_hex(private_key)
    except Exception as e:
        typer.echo(f"Error loading private key: {e}", err=True)
        raise typer.Exit(1)

    public_key_hex = keypair.public_key_hex()

    # Create commitment data
    # Use compact format: {"t": "pk", "k": "<hex>"}
    commitment_data = json.dumps(
        {"t": "pk", "k": public_key_hex},
        separators=(",", ":"),
    )

    typer.echo(f"Publishing backend public key to {network} (netuid={netuid})")
    typer.echo(f"  Public key: {public_key_hex}")
    typer.echo(f"  Commitment size: {len(commitment_data)} bytes")

    # Connect and publish
    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

    typer.echo(f"  Wallet: {wallet_name}/{hotkey_name}")
    typer.echo(f"  Hotkey: {wallet.hotkey.ss58_address}")

    try:
        result = subtensor.set_commitment(
            wallet=wallet,
            netuid=netuid,
            data=commitment_data,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

        success = bool(result) if not hasattr(result, "is_success") else result.is_success
        if success:
            typer.echo("")
            typer.echo("Public key published successfully!")
            typer.echo("")
            typer.echo("Miners can now encrypt their endpoints using:")
            typer.echo(f"  kinitro miner commit ... --encrypt --backend-hotkey {wallet.hotkey.ss58_address}")
        else:
            typer.echo("Failed to publish public key!", err=True)
            raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Error publishing: {e}", err=True)
        raise typer.Exit(1)


@crypto_app.command("fetch-public-key")
def fetch_public_key_cmd(
    backend_hotkey: str = typer.Option(..., "--backend-hotkey", "-b", help="Backend operator's hotkey"),
    network: str = typer.Option("finney", "--network", help="Bittensor network"),
    netuid: int = typer.Option(..., "--netuid", help="Subnet UID"),
):
    """
    Fetch a backend operator's public key from the chain.

    This is what miners use internally when specifying --backend-hotkey.

    Example:
        kinitro crypto fetch-public-key --backend-hotkey 5Dxxx... --netuid 1
    """
    typer.echo(f"Fetching public key from {network} (netuid={netuid})")
    typer.echo(f"  Backend hotkey: {backend_hotkey}")

    public_key = fetch_backend_public_key(network, netuid, backend_hotkey)

    if public_key:
        typer.echo("")
        typer.echo(f"Public key: {public_key}")
    else:
        typer.echo("")
        typer.echo("No public key found for this hotkey.", err=True)
        typer.echo("The backend operator may not have published their key yet.")
        raise typer.Exit(1)


@crypto_app.command("show-public-key")
def show_public_key(
    private_key: str = typer.Option(
        None,
        "--private-key",
        "-k",
        help="Private key as hex string",
    ),
    private_key_file: str = typer.Option(
        None,
        "--private-key-file",
        "-f",
        help="Path to private key file",
    ),
):
    """
    Show the public key corresponding to a private key.

    Useful for deriving the public key to share with miners.

    Example:
        kinitro crypto show-public-key --private-key-file ~/.kinitro/backend.key
    """
    if not private_key and not private_key_file:
        typer.echo("Error: Provide --private-key or --private-key-file", err=True)
        raise typer.Exit(1)

    try:
        if private_key:
            keypair = BackendKeypair.from_private_key_hex(private_key)
        else:
            keypair = BackendKeypair.from_private_key_file(private_key_file)
    except Exception as e:
        typer.echo(f"Error loading private key: {e}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Public key: {keypair.public_key_hex()}")


@crypto_app.command("test-encryption")
def test_encryption(
    deployment_id: str = typer.Option(
        "95edf2b6-e18b-400a-8398-5573df10e5e4",
        "--deployment-id",
        "-d",
        help="Deployment UUID to encrypt (for testing)",
    ),
    public_key: str = typer.Option(
        None,
        "--public-key",
        "-p",
        help="Public key to encrypt with (hex)",
    ),
    private_key: str = typer.Option(
        None,
        "--private-key",
        "-k",
        help="Private key to verify decryption (hex)",
    ),
    private_key_file: str = typer.Option(
        None,
        "--private-key-file",
        "-f",
        help="Path to private key file",
    ),
):
    """
    Test encryption/decryption of a deployment ID.

    This is useful for verifying your keypair works correctly.

    Example:
        # Generate a keypair first
        kinitro crypto generate-keypair -o /tmp -n test

        # Test with the private key file (derives public key automatically)
        kinitro crypto test-encryption --private-key-file /tmp/test.key

        # Or specify both keys explicitly
        kinitro crypto test-encryption --public-key <pub> --private-key <priv>
    """
    from kinitro.crypto import encrypt_deployment_id, decrypt_deployment_id

    # Load keys
    if private_key_file:
        keypair = BackendKeypair.from_private_key_file(private_key_file)
        if not public_key:
            public_key = keypair.public_key_hex()
        private_key = keypair.private_key_hex()
    elif private_key and not public_key:
        keypair = BackendKeypair.from_private_key_hex(private_key)
        public_key = keypair.public_key_hex()

    if not public_key:
        typer.echo("Error: Provide --public-key or --private-key-file", err=True)
        raise typer.Exit(1)

    typer.echo(f"Testing encryption with:")
    typer.echo(f"  Deployment ID: {deployment_id}")
    typer.echo(f"  Public key: {public_key[:16]}...")
    typer.echo("")

    # Encrypt
    try:
        encrypted = encrypt_deployment_id(deployment_id, public_key)
        typer.echo(f"Encrypted blob ({len(encrypted)} chars):")
        typer.echo(f"  {encrypted}")
    except Exception as e:
        typer.echo(f"Encryption failed: {e}", err=True)
        raise typer.Exit(1)

    # Decrypt if private key available
    if private_key:
        typer.echo("")
        try:
            keypair = BackendKeypair.from_private_key_hex(private_key)
            decrypted = decrypt_deployment_id(encrypted, keypair.private_key)
            typer.echo(f"Decrypted: {decrypted}")
            if decrypted == deployment_id:
                typer.echo("SUCCESS: Decryption matches original!")
            else:
                typer.echo("ERROR: Decryption mismatch!", err=True)
                raise typer.Exit(1)
        except Exception as e:
            typer.echo(f"Decryption failed: {e}", err=True)
            raise typer.Exit(1)
    else:
        typer.echo("")
        typer.echo("(Provide --private-key or --private-key-file to verify decryption)")
