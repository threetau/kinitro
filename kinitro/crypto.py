"""
Cryptographic utilities for encrypted endpoint commitments.

This module provides X25519 ECDH key exchange with ChaCha20-Poly1305 authenticated
encryption to protect miner deployment endpoints from public disclosure.

Architecture:
    - Backend operator generates a long-term X25519 keypair
    - Backend public key is published (e.g., in subnet metadata or docs)
    - Miners encrypt their deployment ID using the backend's public key
    - Only the backend operator can decrypt endpoints to evaluate miners

Security Properties:
    - Confidentiality: Only backend operator can decrypt endpoints
    - Authenticity: ChaCha20-Poly1305 provides authenticated encryption
    - Forward secrecy: Each commitment uses a fresh ephemeral key
    - No key reuse: Miners don't need long-term encryption keys
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import x25519
from cryptography.hazmat.primitives.ciphers.aead import ChaCha20Poly1305

# Encrypted blob structure sizes
EPHEMERAL_PUBLIC_KEY_SIZE = 32  # X25519 public key
NONCE_SIZE = 12  # ChaCha20-Poly1305 nonce (derived from ephemeral key, not stored)
AUTH_TAG_SIZE = 16  # Poly1305 tag
UUID_BYTES_SIZE = 16  # UUID as raw bytes (no dashes)

# Package: ephemeral_public (32) + ciphertext (16) + tag (16) = 64 bytes
# Nonce is derived from SHA256(ephemeral_public_key)[:12], not stored
ENCRYPTED_PACKAGE_SIZE = EPHEMERAL_PUBLIC_KEY_SIZE + UUID_BYTES_SIZE + AUTH_TAG_SIZE


@dataclass
class BackendKeypair:
    """Backend operator's X25519 keypair for decrypting miner endpoints."""

    private_key: x25519.X25519PrivateKey
    public_key: x25519.X25519PublicKey

    @classmethod
    def generate(cls) -> BackendKeypair:
        """Generate a new keypair."""
        private_key = x25519.X25519PrivateKey.generate()
        return cls(private_key=private_key, public_key=private_key.public_key())

    @classmethod
    def from_private_key_hex(cls, hex_string: str) -> BackendKeypair:
        """Load keypair from hex-encoded private key."""
        private_bytes = bytes.fromhex(hex_string)
        private_key = x25519.X25519PrivateKey.from_private_bytes(private_bytes)
        return cls(private_key=private_key, public_key=private_key.public_key())

    @classmethod
    def from_private_key_file(cls, path: str | Path) -> BackendKeypair:
        """Load keypair from a file containing hex-encoded private key."""
        path = Path(path)
        hex_string = path.read_text().strip()
        return cls.from_private_key_hex(hex_string)

    def private_key_hex(self) -> str:
        """Export private key as hex string."""
        private_bytes = self.private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )
        return private_bytes.hex()

    def public_key_hex(self) -> str:
        """Export public key as hex string."""
        public_bytes = self.public_key.public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        return public_bytes.hex()

    def save_private_key(self, path: str | Path) -> None:
        """Save private key to file (hex-encoded)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.private_key_hex())
        # Set restrictive permissions
        path.chmod(0o600)

    def save_public_key(self, path: str | Path) -> None:
        """Save public key to file (hex-encoded)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.public_key_hex())


def load_public_key(hex_string: str) -> x25519.X25519PublicKey:
    """Load a public key from hex string."""
    public_bytes = bytes.fromhex(hex_string)
    if len(public_bytes) != EPHEMERAL_PUBLIC_KEY_SIZE:
        raise ValueError(
            f"Invalid public key length: {len(public_bytes)} (expected {EPHEMERAL_PUBLIC_KEY_SIZE})"
        )
    return x25519.X25519PublicKey.from_public_bytes(public_bytes)


def uuid_to_bytes(uuid_str: str) -> bytes:
    """Convert UUID string to 16 bytes (removing dashes)."""
    hex_str = uuid_str.replace("-", "")
    if len(hex_str) != 32:
        raise ValueError(f"Invalid UUID format: {uuid_str}")
    return bytes.fromhex(hex_str)


def bytes_to_uuid(data: bytes) -> str:
    """Convert 16 bytes back to UUID string format."""
    if len(data) != UUID_BYTES_SIZE:
        raise ValueError(f"Invalid UUID bytes length: {len(data)}")
    hex_str = data.hex()
    # Format: 8-4-4-4-12
    return f"{hex_str[:8]}-{hex_str[8:12]}-{hex_str[12:16]}-{hex_str[16:20]}-{hex_str[20:]}"


def _derive_nonce(ephemeral_public_bytes: bytes) -> bytes:
    """Derive a nonce from the ephemeral public key using SHA256."""
    digest = hashes.Hash(hashes.SHA256())
    digest.update(ephemeral_public_bytes)
    return digest.finalize()[:NONCE_SIZE]


def encrypt_deployment_id(
    deployment_id: str,
    backend_public_key: x25519.X25519PublicKey | str,
) -> str:
    """
    Encrypt a deployment ID for on-chain commitment.

    Args:
        deployment_id: Basilica deployment UUID (e.g., "95edf2b6-e18b-400a-8398-5573df10e5e4")
        backend_public_key: Backend operator's public key (hex string or key object)

    Returns:
        Base85-encoded encrypted blob (~80 characters)

    The encrypted blob contains (64 bytes):
        - Ephemeral public key (32 bytes): For ECDH key derivation
        - Ciphertext + tag (32 bytes): Encrypted UUID + authentication tag
        - Nonce is derived from SHA256(ephemeral_public_key)[:12]
    """
    # Parse public key if string
    if isinstance(backend_public_key, str):
        backend_public_key = load_public_key(backend_public_key)

    # Generate ephemeral keypair for this encryption
    ephemeral_private = x25519.X25519PrivateKey.generate()
    ephemeral_public = ephemeral_private.public_key()

    # Derive shared secret via ECDH
    shared_secret = ephemeral_private.exchange(backend_public_key)

    # Convert deployment ID to bytes
    plaintext = uuid_to_bytes(deployment_id)

    # Get ephemeral public key bytes
    ephemeral_public_bytes = ephemeral_public.public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )

    # Derive nonce from ephemeral public key (deterministic, saves 12 bytes)
    nonce = _derive_nonce(ephemeral_public_bytes)

    # Encrypt with ChaCha20-Poly1305
    cipher = ChaCha20Poly1305(shared_secret)
    ciphertext = cipher.encrypt(nonce, plaintext, None)

    # Package: ephemeral_public || ciphertext (no nonce - it's derived)
    package = ephemeral_public_bytes + ciphertext

    # Base85 encode for compact representation
    return base64.b85encode(package).decode("ascii")


def decrypt_deployment_id(
    encrypted_blob: str,
    backend_private_key: x25519.X25519PrivateKey,
) -> str:
    """
    Decrypt a deployment ID from an on-chain commitment.

    Args:
        encrypted_blob: Base85-encoded encrypted blob from commitment
        backend_private_key: Backend operator's private key

    Returns:
        Decrypted deployment UUID string

    Raises:
        ValueError: If decryption fails (invalid key, tampered data, etc.)
    """
    try:
        # Decode from base85
        package = base64.b85decode(encrypted_blob.encode("ascii"))
    except Exception as e:
        raise ValueError(f"Invalid base85 encoding: {e}") from e

    if len(package) != ENCRYPTED_PACKAGE_SIZE:
        raise ValueError(
            f"Invalid encrypted package size: {len(package)} (expected {ENCRYPTED_PACKAGE_SIZE})"
        )

    # Extract ephemeral public key and ciphertext
    ephemeral_public_bytes = package[:EPHEMERAL_PUBLIC_KEY_SIZE]
    ciphertext = package[EPHEMERAL_PUBLIC_KEY_SIZE:]

    # Derive nonce from ephemeral public key
    nonce = _derive_nonce(ephemeral_public_bytes)

    # Reconstruct ephemeral public key
    ephemeral_public = x25519.X25519PublicKey.from_public_bytes(ephemeral_public_bytes)

    # Derive shared secret via ECDH
    shared_secret = backend_private_key.exchange(ephemeral_public)

    # Decrypt with ChaCha20-Poly1305
    cipher = ChaCha20Poly1305(shared_secret)
    try:
        plaintext = cipher.decrypt(nonce, ciphertext, None)
    except Exception as e:
        raise ValueError(f"Decryption failed (invalid key or tampered data): {e}") from e

    # Convert bytes back to UUID string
    return bytes_to_uuid(plaintext)
