"""Tests for encrypted endpoint commitments (crypto module)."""

import base64
import os

import pytest

from kinitro.chain.commitments import parse_commitment
from kinitro.crypto import (
    BackendKeypair,
    bytes_to_uuid,
    decrypt_deployment_id,
    encrypt_deployment_id,
    load_public_key,
    uuid_to_bytes,
)


@pytest.fixture()
def keypair() -> BackendKeypair:
    return BackendKeypair.generate()


class TestUUIDConversion:
    """Tests for UUID <-> bytes conversion."""

    @pytest.mark.parametrize(
        "uuid_str",
        [
            pytest.param("95edf2b6-e18b-400a-8398-5573df10e5e4", id="with_dashes"),
            pytest.param("95edf2b6e18b400a83985573df10e5e4", id="no_dashes"),
        ],
    )
    def test_uuid_to_bytes(self, uuid_str: str) -> None:
        """Both UUID formats should produce the same 16-byte result."""
        result = uuid_to_bytes(uuid_str)

        assert len(result) == 16
        assert result.hex() == "95edf2b6e18b400a83985573df10e5e4"

    def test_bytes_to_uuid_roundtrip(self) -> None:
        """Bytes -> UUID -> Bytes roundtrip."""
        original = "95edf2b6-e18b-400a-8398-5573df10e5e4"
        as_bytes = uuid_to_bytes(original)
        back_to_uuid = bytes_to_uuid(as_bytes)

        assert back_to_uuid == original

    def test_uuid_to_bytes_invalid_length(self) -> None:
        """Invalid UUID length should raise."""
        with pytest.raises(ValueError, match="Invalid UUID format"):
            uuid_to_bytes("abc123")

    def test_bytes_to_uuid_invalid_length(self) -> None:
        """Invalid bytes length should raise."""
        with pytest.raises(ValueError, match="Invalid UUID bytes length"):
            bytes_to_uuid(b"too short")


class TestBackendKeypair:
    """Tests for BackendKeypair class."""

    def test_generate_creates_valid_keypair(self, keypair: BackendKeypair) -> None:
        """Generate should create a valid keypair."""
        assert keypair.private_key is not None
        assert keypair.public_key is not None

    def test_public_key_hex_length(self, keypair: BackendKeypair) -> None:
        """Public key hex should be 64 characters (32 bytes)."""
        pub_hex = keypair.public_key_hex()

        assert len(pub_hex) == 64
        # Should be valid hex
        bytes.fromhex(pub_hex)

    def test_private_key_hex_length(self, keypair: BackendKeypair) -> None:
        """Private key hex should be 64 characters (32 bytes)."""
        priv_hex = keypair.private_key_hex()

        assert len(priv_hex) == 64
        # Should be valid hex
        bytes.fromhex(priv_hex)

    def test_from_private_key_hex_roundtrip(self, keypair: BackendKeypair) -> None:
        """Load keypair from hex should preserve keys."""
        priv_hex = keypair.private_key_hex()

        restored = BackendKeypair.from_private_key_hex(priv_hex)

        assert restored.public_key_hex() == keypair.public_key_hex()
        assert restored.private_key_hex() == keypair.private_key_hex()

    def test_from_private_key_file(self, keypair: BackendKeypair, tmp_path) -> None:
        """Load keypair from file."""
        key_file = tmp_path / "test.key"
        keypair.save_private_key(key_file)

        restored = BackendKeypair.from_private_key_file(key_file)

        assert restored.public_key_hex() == keypair.public_key_hex()

    def test_save_private_key_permissions(self, keypair: BackendKeypair, tmp_path) -> None:
        """Private key file should have restricted permissions (0600)."""
        key_file = tmp_path / "test.key"
        keypair.save_private_key(key_file)

        mode = os.stat(key_file).st_mode & 0o777
        assert mode == 0o600

    def test_save_public_key(self, keypair: BackendKeypair, tmp_path) -> None:
        """Save and read public key."""
        pub_file = tmp_path / "test.pub"
        keypair.save_public_key(pub_file)

        content = pub_file.read_text()
        assert content == keypair.public_key_hex()


class TestLoadPublicKey:
    """Tests for load_public_key function."""

    def test_load_valid_public_key(self) -> None:
        """Load a valid public key from hex."""
        pub_hex = BackendKeypair.generate().public_key_hex()

        loaded = load_public_key(pub_hex)

        # Should be able to use for encryption
        assert loaded is not None

    def test_load_invalid_length(self) -> None:
        """Invalid length should raise."""
        with pytest.raises(ValueError, match="Invalid public key length"):
            load_public_key("abcd1234")


class TestEncryptDecrypt:
    """Tests for encrypt/decrypt deployment ID."""

    DEPLOYMENT_ID = "95edf2b6-e18b-400a-8398-5573df10e5e4"

    @pytest.mark.parametrize(
        "use_key_object",
        [
            pytest.param(False, id="hex_string"),
            pytest.param(True, id="key_object"),
        ],
    )
    def test_encrypt_decrypt_roundtrip(self, keypair: BackendKeypair, use_key_object) -> None:
        """Encrypt and decrypt should return original value (hex string or key object)."""
        pub_key = keypair.public_key if use_key_object else keypair.public_key_hex()

        encrypted = encrypt_deployment_id(self.DEPLOYMENT_ID, pub_key)
        decrypted = decrypt_deployment_id(encrypted, keypair.private_key)

        assert decrypted == self.DEPLOYMENT_ID

    def test_encrypted_blob_is_base85(self, keypair: BackendKeypair) -> None:
        """Encrypted blob should be base85 encoded."""
        encrypted = encrypt_deployment_id(self.DEPLOYMENT_ID, keypair.public_key_hex())

        # Should be decodable as base85
        decoded = base64.b85decode(encrypted.encode("ascii"))
        assert len(decoded) == 64  # 32 + 16 + 16 (pubkey + ciphertext + tag, nonce derived)

    def test_encrypted_blob_length(self, keypair: BackendKeypair) -> None:
        """Encrypted blob should be ~95 characters (base85 of 76 bytes)."""
        encrypted = encrypt_deployment_id(self.DEPLOYMENT_ID, keypair.public_key_hex())

        # Base85: 64 bytes -> ceil(64 * 5 / 4) = 80 characters
        assert len(encrypted) == 80

    def test_decrypt_with_wrong_key_fails(self, keypair: BackendKeypair) -> None:
        """Decryption with wrong key should fail."""
        keypair2 = BackendKeypair.generate()

        encrypted = encrypt_deployment_id(self.DEPLOYMENT_ID, keypair.public_key_hex())

        with pytest.raises(ValueError, match="Decryption failed"):
            decrypt_deployment_id(encrypted, keypair2.private_key)

    def test_decrypt_tampered_data_fails(self, keypair: BackendKeypair) -> None:
        """Decryption of tampered data should fail."""
        encrypted = encrypt_deployment_id(self.DEPLOYMENT_ID, keypair.public_key_hex())

        # Tamper with the encrypted blob
        tampered = encrypted[:-5] + "XXXXX"

        with pytest.raises(ValueError):
            decrypt_deployment_id(tampered, keypair.private_key)

    def test_each_encryption_is_unique(self, keypair: BackendKeypair) -> None:
        """Each encryption should produce different output (fresh ephemeral key)."""
        encrypted1 = encrypt_deployment_id(self.DEPLOYMENT_ID, keypair.public_key_hex())
        encrypted2 = encrypt_deployment_id(self.DEPLOYMENT_ID, keypair.public_key_hex())

        # Same plaintext should produce different ciphertext (different ephemeral keys)
        assert encrypted1 != encrypted2

        # Both should decrypt to the same value
        assert decrypt_deployment_id(encrypted1, keypair.private_key) == self.DEPLOYMENT_ID
        assert decrypt_deployment_id(encrypted2, keypair.private_key) == self.DEPLOYMENT_ID


class TestIntegration:
    """Integration tests for the full encryption flow."""

    def test_full_commitment_flow(self, keypair: BackendKeypair) -> None:
        """Test full flow: generate keys, encrypt, parse, decrypt."""
        # Miner encrypts their deployment ID
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"
        encrypted_blob = encrypt_deployment_id(deployment_id, keypair.public_key_hex())

        # Miner creates commitment (colon-separated format)
        commitment = f"user/policy:abc123def456:e:{encrypted_blob}"

        # Verify commitment is under chain limit (128 bytes)
        assert len(commitment) <= 128

        # Backend parses commitment from chain
        parsed = parse_commitment(commitment)

        assert parsed["huggingface_repo"] == "user/policy"
        assert parsed["revision_sha"] == "abc123def456"
        assert parsed["deployment_id"] == ""  # Empty until decrypted
        assert parsed["encrypted_deployment"] == encrypted_blob

        # Backend decrypts the endpoint
        assert parsed["encrypted_deployment"] is not None
        decrypted = decrypt_deployment_id(parsed["encrypted_deployment"], keypair.private_key)

        assert decrypted == deployment_id

    def test_plain_commitment_still_works(self) -> None:
        """Plain commitments should still work."""
        commitment = "user/policy:abc123:95edf2b6-e18b-400a-8398-5573df10e5e4"

        parsed = parse_commitment(commitment)

        assert parsed["huggingface_repo"] == "user/policy"
        assert parsed["revision_sha"] == "abc123"
        assert parsed["deployment_id"] == "95edf2b6-e18b-400a-8398-5573df10e5e4"
        assert parsed["encrypted_deployment"] is None
