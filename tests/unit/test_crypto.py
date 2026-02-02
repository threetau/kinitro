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


class TestUUIDConversion:
    """Tests for UUID <-> bytes conversion."""

    def test_uuid_to_bytes_standard_format(self):
        """Standard UUID format with dashes."""
        uuid_str = "95edf2b6-e18b-400a-8398-5573df10e5e4"
        result = uuid_to_bytes(uuid_str)

        assert len(result) == 16
        assert result.hex() == "95edf2b6e18b400a83985573df10e5e4"

    def test_uuid_to_bytes_no_dashes(self):
        """UUID format without dashes (already hex)."""
        uuid_str = "95edf2b6e18b400a83985573df10e5e4"
        result = uuid_to_bytes(uuid_str)

        assert len(result) == 16
        assert result.hex() == "95edf2b6e18b400a83985573df10e5e4"

    def test_bytes_to_uuid_roundtrip(self):
        """Bytes -> UUID -> Bytes roundtrip."""
        original = "95edf2b6-e18b-400a-8398-5573df10e5e4"
        as_bytes = uuid_to_bytes(original)
        back_to_uuid = bytes_to_uuid(as_bytes)

        assert back_to_uuid == original

    def test_uuid_to_bytes_invalid_length(self):
        """Invalid UUID length should raise."""
        with pytest.raises(ValueError, match="Invalid UUID format"):
            uuid_to_bytes("abc123")

    def test_bytes_to_uuid_invalid_length(self):
        """Invalid bytes length should raise."""
        with pytest.raises(ValueError, match="Invalid UUID bytes length"):
            bytes_to_uuid(b"too short")


class TestBackendKeypair:
    """Tests for BackendKeypair class."""

    def test_generate_creates_valid_keypair(self):
        """Generate should create a valid keypair."""
        keypair = BackendKeypair.generate()

        assert keypair.private_key is not None
        assert keypair.public_key is not None

    def test_public_key_hex_length(self):
        """Public key hex should be 64 characters (32 bytes)."""
        keypair = BackendKeypair.generate()
        pub_hex = keypair.public_key_hex()

        assert len(pub_hex) == 64
        # Should be valid hex
        bytes.fromhex(pub_hex)

    def test_private_key_hex_length(self):
        """Private key hex should be 64 characters (32 bytes)."""
        keypair = BackendKeypair.generate()
        priv_hex = keypair.private_key_hex()

        assert len(priv_hex) == 64
        # Should be valid hex
        bytes.fromhex(priv_hex)

    def test_from_private_key_hex_roundtrip(self):
        """Load keypair from hex should preserve keys."""
        original = BackendKeypair.generate()
        priv_hex = original.private_key_hex()

        restored = BackendKeypair.from_private_key_hex(priv_hex)

        assert restored.public_key_hex() == original.public_key_hex()
        assert restored.private_key_hex() == original.private_key_hex()

    def test_from_private_key_file(self, tmp_path):
        """Load keypair from file."""
        keypair = BackendKeypair.generate()
        key_file = tmp_path / "test.key"
        keypair.save_private_key(key_file)

        restored = BackendKeypair.from_private_key_file(key_file)

        assert restored.public_key_hex() == keypair.public_key_hex()

    def test_save_private_key_permissions(self, tmp_path):
        """Private key file should have restricted permissions (0600)."""
        keypair = BackendKeypair.generate()
        key_file = tmp_path / "test.key"
        keypair.save_private_key(key_file)

        mode = os.stat(key_file).st_mode & 0o777
        assert mode == 0o600

    def test_save_public_key(self, tmp_path):
        """Save and read public key."""
        keypair = BackendKeypair.generate()
        pub_file = tmp_path / "test.pub"
        keypair.save_public_key(pub_file)

        content = pub_file.read_text()
        assert content == keypair.public_key_hex()


class TestLoadPublicKey:
    """Tests for load_public_key function."""

    def test_load_valid_public_key(self):
        """Load a valid public key from hex."""
        keypair = BackendKeypair.generate()
        pub_hex = keypair.public_key_hex()

        loaded = load_public_key(pub_hex)

        # Should be able to use for encryption
        assert loaded is not None

    def test_load_invalid_length(self):
        """Invalid length should raise."""
        with pytest.raises(ValueError, match="Invalid public key length"):
            load_public_key("abcd1234")


class TestEncryptDecrypt:
    """Tests for encrypt/decrypt deployment ID."""

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypt and decrypt should return original value."""
        keypair = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted = encrypt_deployment_id(deployment_id, keypair.public_key_hex())
        decrypted = decrypt_deployment_id(encrypted, keypair.private_key)

        assert decrypted == deployment_id

    def test_encrypt_with_key_object(self):
        """Encrypt should accept key object directly."""
        keypair = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted = encrypt_deployment_id(deployment_id, keypair.public_key)
        decrypted = decrypt_deployment_id(encrypted, keypair.private_key)

        assert decrypted == deployment_id

    def test_encrypted_blob_is_base85(self):
        """Encrypted blob should be base85 encoded."""
        keypair = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted = encrypt_deployment_id(deployment_id, keypair.public_key_hex())

        # Should be decodable as base85
        decoded = base64.b85decode(encrypted.encode("ascii"))
        assert len(decoded) == 64  # 32 + 16 + 16 (pubkey + ciphertext + tag, nonce derived)

    def test_encrypted_blob_length(self):
        """Encrypted blob should be ~95 characters (base85 of 76 bytes)."""
        keypair = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted = encrypt_deployment_id(deployment_id, keypair.public_key_hex())

        # Base85: 64 bytes -> ceil(64 * 5 / 4) = 80 characters
        assert len(encrypted) == 80

    def test_decrypt_with_wrong_key_fails(self):
        """Decryption with wrong key should fail."""
        keypair1 = BackendKeypair.generate()
        keypair2 = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted = encrypt_deployment_id(deployment_id, keypair1.public_key_hex())

        with pytest.raises(ValueError, match="Decryption failed"):
            decrypt_deployment_id(encrypted, keypair2.private_key)

    def test_decrypt_tampered_data_fails(self):
        """Decryption of tampered data should fail."""
        keypair = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted = encrypt_deployment_id(deployment_id, keypair.public_key_hex())

        # Tamper with the encrypted blob
        tampered = encrypted[:-5] + "XXXXX"

        with pytest.raises(ValueError):
            decrypt_deployment_id(tampered, keypair.private_key)

    def test_each_encryption_is_unique(self):
        """Each encryption should produce different output (fresh ephemeral key)."""
        keypair = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted1 = encrypt_deployment_id(deployment_id, keypair.public_key_hex())
        encrypted2 = encrypt_deployment_id(deployment_id, keypair.public_key_hex())

        # Same plaintext should produce different ciphertext (different ephemeral keys)
        assert encrypted1 != encrypted2

        # Both should decrypt to the same value
        assert decrypt_deployment_id(encrypted1, keypair.private_key) == deployment_id
        assert decrypt_deployment_id(encrypted2, keypair.private_key) == deployment_id


class TestIntegration:
    """Integration tests for the full encryption flow."""

    def test_encrypt_decrypt_deployment_id(self):
        """Test full flow: generate keys, encrypt, decrypt."""
        # Backend generates keypair
        backend_keypair = BackendKeypair.generate()

        # Encrypt a deployment ID
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"
        encrypted_blob = encrypt_deployment_id(deployment_id, backend_keypair.public_key_hex())

        # Encrypted blob should be under 100 chars (for potential future use in commitments)
        assert len(encrypted_blob) <= 100

        # Decrypt the deployment ID
        decrypted = decrypt_deployment_id(encrypted_blob, backend_keypair.private_key)

        assert decrypted == deployment_id

    def test_commitment_format(self):
        """Test the new simplified commitment format."""
        commitment = "user/policy:abc123de"

        parsed = parse_commitment(commitment)

        assert parsed["huggingface_repo"] == "user/policy"
        assert parsed["revision_sha"] == "abc123de"

    def test_legacy_commitment_format_backward_compat(self):
        """Legacy commitments with deployment_id should still parse (backward compat)."""
        # Old format: repo:rev:deployment_id
        commitment = "user/policy:abc123de:95edf2b6-e18b-400a-8398-5573df10e5e4"

        parsed = parse_commitment(commitment)

        # We extract repo and revision, ignoring the deployment_id
        assert parsed["huggingface_repo"] == "user/policy"
        assert parsed["revision_sha"] == "abc123de"
