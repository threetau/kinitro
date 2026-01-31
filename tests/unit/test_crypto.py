"""Tests for encrypted endpoint commitments (crypto module)."""

import pytest

from kinitro.crypto import (
    BackendKeypair,
    bytes_to_uuid,
    decrypt_deployment_id,
    encrypt_deployment_id,
    get_encrypted_blob,
    is_encrypted_commitment,
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
        import os

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
        import base64

        keypair = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted = encrypt_deployment_id(deployment_id, keypair.public_key_hex())

        # Should be decodable as base85
        decoded = base64.b85decode(encrypted.encode("ascii"))
        assert len(decoded) == 76  # 32 + 12 + 32 (pubkey + nonce + ciphertext+tag)

    def test_encrypted_blob_length(self):
        """Encrypted blob should be ~95 characters (base85 of 76 bytes)."""
        keypair = BackendKeypair.generate()
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"

        encrypted = encrypt_deployment_id(deployment_id, keypair.public_key_hex())

        # Base85: 76 bytes -> ceil(76 * 5 / 4) = 95 characters
        assert len(encrypted) == 95

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


class TestCommitmentHelpers:
    """Tests for commitment parsing helpers."""

    def test_is_encrypted_commitment_with_e_key(self):
        """Commitment with 'e' key is encrypted."""
        data = {"m": "user/repo", "r": "abc123", "e": "encrypted_blob"}
        assert is_encrypted_commitment(data) is True

    def test_is_encrypted_commitment_with_full_key(self):
        """Commitment with 'encrypted_endpoint' key is encrypted."""
        data = {"model": "user/repo", "revision": "abc123", "encrypted_endpoint": "blob"}
        assert is_encrypted_commitment(data) is True

    def test_is_encrypted_commitment_plain(self):
        """Plain commitment without encryption key."""
        data = {"m": "user/repo", "r": "abc123", "d": "uuid-here"}
        assert is_encrypted_commitment(data) is False

    def test_get_encrypted_blob_short_key(self):
        """Get encrypted blob with short key 'e'."""
        data = {"m": "user/repo", "r": "abc123", "e": "encrypted_blob_here"}
        assert get_encrypted_blob(data) == "encrypted_blob_here"

    def test_get_encrypted_blob_full_key(self):
        """Get encrypted blob with full key."""
        data = {"model": "user/repo", "encrypted_endpoint": "full_blob"}
        assert get_encrypted_blob(data) == "full_blob"

    def test_get_encrypted_blob_none_for_plain(self):
        """Get encrypted blob returns None for plain commitment."""
        data = {"m": "user/repo", "r": "abc123", "d": "uuid"}
        assert get_encrypted_blob(data) is None


class TestIntegration:
    """Integration tests for the full encryption flow."""

    def test_full_commitment_flow(self):
        """Test full flow: generate keys, encrypt, parse, decrypt."""
        from kinitro.chain.commitments import parse_commitment

        # Backend generates keypair
        backend_keypair = BackendKeypair.generate()

        # Miner encrypts their deployment ID
        deployment_id = "95edf2b6-e18b-400a-8398-5573df10e5e4"
        encrypted_blob = encrypt_deployment_id(
            deployment_id, backend_keypair.public_key_hex()
        )

        # Miner creates commitment (this is what goes on-chain)
        import json

        commitment_json = json.dumps(
            {"m": "user/policy", "r": "abc123def456", "e": encrypted_blob},
            separators=(",", ":"),
        )

        # Verify commitment is under chain limit (~128 bytes)
        assert len(commitment_json) < 150

        # Backend parses commitment from chain
        parsed = parse_commitment(commitment_json)

        assert parsed["huggingface_repo"] == "user/policy"
        assert parsed["revision_sha"] == "abc123def456"
        assert parsed["deployment_id"] == ""  # Empty until decrypted
        assert parsed["encrypted_deployment"] == encrypted_blob

        # Backend decrypts the endpoint
        decrypted = decrypt_deployment_id(
            parsed["encrypted_deployment"], backend_keypair.private_key
        )

        assert decrypted == deployment_id

    def test_plain_commitment_still_works(self):
        """Plain commitments should still work."""
        from kinitro.chain.commitments import parse_commitment

        commitment_json = '{"m":"user/policy","r":"abc123","d":"95edf2b6-e18b-400a-8398-5573df10e5e4"}'

        parsed = parse_commitment(commitment_json)

        assert parsed["huggingface_repo"] == "user/policy"
        assert parsed["revision_sha"] == "abc123"
        assert parsed["deployment_id"] == "95edf2b6-e18b-400a-8398-5573df10e5e4"
        assert parsed["encrypted_deployment"] is None
