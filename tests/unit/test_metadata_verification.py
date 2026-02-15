"""Tests for the Basilica metadata-based deployment verification."""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from kinitro.chain.commitments import MinerCommitment
from kinitro.executor.verification import (
    DOCKER_HUB_REGISTRY,
    MetadataVerifier,
    _check_image_public,
    parse_image_ref,
)
from kinitro.types import BlockNumber, Hotkey, MinerUID

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_commitment(
    uid: int = 1,
    hotkey: str = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    deployment_id: str = "test-deployment-abc123",
) -> MinerCommitment:
    return MinerCommitment(
        uid=MinerUID(uid),
        hotkey=Hotkey(hotkey),
        deployment_id=deployment_id,
        committed_block=BlockNumber(1000),
    )


@dataclass
class _FakeMetadata:
    id: str = "dep-id-1"
    instance_name: str = "test-deployment-abc123"
    image: str = "python"
    image_tag: str = "3.11-slim"
    replicas: int = 1
    state: str = "Running"
    uptime_seconds: float = 3600.0


# ---------------------------------------------------------------------------
# parse_image_ref
# ---------------------------------------------------------------------------


class TestParseImageRef:
    def test_docker_hub_library_image(self):
        ref = parse_image_ref("python", "3.11-slim")
        assert ref.registry == DOCKER_HUB_REGISTRY
        assert ref.repository == "library/python"
        assert ref.tag == "3.11-slim"

    def test_docker_hub_library_image_inline_tag(self):
        ref = parse_image_ref("python:3.11-slim")
        assert ref.registry == DOCKER_HUB_REGISTRY
        assert ref.repository == "library/python"
        assert ref.tag == "3.11-slim"

    def test_docker_hub_org_image(self):
        ref = parse_image_ref("pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime")
        assert ref.registry == DOCKER_HUB_REGISTRY
        assert ref.repository == "pytorch/pytorch"
        assert ref.tag == "2.1.0-cuda12.1-cudnn8-runtime"

    def test_fully_qualified_ghcr(self):
        ref = parse_image_ref("ghcr.io/org/image:v1")
        assert ref.registry == "ghcr.io"
        assert ref.repository == "org/image"
        assert ref.tag == "v1"

    def test_fully_qualified_nested(self):
        ref = parse_image_ref("nvcr.io/nvidia/pytorch:23.10-py3")
        assert ref.registry == "nvcr.io"
        assert ref.repository == "nvidia/pytorch"
        assert ref.tag == "23.10-py3"

    def test_no_tag_defaults_to_latest(self):
        ref = parse_image_ref("python")
        assert ref.tag == "latest"

    def test_separate_tag_overrides_inline(self):
        ref = parse_image_ref("python:3.10", "3.11-slim")
        assert ref.tag == "3.11-slim"

    def test_localhost_registry(self):
        ref = parse_image_ref("localhost:5000/myimage:v1")
        assert ref.registry == "localhost:5000"
        assert ref.repository == "myimage"
        assert ref.tag == "v1"


# ---------------------------------------------------------------------------
# MetadataVerifier.verify_miner
# ---------------------------------------------------------------------------


class TestVerifyMiner:
    @pytest.mark.asyncio
    async def test_happy_path(self):
        """Healthy deployment with public image passes verification."""
        verifier = MetadataVerifier.__new__(MetadataVerifier)
        verifier._client = MagicMock()
        verifier._client.get_public_deployment_metadata = MagicMock(return_value=_FakeMetadata())

        with patch(
            "kinitro.executor.verification._check_image_public",
            new_callable=AsyncMock,
            return_value=True,
        ):
            result = await verifier.verify_miner(_make_commitment())

        assert result.verified is True
        assert result.state == "Running"
        assert result.image == "python"
        assert result.image_tag == "3.11-slim"
        assert result.image_public is True

    @pytest.mark.asyncio
    async def test_unhealthy_state(self):
        """Deployment in non-healthy state fails verification."""
        verifier = MetadataVerifier.__new__(MetadataVerifier)
        verifier._client = MagicMock()
        verifier._client.get_public_deployment_metadata = MagicMock(
            return_value=_FakeMetadata(state="Stopped")
        )

        result = await verifier.verify_miner(_make_commitment())

        assert result.verified is False
        assert result.state == "Stopped"
        assert "not healthy" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_private_image(self):
        """Non-public image fails verification."""
        verifier = MetadataVerifier.__new__(MetadataVerifier)
        verifier._client = MagicMock()
        verifier._client.get_public_deployment_metadata = MagicMock(
            return_value=_FakeMetadata(image="private-registry.corp/secret-img", image_tag="v1")
        )

        with patch(
            "kinitro.executor.verification._check_image_public",
            new_callable=AsyncMock,
            return_value=False,
        ):
            result = await verifier.verify_miner(_make_commitment())

        assert result.verified is False
        assert result.image_public is False
        assert "not publicly pullable" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_no_image_in_metadata(self):
        """Missing image field fails verification."""
        verifier = MetadataVerifier.__new__(MetadataVerifier)
        verifier._client = MagicMock()
        verifier._client.get_public_deployment_metadata = MagicMock(
            return_value=_FakeMetadata(image="")
        )

        result = await verifier.verify_miner(_make_commitment())

        assert result.verified is False
        assert "No image" in (result.failure_reason or "")

    @pytest.mark.asyncio
    async def test_metadata_api_error(self):
        """Exception from SDK fails verification gracefully."""
        verifier = MetadataVerifier.__new__(MetadataVerifier)
        verifier._client = MagicMock()
        verifier._client.get_public_deployment_metadata = MagicMock(
            side_effect=RuntimeError("API unavailable")
        )

        result = await verifier.verify_miner(_make_commitment())

        assert result.verified is False
        assert result.error == "API unavailable"
        assert "Metadata API call failed" in (result.failure_reason or "")


# ---------------------------------------------------------------------------
# MetadataVerifier.verify_miners (batch)
# ---------------------------------------------------------------------------


class TestVerifyMiners:
    @pytest.mark.asyncio
    async def test_batch_mixed_results(self):
        """Batch with a mix of passing and failing miners."""
        verifier = MetadataVerifier.__new__(MetadataVerifier)
        verifier._client = MagicMock()

        def _fake_metadata(instance_name):
            if instance_name == "good-deploy":
                return _FakeMetadata(instance_name="good-deploy", state="Running")
            return _FakeMetadata(instance_name="bad-deploy", state="Failed")

        verifier._client.get_public_deployment_metadata = MagicMock(side_effect=_fake_metadata)

        commitments = [
            _make_commitment(uid=1, deployment_id="good-deploy"),
            _make_commitment(uid=2, deployment_id="bad-deploy"),
        ]

        with patch(
            "kinitro.executor.verification._check_image_public",
            new_callable=AsyncMock,
            return_value=True,
        ):
            results = await verifier.verify_miners(commitments)

        assert len(results) == 2
        assert results[0].verified is True
        assert results[1].verified is False


# ---------------------------------------------------------------------------
# _check_image_public (integration-style with mocked HTTP)
# ---------------------------------------------------------------------------


class TestCheckImagePublic:
    @pytest.mark.asyncio
    async def test_docker_hub_public_image(self, monkeypatch):
        """Docker Hub image that is publicly accessible."""

        async def _mock_get(self, url, **kwargs):
            if "auth.docker.io" in str(url):
                return httpx.Response(200, json={"token": "fake-token"})
            return httpx.Response(200)

        async def _mock_head(self, url, **kwargs):
            return httpx.Response(200)

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)
        monkeypatch.setattr(httpx.AsyncClient, "head", _mock_head)

        result = await _check_image_public("python", "3.11-slim")
        assert result is True

    @pytest.mark.asyncio
    async def test_docker_hub_token_failure(self, monkeypatch):
        """Docker Hub token request fails."""

        async def _mock_get(self, url, **kwargs):
            return httpx.Response(403)

        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)

        result = await _check_image_public("python", "3.11-slim")
        assert result is False

    @pytest.mark.asyncio
    async def test_generic_registry_public(self, monkeypatch):
        """Generic registry image accessible without auth."""

        async def _mock_head(self, url, **kwargs):
            return httpx.Response(200)

        monkeypatch.setattr(httpx.AsyncClient, "head", _mock_head)

        result = await _check_image_public("ghcr.io/org/image", "v1")
        assert result is True

    @pytest.mark.asyncio
    async def test_network_error_returns_false(self, monkeypatch):
        """Network error during registry check returns False."""

        async def _mock_head(self, url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        async def _mock_get(self, url, **kwargs):
            raise httpx.ConnectError("Connection refused")

        monkeypatch.setattr(httpx.AsyncClient, "head", _mock_head)
        monkeypatch.setattr(httpx.AsyncClient, "get", _mock_get)

        result = await _check_image_public("python", "3.11-slim")
        assert result is False
