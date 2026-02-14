"""Integration tests for the metadata verification E2E flow.

Exercises: push --image, verify --deployment-id, verify --netuid --uid,
and the full push → commit → verify pipeline — all with mocked externals
(Basilica API, Docker registry, chain).
"""

from __future__ import annotations

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

from typer.testing import CliRunner

from kinitro.chain.commitments import MinerCommitment
from kinitro.cli.miner import miner_app
from kinitro.types import BlockNumber, Hotkey, MinerUID

runner = CliRunner()


# ---------------------------------------------------------------------------
# Fakes / helpers
# ---------------------------------------------------------------------------


@dataclass
class _FakeDeployment:
    name: str = "test-deploy"
    url: str = "https://test-deploy.deployments.basilica.ai"
    state: str = "Running"


@dataclass
class _FakeMetadata:
    id: str = "dep-id-1"
    instance_name: str = "test-deploy"
    image: str = "python"
    image_tag: str = "3.11-slim"
    replicas: int = 1
    state: str = "Running"
    uptime_seconds: float = 3600.0


def _make_commitment(
    uid: int = 0,
    hotkey: str = "5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
    deployment_id: str = "test-deploy",
) -> MinerCommitment:
    return MinerCommitment(
        uid=MinerUID(uid),
        hotkey=Hotkey(hotkey),
        huggingface_repo="user/policy",
        revision_sha="abc12345",
        deployment_id=deployment_id,
        docker_image="user/policy:abc12345",
        committed_block=BlockNumber(1000),
    )


# ---------------------------------------------------------------------------
# push --image
# ---------------------------------------------------------------------------


class TestPushImage:
    def test_push_image_deploys_to_basilica(self):
        """basilica_push with --image deploys without source, enrolls metadata."""
        mock_client = MagicMock()
        mock_client.deploy.return_value = _FakeDeployment()
        mock_client.enroll_metadata.return_value = None

        with patch("kinitro.cli.miner.deploy.BasilicaClient", return_value=mock_client):
            result = runner.invoke(
                miner_app,
                [
                    "push",
                    "--image",
                    "python:3.11-slim",
                    "--name",
                    "test-deploy",
                    "--api-token",
                    "fake-token",
                ],
            )

        assert result.exit_code == 0, result.output
        assert "DEPLOYMENT SUCCESSFUL" in result.output

        # Verify deploy was called without source or pip_packages
        deploy_call = mock_client.deploy.call_args
        assert deploy_call.kwargs.get("source") is None or "source" not in deploy_call.kwargs
        assert (
            deploy_call.kwargs.get("pip_packages") is None
            or "pip_packages" not in deploy_call.kwargs
        )
        assert deploy_call.kwargs["image"] == "python:3.11-slim"
        assert deploy_call.kwargs["name"] == "test-deploy"

        # Metadata enrolled
        mock_client.enroll_metadata.assert_called_once_with("test-deploy", enabled=True)

    def test_push_image_requires_name(self):
        """--image without --name should fail."""
        result = runner.invoke(
            miner_app,
            ["push", "--image", "python:3.11-slim", "--api-token", "fake-token"],
        )
        assert result.exit_code != 0
        assert "--name is required" in result.output

    def test_push_hf_mode_requires_repo(self):
        """HF mode without --repo should fail."""
        result = runner.invoke(
            miner_app,
            ["push", "--api-token", "fake-token"],
        )
        assert result.exit_code != 0
        assert "--repo is required" in result.output


# ---------------------------------------------------------------------------
# verify --deployment-id
# ---------------------------------------------------------------------------


class TestVerifyDeployment:
    def test_verify_deployment_happy_path(self):
        """verify --deployment-id returns verified for healthy deployment."""
        mock_client = MagicMock()
        mock_client.get_public_deployment_metadata.return_value = _FakeMetadata()

        with (
            patch("kinitro.executor.verification.BasilicaClient", return_value=mock_client),
            patch(
                "kinitro.executor.verification._check_image_public",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = runner.invoke(
                miner_app,
                ["verify", "--deployment-id", "test-deploy"],
            )

        assert result.exit_code == 0, result.output
        assert "VERIFIED" in result.output
        assert "Running" in result.output

    def test_verify_deployment_unhealthy(self):
        """verify --deployment-id returns failed for stopped deployment."""
        mock_client = MagicMock()
        mock_client.get_public_deployment_metadata.return_value = _FakeMetadata(state="Stopped")

        with patch("kinitro.executor.verification.BasilicaClient", return_value=mock_client):
            result = runner.invoke(
                miner_app,
                ["verify", "--deployment-id", "test-deploy"],
            )

        assert result.exit_code != 0
        assert "FAILED" in result.output
        assert "not healthy" in result.output

    def test_verify_deployment_private_image(self):
        """verify --deployment-id returns failed when image is not public."""
        mock_client = MagicMock()
        mock_client.get_public_deployment_metadata.return_value = _FakeMetadata(
            image="private.registry.io/secret", image_tag="v1"
        )

        with (
            patch("kinitro.executor.verification.BasilicaClient", return_value=mock_client),
            patch(
                "kinitro.executor.verification._check_image_public",
                new_callable=AsyncMock,
                return_value=False,
            ),
        ):
            result = runner.invoke(
                miner_app,
                ["verify", "--deployment-id", "test-deploy"],
            )

        assert result.exit_code != 0
        assert "FAILED" in result.output
        assert "not publicly pullable" in result.output


# ---------------------------------------------------------------------------
# verify --netuid --uid (chain mode)
# ---------------------------------------------------------------------------


class TestVerifyFromChain:
    def test_verify_from_chain(self):
        """verify --netuid --uid reads commitment from chain and verifies."""
        mock_client = MagicMock()
        mock_client.get_public_deployment_metadata.return_value = _FakeMetadata()

        with (
            patch(
                "kinitro.cli.miner.verify._get_hotkey_for_uid",
                new_callable=AsyncMock,
                return_value="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ),
            patch(
                "kinitro.cli.miner.verify._read_commitment_from_chain",
                new_callable=AsyncMock,
                return_value=_make_commitment(),
            ),
            patch("kinitro.executor.verification.BasilicaClient", return_value=mock_client),
            patch(
                "kinitro.executor.verification._check_image_public",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            result = runner.invoke(
                miner_app,
                ["verify", "--netuid", "1", "--uid", "5"],
            )

        assert result.exit_code == 0, result.output
        assert "VERIFIED" in result.output

    def test_verify_no_commitment_on_chain(self):
        """verify --netuid --uid fails when no commitment exists."""
        with (
            patch(
                "kinitro.cli.miner.verify._get_hotkey_for_uid",
                new_callable=AsyncMock,
                return_value="5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty",
            ),
            patch(
                "kinitro.cli.miner.verify._read_commitment_from_chain",
                new_callable=AsyncMock,
                return_value=None,
            ),
        ):
            result = runner.invoke(
                miner_app,
                ["verify", "--netuid", "1", "--uid", "5"],
            )

        assert result.exit_code != 0
        assert "No valid commitment" in result.output

    def test_verify_uid_not_found(self):
        """verify --netuid --uid fails when UID doesn't exist."""
        with patch(
            "kinitro.cli.miner.verify._get_hotkey_for_uid",
            new_callable=AsyncMock,
            return_value=None,
        ):
            result = runner.invoke(
                miner_app,
                ["verify", "--netuid", "1", "--uid", "999"],
            )

        assert result.exit_code != 0
        assert "not found" in result.output


# ---------------------------------------------------------------------------
# Full flow: push --image → commit → verify
# ---------------------------------------------------------------------------


class TestFullFlow:
    def test_full_push_then_verify(self):
        """Push image then verify the deployment (mocked externals)."""
        # Step 1: Push
        mock_basilica_client = MagicMock()
        mock_basilica_client.deploy.return_value = _FakeDeployment()
        mock_basilica_client.enroll_metadata.return_value = None

        with patch("kinitro.cli.miner.deploy.BasilicaClient", return_value=mock_basilica_client):
            push_result = runner.invoke(
                miner_app,
                [
                    "push",
                    "--image",
                    "python:3.11-slim",
                    "--name",
                    "test-deploy",
                    "--api-token",
                    "fake-token",
                ],
            )

        assert push_result.exit_code == 0, push_result.output
        assert "DEPLOYMENT SUCCESSFUL" in push_result.output

        # Step 2: Verify the same deployment
        mock_verify_client = MagicMock()
        mock_verify_client.get_public_deployment_metadata.return_value = _FakeMetadata()

        with (
            patch(
                "kinitro.executor.verification.BasilicaClient",
                return_value=mock_verify_client,
            ),
            patch(
                "kinitro.executor.verification._check_image_public",
                new_callable=AsyncMock,
                return_value=True,
            ),
        ):
            verify_result = runner.invoke(
                miner_app,
                ["verify", "--deployment-id", "test-deploy"],
            )

        assert verify_result.exit_code == 0, verify_result.output
        assert "VERIFIED" in verify_result.output


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------


class TestVerifyArgValidation:
    def test_verify_no_args(self):
        """verify with no arguments should fail."""
        result = runner.invoke(miner_app, ["verify"])
        assert result.exit_code != 0
        assert "Provide --deployment-id" in result.output

    def test_verify_netuid_without_uid_or_hotkey(self):
        """verify --netuid alone should fail."""
        result = runner.invoke(miner_app, ["verify", "--netuid", "1"])
        assert result.exit_code != 0
        assert "Provide --deployment-id" in result.output
