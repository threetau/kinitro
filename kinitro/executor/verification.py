"""Deployment metadata verification for miner Basilica deployments.

Uses the Basilica public metadata API to verify that miner deployments
are running with publicly pullable Docker images. This replaces the previous
spot-check system that required downloading full HuggingFace repos.

Verification checks:
1. Deployment exists and metadata is publicly accessible
2. Deployment is in a healthy state (Active/Running)
3. Docker image is publicly pullable from its container registry
"""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass

import httpx
import structlog
from basilica import BasilicaClient

from kinitro.chain.commitments import MinerCommitment
from kinitro.types import Hotkey, MinerUID

logger = structlog.get_logger()

# Deployment states considered healthy
HEALTHY_STATES: frozenset[str] = frozenset({"Active", "Running"})

# Docker Hub registry host used when no registry is specified in the image ref
DOCKER_HUB_REGISTRY = "registry-1.docker.io"
DOCKER_HUB_AUTH_URL = "https://auth.docker.io/token"

# Accept header for Docker Registry HTTP V2 manifest requests
_MANIFEST_ACCEPT = (
    "application/vnd.docker.distribution.manifest.v2+json, "
    "application/vnd.oci.image.manifest.v1+json"
)


@dataclass
class ImageRef:
    """Parsed container image reference."""

    registry: str  # e.g. "registry-1.docker.io", "ghcr.io"
    repository: str  # e.g. "library/python", "pytorch/pytorch"
    tag: str  # e.g. "3.11-slim", "latest"


@dataclass
class MetadataVerificationResult:
    """Result of a metadata-based deployment verification."""

    miner_uid: MinerUID
    miner_hotkey: Hotkey
    deployment_id: str
    verified: bool
    state: str | None = None
    image: str | None = None
    image_tag: str | None = None
    image_public: bool | None = None
    uptime_seconds: float | None = None
    error: str | None = None
    failure_reason: str | None = None


def parse_image_ref(image: str, image_tag: str | None = None) -> ImageRef:
    """Parse a Docker image reference into registry, repository, and tag.

    Handles:
    - Docker Hub shorthand: ``python:3.11-slim`` → registry-1.docker.io/library/python:3.11-slim
    - Docker Hub org: ``pytorch/pytorch:2.1.0`` → registry-1.docker.io/pytorch/pytorch:2.1.0
    - Fully qualified: ``ghcr.io/org/image:v1`` → ghcr.io/org/image:v1
    """
    # If image_tag provided separately, strip any tag already on the image name
    if image_tag:
        name = image.split(":")[0]
        tag = image_tag
    elif ":" in image:
        name, tag = image.rsplit(":", 1)
    else:
        name = image
        tag = "latest"

    # Determine if the first component is a registry host.
    # Registry hosts contain a dot or a colon (port), or are "localhost".
    parts = name.split("/", 1)
    if len(parts) == 1:
        # Simple name like "python" → Docker Hub library image
        return ImageRef(
            registry=DOCKER_HUB_REGISTRY,
            repository=f"library/{name}",
            tag=tag,
        )

    first = parts[0]
    has_dot = "." in first
    has_colon = ":" in first
    is_localhost = first == "localhost"

    if has_dot or has_colon or is_localhost:
        # Fully qualified: ghcr.io/org/image or localhost:5000/img
        return ImageRef(registry=first, repository=parts[1], tag=tag)

    # No registry prefix → Docker Hub with org, e.g. "pytorch/pytorch"
    return ImageRef(registry=DOCKER_HUB_REGISTRY, repository=name, tag=tag)


class MetadataVerifier:
    """Verifies miner Basilica deployments using the public metadata API.

    Checks:
    1. Deployment metadata is accessible (miner enrolled for public metadata)
    2. Deployment is in a healthy state (Active/Running)
    3. Docker image is publicly pullable from its container registry
    """

    def __init__(self) -> None:
        # No API key needed for public metadata reads
        self._client = BasilicaClient()

    async def verify_miner(self, commitment: MinerCommitment) -> MetadataVerificationResult:
        """Verify a single miner's deployment via metadata API."""
        logger.info(
            "metadata_verification_starting",
            miner_uid=commitment.uid,
            deployment_id=commitment.deployment_id,
        )

        try:
            metadata = await asyncio.to_thread(
                self._client.get_public_deployment_metadata,
                commitment.deployment_id,
            )
        except Exception as e:
            logger.warning(
                "metadata_api_error",
                miner_uid=commitment.uid,
                deployment_id=commitment.deployment_id,
                error=str(e),
            )
            return MetadataVerificationResult(
                miner_uid=commitment.uid,
                miner_hotkey=commitment.hotkey,
                deployment_id=commitment.deployment_id,
                verified=False,
                error=str(e),
                failure_reason="Metadata API call failed (deployment may not have public metadata enrolled)",
            )

        state = metadata.state
        image = metadata.image
        image_tag = metadata.image_tag

        # Check deployment state
        if state not in HEALTHY_STATES:
            return MetadataVerificationResult(
                miner_uid=commitment.uid,
                miner_hotkey=commitment.hotkey,
                deployment_id=commitment.deployment_id,
                verified=False,
                state=state,
                image=image,
                image_tag=image_tag,
                uptime_seconds=metadata.uptime_seconds,
                failure_reason=f"Deployment state '{state}' is not healthy",
            )

        # Check image is publicly pullable
        if not image:
            return MetadataVerificationResult(
                miner_uid=commitment.uid,
                miner_hotkey=commitment.hotkey,
                deployment_id=commitment.deployment_id,
                verified=False,
                state=state,
                image_public=False,
                uptime_seconds=metadata.uptime_seconds,
                failure_reason="No image reported in deployment metadata",
            )

        image_public = await _check_image_public(image, image_tag)

        if not image_public:
            full_ref = f"{image}:{image_tag}" if image_tag else image
            return MetadataVerificationResult(
                miner_uid=commitment.uid,
                miner_hotkey=commitment.hotkey,
                deployment_id=commitment.deployment_id,
                verified=False,
                state=state,
                image=image,
                image_tag=image_tag,
                image_public=False,
                uptime_seconds=metadata.uptime_seconds,
                failure_reason=f"Image '{full_ref}' is not publicly pullable",
            )

        # All checks passed
        return MetadataVerificationResult(
            miner_uid=commitment.uid,
            miner_hotkey=commitment.hotkey,
            deployment_id=commitment.deployment_id,
            verified=True,
            state=state,
            image=image,
            image_tag=image_tag,
            image_public=True,
            uptime_seconds=metadata.uptime_seconds,
        )

    async def verify_miners(
        self, commitments: list[MinerCommitment]
    ) -> list[MetadataVerificationResult]:
        """Verify all miners concurrently."""
        tasks = [self.verify_miner(c) for c in commitments]
        return await asyncio.gather(*tasks)


async def _check_image_public(image: str, image_tag: str | None = None) -> bool:
    """Check whether a container image is publicly pullable.

    Queries the Docker Registry HTTP V2 API to verify the manifest exists
    and is accessible without credentials.
    """
    ref = parse_image_ref(image, image_tag)

    try:
        async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
            if ref.registry == DOCKER_HUB_REGISTRY:
                return await _check_docker_hub(client, ref)
            return await _check_generic_registry(client, ref)
    except Exception as e:
        logger.debug("image_public_check_error", image=image, error=str(e))
        return False


async def _check_docker_hub(client: httpx.AsyncClient, ref: ImageRef) -> bool:
    """Check image pullability on Docker Hub (requires anonymous token)."""
    # Get anonymous bearer token
    token_resp = await client.get(
        DOCKER_HUB_AUTH_URL,
        params={
            "service": "registry.docker.io",
            "scope": f"repository:{ref.repository}:pull",
        },
    )
    if token_resp.status_code != 200:
        return False

    token = token_resp.json().get("token")
    if not token:
        return False

    # Check manifest
    manifest_url = f"https://{ref.registry}/v2/{ref.repository}/manifests/{ref.tag}"
    resp = await client.head(
        manifest_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": _MANIFEST_ACCEPT,
        },
    )
    return resp.status_code == 200


async def _check_generic_registry(client: httpx.AsyncClient, ref: ImageRef) -> bool:
    """Check image pullability on a generic OCI registry."""
    manifest_url = f"https://{ref.registry}/v2/{ref.repository}/manifests/{ref.tag}"

    # Try unauthenticated first
    resp = await client.head(
        manifest_url,
        headers={"Accept": _MANIFEST_ACCEPT},
    )

    if resp.status_code == 200:
        return True

    # If 401 with Www-Authenticate, try anonymous token exchange
    if resp.status_code == 401:
        return await _try_anonymous_token(client, resp, ref)

    return False


async def _try_anonymous_token(
    client: httpx.AsyncClient,
    unauthorized_resp: httpx.Response,
    ref: ImageRef,
) -> bool:
    """Attempt anonymous token exchange from a 401 Www-Authenticate header."""
    www_auth = unauthorized_resp.headers.get("www-authenticate", "")
    if not www_auth:
        return False

    # Parse Bearer realm="...",service="...",scope="..."
    realm_match = re.search(r'realm="([^"]+)"', www_auth)
    service_match = re.search(r'service="([^"]+)"', www_auth)

    if not realm_match:
        return False

    realm = realm_match.group(1)
    params: dict[str, str] = {}
    if service_match:
        params["service"] = service_match.group(1)
    params["scope"] = f"repository:{ref.repository}:pull"

    token_resp = await client.get(realm, params=params)
    if token_resp.status_code != 200:
        return False

    token = token_resp.json().get("token") or token_resp.json().get("access_token")
    if not token:
        return False

    manifest_url = f"https://{ref.registry}/v2/{ref.repository}/manifests/{ref.tag}"
    resp = await client.head(
        manifest_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": _MANIFEST_ACCEPT,
        },
    )
    return resp.status_code == 200
