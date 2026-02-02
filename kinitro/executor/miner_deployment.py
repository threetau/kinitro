"""Miner deployment management via affinetes and Basilica SDK.

This module manages miner policy deployments using:
- Docker mode: affinetes for local container management
- Basilica mode: Basilica SDK directly for persistent cloud deployments

The executor downloads miner policies from HuggingFace and runs them as
ephemeral deployments with TTL-based caching.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import affinetes as af
import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class MinerDeploymentConfig:
    """Configuration for miner deployments."""

    # Affinetes settings
    image: str = "kinitro/miner-runner:v1"
    mode: str = "docker"  # "docker" or "basilica"
    basilica_api_token: str | None = None

    # HuggingFace settings
    hf_token: str | None = None

    # Deployment settings
    ttl_seconds: int = 600  # 10 minutes
    warmup_timeout: int = 300  # 5 minutes
    mem_limit: str = "4g"

    # Resource allocation (for basilica mode)
    gpu_count: int = 0
    min_gpu_memory_gb: int | None = None
    cpu: str = "1"
    memory: str = "4Gi"


@dataclass
class ManagedDeployment:
    """Tracks a miner's managed deployment."""

    env: Any  # affinetes environment object (for Docker) or None (for Basilica)
    url: str
    repo: str
    revision: str
    miner_uid: int
    mode: str = "docker"
    basilica_deployment_id: str | None = None  # For Basilica mode cleanup
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_used: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def touch(self) -> None:
        """Update last_used timestamp."""
        self.last_used = datetime.now(timezone.utc)

    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if deployment has exceeded TTL since last use."""
        now = datetime.now(timezone.utc)
        elapsed = (now - self.last_used).total_seconds()
        return elapsed > ttl_seconds


class MinerDeploymentManager:
    """
    Manages miner deployments via affinetes with TTL caching.

    Deployments are created on-demand when tasks reference a miner's repo:revision.
    The same deployment is reused for the same (repo, revision) combination.
    Deployments are cleaned up after TTL expiry.

    This uses affinetes as the abstraction layer, supporting both Docker and
    Basilica backends (matching how evaluation environments work).
    """

    def __init__(self, config: MinerDeploymentConfig):
        self.config = config
        # Cache key: (repo, revision) -> ManagedDeployment
        self._deployments: dict[tuple[str, str], ManagedDeployment] = {}
        self._lock = asyncio.Lock()

    async def get_or_create_deployment(self, miner_uid: int, repo: str, revision: str) -> str:
        """
        Get cached or create new deployment. Returns URL.

        Args:
            miner_uid: Miner's UID (used for logging)
            repo: HuggingFace repository (e.g., "user/model")
            revision: HuggingFace commit SHA

        Returns:
            URL for the miner's policy endpoint
        """
        cache_key = (repo, revision)

        async with self._lock:
            # Check cache
            if cache_key in self._deployments:
                deployment = self._deployments[cache_key]
                deployment.touch()
                logger.debug(
                    "deployment_cache_hit",
                    repo=repo,
                    revision=revision[:8],
                    url=deployment.url,
                )
                return deployment.url

        # Create new deployment (outside lock to avoid blocking)
        logger.info(
            "creating_miner_deployment",
            miner_uid=miner_uid,
            repo=repo,
            revision=revision[:12],
            mode=self.config.mode,
        )

        deployment = await self._create_deployment(miner_uid, repo, revision)

        async with self._lock:
            # Double-check cache (another task may have created it)
            if cache_key in self._deployments:
                # Another task created it, cleanup ours and use theirs
                await self._cleanup_deployment(deployment)
                existing = self._deployments[cache_key]
                existing.touch()
                return existing.url

            self._deployments[cache_key] = deployment
            logger.info(
                "miner_deployment_created",
                miner_uid=miner_uid,
                repo=repo,
                revision=revision[:12],
                url=deployment.url,
                mode=self.config.mode,
            )
            return deployment.url

    async def _create_deployment(
        self, miner_uid: int, repo: str, revision: str
    ) -> ManagedDeployment:
        """Create a new deployment for the miner."""
        if self.config.mode == "basilica":
            return await self._create_basilica_deployment(miner_uid, repo, revision)
        else:
            return await self._create_docker_deployment(miner_uid, repo, revision)

    async def _create_docker_deployment(
        self, miner_uid: int, repo: str, revision: str
    ) -> ManagedDeployment:
        """Create a Docker deployment via affinetes."""
        # Build environment variables for the miner runner
        env_vars: dict[str, str] = {
            "HF_REPO": repo,
            "HF_REVISION": revision,
        }
        if self.config.hf_token:
            env_vars["HF_TOKEN"] = self.config.hf_token

        # Generate container name
        revision_short = revision[:8]
        container_name = f"kinitro-miner-{miner_uid}-{revision_short}"

        # Load environment via affinetes
        load_kwargs: dict[str, Any] = {
            "image": self.config.image,
            "mode": "docker",
            "env_vars": env_vars,
            "container_name": container_name,
            "mem_limit": self.config.mem_limit,
        }

        # Load the miner environment
        env = await asyncio.to_thread(af.load_env, **load_kwargs)

        # Wait for server to be ready by calling health_check
        await self._wait_for_ready(env, miner_uid, repo, revision)

        # Get the endpoint URL
        url = await self._get_endpoint_url(env)

        return ManagedDeployment(
            env=env,
            url=url,
            repo=repo,
            revision=revision,
            miner_uid=miner_uid,
            mode="docker",
        )

    async def _create_basilica_deployment(
        self, miner_uid: int, repo: str, revision: str
    ) -> ManagedDeployment:
        """Create a Basilica deployment using the SDK directly for persistence."""
        from basilica import BasilicaClient  # noqa: PLC0415 - lazy import

        # Build environment variables for the miner runner
        env_vars: dict[str, str] = {
            "HF_REPO": repo,
            "HF_REVISION": revision,
        }
        if self.config.hf_token:
            env_vars["HF_TOKEN"] = self.config.hf_token

        # Generate deployment name
        revision_short = revision[:8]
        timestamp = int(time.time())
        deployment_name = f"kinitro-miner-{miner_uid}-{revision_short}-{timestamp}"[:63]

        # Set API token in environment for Basilica client
        api_token = self.config.basilica_api_token or os.environ.get("BASILICA_API_TOKEN")
        if not api_token:
            raise RuntimeError("Basilica API token not configured")

        # Create Basilica client and deployment
        client = BasilicaClient(api_key=api_token)

        logger.info(
            "creating_basilica_deployment",
            miner_uid=miner_uid,
            deployment_name=deployment_name,
            image=self.config.image,
        )

        # Create deployment with TTL (convert to seconds)
        ttl_seconds = self.config.ttl_seconds + 300  # Add buffer for warmup

        deployment = await asyncio.to_thread(
            client.create_deployment,
            instance_name=deployment_name,
            image=self.config.image,
            port=8000,  # Affinetes HTTP server runs on port 8000
            env=env_vars,
            cpu=self.config.cpu,
            memory=self.config.memory,
            ttl_seconds=ttl_seconds,
        )

        # Wait for deployment to be ready by polling
        instance_name = deployment.instance_name
        logger.info(
            "waiting_for_basilica_deployment",
            deployment_name=instance_name,
            miner_uid=miner_uid,
        )

        # Poll for deployment readiness
        start = time.time()
        while time.time() - start < self.config.warmup_timeout:
            status = await asyncio.to_thread(client.get_deployment, instance_name)
            if status.phase == "Running" or (
                status.url and "deployments.basilica.ai" in status.url
            ):
                deployment = status
                break
            logger.debug(
                "basilica_deployment_polling",
                phase=status.phase,
                state=status.state,
                deployment_name=instance_name,
            )
            await asyncio.sleep(2)
        else:
            raise TimeoutError(f"Basilica deployment not ready after {self.config.warmup_timeout}s")

        url = deployment.url
        logger.info(
            "basilica_deployment_ready",
            deployment_name=deployment.instance_name,
            url=url,
            miner_uid=miner_uid,
        )

        # Wait for HTTP server to be ready
        await self._wait_for_http_ready(url, miner_uid, repo, revision)

        return ManagedDeployment(
            env=None,  # No affinetes env for Basilica
            url=url,
            repo=repo,
            revision=revision,
            miner_uid=miner_uid,
            mode="basilica",
            basilica_deployment_id=deployment.instance_name,
        )

    async def _wait_for_http_ready(
        self, url: str, miner_uid: int, repo: str, revision: str
    ) -> None:
        """Wait for the HTTP server to be ready at the given URL."""
        timeout = self.config.warmup_timeout
        start = time.time()

        while time.time() - start < timeout:
            try:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.get(f"{url}/health")
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("status") in ("healthy", "ok"):
                            logger.info(
                                "miner_deployment_ready",
                                miner_uid=miner_uid,
                                repo=repo,
                                revision=revision[:8],
                            )
                            return
            except Exception as e:
                logger.debug(
                    "miner_http_not_ready",
                    miner_uid=miner_uid,
                    error=str(e)[:100],
                )
            await asyncio.sleep(2)

        raise TimeoutError(f"Miner HTTP server not ready after {timeout}s: {repo}:{revision[:8]}")

    async def _wait_for_ready(self, env: Any, miner_uid: int, repo: str, revision: str) -> None:
        """Wait for the miner deployment to be ready."""
        timeout = self.config.warmup_timeout
        start = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start < timeout:
            try:
                # Call health_check on the Actor
                result = await env.health_check()
                if result.get("status") == "healthy":
                    logger.info(
                        "miner_deployment_ready",
                        miner_uid=miner_uid,
                        repo=repo,
                        revision=revision[:8],
                    )
                    return
            except Exception as e:
                logger.debug(
                    "miner_deployment_not_ready",
                    miner_uid=miner_uid,
                    error=str(e),
                )
            await asyncio.sleep(2)

        raise TimeoutError(f"Miner deployment not ready after {timeout}s: {repo}:{revision[:8]}")

    async def _get_endpoint_url(self, env: Any) -> str:
        """Get the endpoint URL from the affinetes environment.

        For miner deployments, we need the external container URL (affinetes' URL)
        so that eval environments can reach the miner. The Actor's internal URL
        (127.0.0.1:8001) only works within the same container.
        """
        # Get URL from affinetes' internal HTTP executor
        # Path: env._backend._http_executor.base_url
        try:
            if hasattr(env, "_backend"):
                backend = env._backend
                if hasattr(backend, "_http_executor"):
                    http_exec = backend._http_executor
                    if hasattr(http_exec, "base_url") and isinstance(http_exec.base_url, str):
                        return http_exec.base_url
        except Exception:
            pass

        raise RuntimeError("Could not determine endpoint URL from affinetes environment")

    async def _cleanup_deployment(self, deployment: ManagedDeployment) -> None:
        """Cleanup a single deployment."""
        try:
            if deployment.mode == "basilica" and deployment.basilica_deployment_id:
                # Use Basilica SDK to delete the deployment
                from basilica import BasilicaClient  # noqa: PLC0415 - lazy import

                api_token = self.config.basilica_api_token or os.environ.get("BASILICA_API_TOKEN")
                if api_token:
                    client = BasilicaClient(api_key=api_token)
                    await asyncio.to_thread(
                        client.delete_deployment, deployment.basilica_deployment_id
                    )
                    logger.info(
                        "basilica_deployment_deleted",
                        deployment_id=deployment.basilica_deployment_id,
                        repo=deployment.repo,
                    )
            elif deployment.env is not None:
                # Docker mode: cleanup via affinetes
                await deployment.env.cleanup()
        except Exception as e:
            logger.warning(
                "deployment_cleanup_failed",
                repo=deployment.repo,
                revision=deployment.revision[:8],
                error=str(e),
            )

    async def cleanup_expired(self) -> int:
        """
        Remove deployments past TTL.

        Returns:
            Count of deployments removed
        """
        async with self._lock:
            expired_keys = [
                key
                for key, deployment in self._deployments.items()
                if deployment.is_expired(self.config.ttl_seconds)
            ]

        removed = 0
        for key in expired_keys:
            async with self._lock:
                if key in self._deployments:
                    deployment = self._deployments.pop(key)

            try:
                await self._cleanup_deployment(deployment)
                logger.info(
                    "deployment_expired_cleaned",
                    repo=deployment.repo,
                    revision=deployment.revision[:8],
                )
                removed += 1
            except Exception as e:
                logger.warning(
                    "deployment_cleanup_failed",
                    repo=deployment.repo,
                    error=str(e),
                )

        if removed > 0:
            logger.info("expired_deployments_cleaned", count=removed)

        return removed

    async def shutdown(self) -> None:
        """Shutdown all active deployments."""
        async with self._lock:
            deployments_to_cleanup = list(self._deployments.values())
            self._deployments.clear()

        logger.info("shutting_down_deployments", count=len(deployments_to_cleanup))

        for deployment in deployments_to_cleanup:
            try:
                await self._cleanup_deployment(deployment)
                logger.info(
                    "deployment_shutdown",
                    repo=deployment.repo,
                    revision=deployment.revision[:8],
                )
            except Exception as e:
                logger.warning(
                    "deployment_shutdown_failed",
                    repo=deployment.repo,
                    error=str(e),
                )

    def get_active_deployments(self) -> list[ManagedDeployment]:
        """Get list of all active deployments."""
        return list(self._deployments.values())

    def get_deployment_count(self) -> int:
        """Get count of active deployments."""
        return len(self._deployments)
