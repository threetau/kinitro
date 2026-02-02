"""
Affinetes-compatible Miner Policy Runner.

This Actor class runs inside an affinetes-managed container and:
1. Downloads miner policy from HuggingFace
2. Runs the miner's server.py as an internal subprocess
3. Proxies requests to the internal server

Environment Variables:
    HF_REPO     - HuggingFace repository (e.g., "user/policy")
    HF_REVISION - HuggingFace commit SHA
    HF_TOKEN    - Optional HuggingFace token for private repos

Usage (from executor):
    import affinetes as af

    env = af.load_env(
        image="kinitro/miner-runner:v1",
        env_vars={"HF_REPO": "user/policy", "HF_REVISION": "abc123"},
    )

    # Check health
    health = await env.health_check()

    # Reset for episode
    await env.reset(task_config={...})

    # Get action
    action = await env.act(obs={...})
"""

import asyncio
import os
import signal
import subprocess
import sys
import time

import httpx
import structlog

logger = structlog.get_logger()

# Internal server port (affinetes exposes Actor on 8000, we use 8001 internally)
INTERNAL_PORT = 8001
MINER_DIR = "/miner"


class Actor:
    """
    Miner policy runner actor for affinetes.

    Downloads policy from HuggingFace and runs it as an internal server,
    then proxies requests to that server.
    """

    def __init__(self):
        """Initialize the miner runner."""
        self._server_process: subprocess.Popen | None = None
        self._initialized = False
        self._internal_url = f"http://127.0.0.1:{INTERNAL_PORT}"

    async def _ensure_initialized(self) -> None:
        """Ensure miner policy is downloaded and server is running."""
        if self._initialized:
            return

        hf_repo = os.environ.get("HF_REPO")
        hf_revision = os.environ.get("HF_REVISION")
        hf_token = os.environ.get("HF_TOKEN")

        if not hf_repo or not hf_revision:
            raise ValueError("HF_REPO and HF_REVISION environment variables are required")

        logger.info(
            "initializing_miner_runner",
            repo=hf_repo,
            revision=hf_revision[:8] if hf_revision else None,
        )

        # Download from HuggingFace
        await self._download_model(hf_repo, hf_revision, hf_token)

        # Install miner's requirements if they exist
        await self._install_requirements()

        # Start the internal server
        await self._start_server()

        # Wait for server to be ready
        await self._wait_for_server()

        self._initialized = True
        logger.info("miner_runner_initialized", url=self._internal_url)

    async def _download_model(self, repo: str, revision: str, token: str | None) -> None:
        """Download model from HuggingFace."""
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        logger.info("downloading_model", repo=repo, revision=revision[:8])

        # Run in thread to avoid blocking
        await asyncio.to_thread(
            snapshot_download,
            repo,
            revision=revision,
            local_dir=MINER_DIR,
            token=token,
        )

        logger.info("model_downloaded", path=MINER_DIR)

    async def _install_requirements(self) -> None:
        """Install miner's requirements.txt if it exists."""
        requirements_path = os.path.join(MINER_DIR, "requirements.txt")

        if not os.path.exists(requirements_path):
            logger.info("no_requirements_file")
            return

        logger.info("installing_requirements", path=requirements_path)

        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-cache-dir",
            "-r",
            requirements_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            logger.warning(
                "requirements_install_failed",
                returncode=proc.returncode,
                stderr=stderr.decode()[:500],
            )
        else:
            logger.info("requirements_installed")

    async def _start_server(self) -> None:
        """Start the miner's server as a subprocess."""
        server_path = os.path.join(MINER_DIR, "server.py")

        if not os.path.exists(server_path):
            raise FileNotFoundError(f"server.py not found at {server_path}")

        logger.info("starting_internal_server", port=INTERNAL_PORT)

        # Start uvicorn in subprocess
        self._server_process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "server:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(INTERNAL_PORT),
            ],
            cwd=MINER_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "PYTHONPATH": MINER_DIR},
        )

        logger.info("internal_server_started", pid=self._server_process.pid)

    async def _wait_for_server(self, timeout: float = 60.0) -> None:
        """Wait for internal server to be ready."""
        start = time.time()
        last_error = None

        while time.time() - start < timeout:
            try:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    resp = await client.get(f"{self._internal_url}/health")
                    if resp.status_code == 200:
                        logger.info("internal_server_ready")
                        return
            except Exception as e:
                last_error = e

            # Check if process died
            if self._server_process and self._server_process.poll() is not None:
                stdout, stderr = self._server_process.communicate()
                raise RuntimeError(
                    f"Server process died with code {self._server_process.returncode}. "
                    f"stderr: {stderr.decode()[:500]}"
                )

            await asyncio.sleep(0.5)

        raise TimeoutError(f"Server not ready after {timeout}s. Last error: {last_error}")

    async def _call_internal(
        self,
        method: str,
        endpoint: str,
        json: dict | None = None,
        timeout: float = 5.0,
    ) -> dict:
        """Call the internal miner server."""
        await self._ensure_initialized()

        url = f"{self._internal_url}/{endpoint}"

        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                resp = await client.get(url)
            else:
                resp = await client.post(url, json=json)

            resp.raise_for_status()
            return resp.json()

    async def health_check(self) -> dict:
        """
        Check health of the miner server.

        Returns:
            Health status dict with status, model_loaded, uptime_seconds
        """
        return await self._call_internal("GET", "health")

    async def reset(self, task_config: dict) -> dict:
        """
        Reset miner policy for new episode.

        Args:
            task_config: Task configuration dict

        Returns:
            Reset response with status and optional episode_id
        """
        return await self._call_internal(
            "POST",
            "reset",
            json={"task_config": task_config},
            timeout=10.0,
        )

    async def act(self, obs: dict, timeout: float = 0.5) -> dict:
        """
        Get action from miner policy.

        Args:
            obs: Observation dict (canonical observation format)
            timeout: Request timeout in seconds

        Returns:
            Action response with action dict
        """
        return await self._call_internal(
            "POST",
            "act",
            json={"obs": obs},
            timeout=timeout,
        )

    async def get_endpoint_url(self) -> str:
        """
        Get the URL for direct access to the miner server.

        This is used by eval environments to call the miner directly.

        Returns:
            Internal server URL (e.g., "http://127.0.0.1:8001")
        """
        await self._ensure_initialized()
        return self._internal_url

    async def cleanup(self) -> None:
        """Cleanup resources and stop internal server."""
        if self._server_process:
            logger.info("stopping_internal_server", pid=self._server_process.pid)

            # Send SIGTERM
            self._server_process.send_signal(signal.SIGTERM)

            try:
                self._server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                # Force kill if not responding
                self._server_process.kill()
                self._server_process.wait()

            self._server_process = None
            logger.info("internal_server_stopped")

        self._initialized = False
