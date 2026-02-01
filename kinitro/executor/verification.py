"""
Model verification module for spot-checking miner deployments.

This module verifies that what miners deploy to Basilica matches what they
uploaded to HuggingFace. It works by:

1. Downloading the policy from HuggingFace
2. Running inference locally with a fixed seed
3. Comparing against the miner's endpoint response

If outputs differ significantly, the miner may be running different code
than what they committed.
"""

import asyncio
import hashlib
import importlib.util
import os
import random
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import structlog
from huggingface_hub import HfApi, snapshot_download

from kinitro.rl_interface import CanonicalAction, CanonicalObservation

logger = structlog.get_logger()


@dataclass
class VerificationResult:
    """Result of a model verification check."""

    miner_uid: int
    miner_hotkey: str
    repo: str
    revision: str
    verified: bool
    match_score: float  # 0.0 = no match, 1.0 = perfect match
    error: str | None = None
    details: dict[str, Any] | None = None


class PolicyVerifier:
    """
    Verifies that miner deployments match their HuggingFace uploads.

    Uses spot-checking: randomly selects a percentage of evaluations
    to verify, comparing local inference against remote endpoint.
    """

    # Default max repo size: 5GB
    DEFAULT_MAX_REPO_SIZE_GB = 5.0

    def __init__(
        self,
        verification_rate: float = 0.05,  # 5% of evaluations
        tolerance: float = 1e-3,  # Relative tolerance for floating point comparison
        num_samples: int = 5,  # Number of observations to compare
        cache_dir: str | None = None,
        max_repo_size_gb: float = DEFAULT_MAX_REPO_SIZE_GB,
    ):
        """
        Initialize the policy verifier.

        Args:
            verification_rate: Probability of verifying each miner (0.0 to 1.0)
            tolerance: Relative tolerance for comparing actions
            num_samples: Number of test observations per verification
            cache_dir: Directory to cache downloaded models
            max_repo_size_gb: Maximum allowed HuggingFace repo size in GB

        Raises:
            ValueError: If any parameter is invalid
        """
        if not 0.0 <= verification_rate <= 1.0:
            raise ValueError("verification_rate must be between 0.0 and 1.0")
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        if num_samples <= 0:
            raise ValueError("num_samples must be >= 1")
        if max_repo_size_gb <= 0:
            raise ValueError("max_repo_size_gb must be > 0")

        self.verification_rate = verification_rate
        self.tolerance = tolerance
        self.num_samples = num_samples
        self.cache_dir = cache_dir or tempfile.mkdtemp(prefix="kinitro_verify_")
        self.max_repo_size_bytes = int(max_repo_size_gb * 1024 * 1024 * 1024)
        self._policy_cache: dict[str, Any] = {}

    def should_verify(self) -> bool:
        """Randomly decide whether to verify based on verification_rate."""
        return random.random() < self.verification_rate

    async def verify_miner(
        self,
        miner_uid: int,
        miner_hotkey: str,
        repo: str,
        revision: str,
        endpoint: str,
    ) -> VerificationResult:
        """
        Verify a miner's deployment matches their HuggingFace model.

        Args:
            miner_uid: Miner's UID
            miner_hotkey: Miner's hotkey
            repo: HuggingFace repo (e.g., "user/model")
            revision: HuggingFace commit SHA
            endpoint: Miner's Basilica endpoint URL

        Returns:
            VerificationResult with match status
        """
        logger.info(
            "verification_starting",
            miner_uid=miner_uid,
            repo=repo,
            revision=revision[:12],
        )

        try:
            # Load policy from HuggingFace
            policy = await self._load_policy_from_hf(repo, revision)

            # Generate deterministic test observations
            # Use hashlib for cross-process determinism (hash() is randomized by PYTHONHASHSEED)
            seed_str = f"{miner_uid}:{revision}".encode()
            test_seed = int(hashlib.sha256(seed_str).hexdigest()[:8], 16) % (2**31)
            np.random.seed(test_seed)

            # Generate CanonicalObservation objects for testing
            test_observations = []
            for _ in range(self.num_samples):
                obs = CanonicalObservation(
                    ee_pos_m=np.random.uniform(-1, 1, size=3).tolist(),
                    ee_quat_xyzw=[0.0, 0.0, 0.0, 1.0],  # Identity quaternion
                    ee_lin_vel_mps=np.random.uniform(-0.5, 0.5, size=3).tolist(),
                    ee_ang_vel_rps=np.random.uniform(-0.5, 0.5, size=3).tolist(),
                    gripper_01=float(np.random.uniform(0, 1)),
                    rgb={},  # No images for verification (simpler comparison)
                )
                test_observations.append(obs)

            # Get actions from local policy
            local_actions = []
            for i, obs in enumerate(test_observations):
                seed = test_seed + i
                self._set_seed(seed)
                action = await self._get_local_action(policy, obs, seed)
                local_actions.append(action)

            # Get actions from remote endpoint
            remote_actions = []
            for i, obs in enumerate(test_observations):
                seed = test_seed + i
                action = await self._get_remote_action(endpoint, obs, seed)
                remote_actions.append(action)

            # Compare actions
            match_scores = []
            for local, remote in zip(local_actions, remote_actions):
                if local is None or remote is None:
                    match_scores.append(0.0)
                else:
                    match_scores.append(self._compare_actions(local, remote))

            avg_match = np.mean(match_scores)
            verified = avg_match >= (1.0 - self.tolerance)

            logger.info(
                "verification_complete",
                miner_uid=miner_uid,
                verified=verified,
                match_score=round(avg_match, 4),
                num_samples=self.num_samples,
            )

            return VerificationResult(
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                repo=repo,
                revision=revision,
                verified=verified,
                match_score=avg_match,
                details={
                    "match_scores": match_scores,
                    "test_seed": test_seed,
                    "num_samples": self.num_samples,
                },
            )

        except Exception as e:
            logger.error(
                "verification_failed",
                miner_uid=miner_uid,
                error=str(e),
            )
            return VerificationResult(
                miner_uid=miner_uid,
                miner_hotkey=miner_hotkey,
                repo=repo,
                revision=revision,
                verified=False,
                match_score=0.0,
                error=str(e),
            )

    async def _load_policy_from_hf(self, repo: str, revision: str) -> Any:
        """
        Load a policy from HuggingFace.

        Downloads the model files and imports the policy class.
        Checks repo size before downloading to prevent DoS attacks.
        """
        cache_key = f"{repo}:{revision}"
        if cache_key in self._policy_cache:
            return self._policy_cache[cache_key]

        # Check repo size before downloading
        api = HfApi()
        try:
            repo_info = await asyncio.to_thread(
                api.repo_info,
                repo_id=repo,
                revision=revision,
                repo_type="model",
            )

            # Calculate total size from siblings (files in repo)
            total_size = 0
            if repo_info.siblings:
                for sibling in repo_info.siblings:
                    if sibling.size is not None:
                        total_size += sibling.size

            if total_size > self.max_repo_size_bytes:
                size_gb = total_size / (1024 * 1024 * 1024)
                max_gb = self.max_repo_size_bytes / (1024 * 1024 * 1024)
                raise ValueError(
                    f"Repository size ({size_gb:.2f}GB) exceeds maximum allowed ({max_gb:.2f}GB)"
                )

            logger.info(
                "repo_size_checked",
                repo=repo,
                revision=revision[:12],
                size_mb=round(total_size / (1024 * 1024), 2),
            )

        except Exception as e:
            if "exceeds maximum" in str(e):
                raise
            # Fail closed: don't download if we can't verify size (security requirement)
            logger.error(
                "repo_size_check_failed",
                repo=repo,
                error=str(e),
            )
            raise ValueError(
                f"Cannot verify repository size for {repo}: {e}. "
                "Size check is required for security."
            ) from e

        # Download from HuggingFace
        model_path = await asyncio.to_thread(
            snapshot_download,
            repo,
            revision=revision,
            cache_dir=self.cache_dir,
            local_dir=os.path.join(self.cache_dir, repo.replace("/", "_"), revision[:12]),
        )

        if isinstance(model_path, list):
            raise ValueError("Unexpected model path format from HuggingFace download")
        model_path_str = str(model_path)

        # Load the policy module
        policy_file = os.path.join(model_path_str, "policy.py")
        if not os.path.exists(policy_file):
            raise FileNotFoundError(f"policy.py not found in {repo}@{revision}")

        # Import the policy module dynamically
        spec = importlib.util.spec_from_file_location("miner_policy", policy_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load policy module from {policy_file}")
        module = importlib.util.module_from_spec(spec)

        # Add model path to sys.path for relative imports
        sys.path.insert(0, model_path_str)
        try:
            spec.loader.exec_module(module)
        finally:
            sys.path.remove(model_path_str)

        # Instantiate the policy
        if not hasattr(module, "RobotPolicy"):
            raise AttributeError(f"RobotPolicy class not found in {repo}@{revision}")

        policy = module.RobotPolicy()
        self._policy_cache[cache_key] = policy

        logger.info(
            "policy_loaded_from_hf",
            repo=repo,
            revision=revision[:12],
            model_path=model_path_str,
        )

        return policy

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)

    async def _get_local_action(
        self, policy: Any, canonical_obs: CanonicalObservation, seed: int
    ) -> np.ndarray | None:
        """Get action from local policy."""
        try:
            self._set_seed(seed)

            # Try async first, fall back to sync
            if asyncio.iscoroutinefunction(policy.act):
                action = await policy.act(canonical_obs)
            else:
                action = policy.act(canonical_obs)

            # Handle different action return types
            # Note: Check by method/attribute rather than isinstance() because
            # the miner's bundled rl_interface.py has its own CanonicalAction class
            if hasattr(action, "to_array"):
                return action.to_array()
            if isinstance(action, dict):
                return CanonicalAction.model_validate(action).to_array()
            if hasattr(action, "numpy"):
                action = action.numpy()
            return np.array(action, dtype=np.float32)
        except Exception as e:
            logger.warning("local_inference_failed", error=str(e))
            return None

    async def _get_remote_action(
        self, endpoint: str, canonical_obs: CanonicalObservation, seed: int
    ) -> np.ndarray | None:
        """Get action from remote miner endpoint."""
        try:
            url = f"{endpoint.rstrip('/')}/act"
            # Use the same format as the evaluator: {"obs": CanonicalObservation}
            payload = {"obs": canonical_obs.model_dump(mode="python")}

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                action_data = data.get("action", {})
                if isinstance(action_data, dict):
                    return CanonicalAction.model_validate(action_data).to_array()
                return np.array(action_data, dtype=np.float32)

        except Exception as e:
            logger.warning("remote_inference_failed", error=str(e))
            return None

    def _compare_actions(self, local: np.ndarray, remote: np.ndarray) -> float:
        """
        Compare two action vectors.

        Returns a match score from 0.0 (no match) to 1.0 (perfect match).
        """
        if local.shape != remote.shape:
            return 0.0

        # Use relative comparison for floating point
        if np.allclose(local, remote, rtol=self.tolerance, atol=1e-6):
            return 1.0

        # Calculate a continuous match score based on relative error
        rel_error = np.abs(local - remote) / (np.abs(local) + 1e-8)
        mean_rel_error = np.mean(rel_error)

        # Convert to match score (exponential decay)
        match_score = np.exp(-mean_rel_error / self.tolerance)
        return float(match_score)

    def compute_model_hash(self, model_path: str) -> str:
        """
        Compute a deterministic hash of model weights.

        This can be used for plagiarism detection - models with the same
        hash are copies of each other.
        """
        hasher = hashlib.sha256()

        # Find all weight files
        weight_extensions = [".pt", ".pth", ".safetensors", ".bin", ".ckpt"]
        weight_files = []

        for ext in weight_extensions:
            weight_files.extend(Path(model_path).rglob(f"*{ext}"))

        # Sort for deterministic ordering
        weight_files = sorted(weight_files)

        for weight_file in weight_files:
            with open(weight_file, "rb") as f:
                # Read in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)

        return hasher.hexdigest()

    def cleanup(self) -> None:
        """Clean up cached models."""
        self._policy_cache.clear()
        if os.path.exists(self.cache_dir) and self.cache_dir.startswith(tempfile.gettempdir()):
            shutil.rmtree(self.cache_dir, ignore_errors=True)
