"""
Episode and step logging system.

This module provides utilities for tracking episode data during evaluation.
Real-time streaming of episode/step data and S3 uploads are not yet implemented
- only final aggregated results are sent after evaluation completes.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from core.constants import ImageFormat
from core.storage import S3Config

from .worker_utils import extract_success_flag

logger = logging.getLogger(__name__)


@dataclass
class LoggingConfig:
    """Configuration for episode and step logging."""

    # Logging intervals
    episode_log_interval: int = 1  # Log every N episodes
    step_log_interval: int = 10  # Log every N steps within an episode

    # Storage configuration (not currently used)
    enable_s3_upload: bool = False
    s3_config: Optional[S3Config] = None

    # Local storage fallback (not currently used)
    local_save_dir: Optional[Path] = None

    # Image settings (not currently used)
    image_format: ImageFormat = ImageFormat.PNG
    image_quality: int = 95  # For JPEG


@dataclass
class EpisodeLogger:
    """Logger for episode and step data.
    
    This logger tracks episode progress for aggregation into final results.
    Real-time streaming of episode/step data to the backend and S3 uploads
    are not implemented yet - only final aggregated results are sent after
    evaluation completes.
    """

    config: LoggingConfig
    submission_id: str
    job_id: int
    task_id: str
    env_name: str
    benchmark_name: str

    # Internal state
    _episode_count: int = field(default=0, init=False)
    _current_episode_id: Optional[int] = field(default=None, init=False)
    _current_episode_steps: List[Dict[str, Any]] = field(
        default_factory=list, init=False
    )
    _current_episode_start: Optional[datetime] = field(default=None, init=False)

    def start_episode(self, episode_id: int) -> None:
        """Start tracking a new episode.

        Args:
            episode_id: Unique identifier for the episode
        """
        self._current_episode_id = episode_id
        self._current_episode_steps = []
        self._current_episode_start = datetime.now(timezone.utc)
        self._episode_count += 1

        logger.debug(f"Started tracking episode {episode_id}")

    def _infer_success_from_steps(self) -> bool:
        """Return True if any recorded step info reports success."""
        for step_data in reversed(self._current_episode_steps):
            info = step_data.get("info")
            if info and extract_success_flag(info):
                return True
        return False

    def has_logged_success(self) -> bool:
        """Public helper used by callers that need to confirm success state."""
        return self._infer_success_from_steps()

    async def log_step(
        self,
        step: int,
        action: np.ndarray | List | Dict,
        reward: float,
        done: bool,
        truncated: bool,
        observations: Optional[List[Tuple[np.ndarray, str]]] = None,
        info: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a single step within an episode.

        Args:
            step: Step number within the episode
            action: Action taken at this step
            reward: Reward received
            done: Whether episode terminated
            truncated: Whether episode was truncated
            observations: List of (image, camera_name) tuples (not currently used)
            info: Additional info from environment
        """
        logger.info("Logging step data")
        if self._current_episode_id is None:
            logger.warning("Attempted to log step without active episode")
            return

        # Accumulate step data for episode summary
        step_data = {
            "step": step,
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "info": info or {},
        }

        self._current_episode_steps.append(step_data)

    async def end_episode(
        self,
        final_reward: float,
        success: bool,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End the current episode and save data.

        Args:
            final_reward: Final reward for the episode
            success: Whether the episode was successful
            extra_metrics: Additional metrics to store
        """
        if self._current_episode_id is None:
            logger.warning("Attempted to end episode without active episode")
            return

        # Log episode completion (data is aggregated in final results, not streamed)
        logger.debug(
            "Episode %s completed: reward=%.3f success=%s steps=%d",
            self._current_episode_id,
            final_reward,
            success,
            len(self._current_episode_steps),
        )

        # Reset for next episode
        self._current_episode_id = None
        self._current_episode_steps = []
        self._current_episode_start = None

    def cleanup(self) -> None:
        """Clean up resources (no-op for now)."""
        pass

    def get_statistics(self) -> Dict[str, Any]:
        """Get current logging statistics.

        Returns:
            Dictionary with logging statistics
        """
        return {
            "episodes_tracked": self._episode_count,
            "current_episode_id": self._current_episode_id,
            "current_episode_steps": len(self._current_episode_steps),
            "storage_enabled": False,
        }
