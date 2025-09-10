"""
Episode and step logging system with R2 storage integration.

This module provides utilities for logging episode data and step-level
observations to both database and R2 storage with configurable intervals.
"""

import logging
from concurrent.futures import (
    ThreadPoolExecutor,
    wait,
)
from concurrent.futures import (
    TimeoutError as FutureTimeoutError,
)
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
import numpy as np
from pgqueuer import Queries
from pgqueuer.db import AsyncpgDriver

from core.constants import ImageFormat
from core.messages import EpisodeDataMessage, EpisodeStepDataMessage
from core.storage import R2Config, R2StorageClient

logger = logging.getLogger(__name__)

OBS_UPLOAD_CLEANUP_TIMEOUT = 5.0
OBS_UPLOAD_TIMEOUT = 20.0
MAX_UPLOAD_WORKERS = 4


@dataclass
class LoggingConfig:
    """Configuration for episode and step logging."""

    # Logging intervals
    episode_log_interval: int = 1  # Log every N episodes
    step_log_interval: int = 10  # Log every N steps within an episode

    # Storage configuration
    enable_r2_upload: bool = True
    r2_config: Optional[R2Config] = None

    # Local storage fallback
    local_save_dir: Optional[Path] = None

    # Image settings
    image_format: ImageFormat = ImageFormat.PNG
    image_quality: int = 95  # For JPEG

    # Database URL for pgqueuer
    database_url: Optional[str] = None


@dataclass
class EpisodeLogger:
    """Logger for episode and step data with R2 integration."""

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
    _storage_client: Optional[R2StorageClient] = field(default=None, init=False)

    # Background upload system
    _upload_executor: Optional[ThreadPoolExecutor] = field(default=None, init=False)
    _upload_futures: List[Any] = field(default_factory=list, init=False)
    _pending_uploads: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Initialize storage client and upload executor if R2 is enabled."""
        if self.config.enable_r2_upload and self.config.r2_config:
            try:
                self._storage_client = R2StorageClient(self.config.r2_config)
                # Create thread pool for background uploads
                self._upload_executor = ThreadPoolExecutor(
                    max_workers=MAX_UPLOAD_WORKERS, thread_name_prefix="r2_upload"
                )
                logger.info("R2 storage client initialized with background upload pool")
            except Exception as e:
                logger.error(f"Failed to initialize R2 storage: {e}")
                logger.warning("Falling back to local storage only")
                self._storage_client = None
                self._upload_executor = None

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
            observations: List of (image, camera_name) tuples
            info: Additional info from environment
        """
        logger.info("Logging step data")
        if self._current_episode_id is None:
            logger.warning("Attempted to log step without active episode")
            return

        # Check if we should log this step based on interval
        should_log_step = (
            (step % self.config.step_log_interval == 0) or done or truncated
        )

        if not should_log_step:
            # Still accumulate basic data for episode summary
            self._current_episode_steps.append(
                {
                    "step": step,
                    "reward": reward,
                    "done": done,
                    "truncated": truncated,
                    "should_log": False,  # Mark as not to be logged
                }
            )
            return

        step_data = {
            "step": step,
            "action": self._serialize_action(action),
            "reward": reward,
            "done": done,
            "truncated": truncated,
            "timestamp": datetime.now(timezone.utc),
            "info": info or {},
        }

        # Schedule background upload of observations if available
        if observations and self._storage_client:
            upload_key = self._upload_observations_async(
                observations, self._current_episode_id, step
            )
            # Store placeholder for now, will be resolved before sending to backend
            step_data["upload_key"] = upload_key
            step_data["observation_refs"] = {}  # Will be filled in later
        else:
            step_data["observation_refs"] = {}
            step_data["upload_key"] = None

        # Mark step for later queuing if it should be logged
        step_data["should_log"] = should_log_step
        self._current_episode_steps.append(step_data)

        # Don't queue immediately - queue at end of episode to avoid race condition

    async def end_episode(
        self,
        total_reward: float,
        success: bool,
        extra_metrics: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End the current episode and save data.

        Args:
            total_reward: Total reward for the episode
            success: Whether the episode was successful
            extra_metrics: Additional metrics to store
        """
        if self._current_episode_id is None:
            logger.warning("Attempted to end episode without active episode")
            return

        # Check if we should log this episode based on interval
        should_log_episode = self._episode_count % self.config.episode_log_interval == 0

        if should_log_episode:
            episode_data = {
                "job_id": self.job_id,
                "submission_id": self.submission_id,
                "task_id": self.task_id,
                "episode_id": self._current_episode_id,
                "env_name": self.env_name,
                "benchmark_name": self.benchmark_name,
                "total_reward": total_reward,
                "success": success,
                "steps": len(self._current_episode_steps),
                "start_time": self._current_episode_start,
                "end_time": datetime.now(timezone.utc),
                "extra_metrics": extra_metrics or {},
            }

            # Wait for all background uploads to complete before sending data
            self._wait_for_uploads(
                timeout=OBS_UPLOAD_TIMEOUT
            )  # 10 second timeout for uploads

            # Queue episode to database if configured
            if self.config.database_url:
                await self._queue_episode_data(episode_data)

                # Queue all step data after episode is queued to avoid race condition
                for step_data in self._current_episode_steps:
                    if step_data.get("should_log", False):
                        await self._queue_step_data(step_data)

        # Reset for next episode
        self._current_episode_id = None
        self._current_episode_steps = []
        self._current_episode_start = None

    def _serialize_action(self, action: Any) -> Dict:
        """Serialize action to JSON-compatible format.

        Args:
            action: Action in various formats

        Returns:
            JSON-serializable dictionary
        """
        if isinstance(action, np.ndarray):
            return {"type": "array", "value": action.tolist(), "shape": action.shape}
        elif isinstance(action, (list, tuple)):
            return {"type": "list", "value": list(action)}
        elif isinstance(action, dict):
            return {"type": "dict", "value": action}
        else:
            return {"type": "scalar", "value": action}

    def _upload_observations_sync(
        self,
        observations: List[Tuple[np.ndarray, str]],
        episode_id: int,
        step: int,
    ) -> Dict[str, Dict[str, str]]:
        """Synchronously upload observations to R2.

        This is called in a background thread by _upload_observations_async.

        Args:
            observations: List of (image, camera_name) tuples
            episode_id: Episode identifier
            step: Step number

        Returns:
            Dictionary mapping camera names to storage references
        """
        refs = {}

        for image, camera_name in observations:
            try:
                result = self._storage_client.upload_observation_image(
                    image=image,
                    submission_id=self.submission_id,
                    task_id=self.task_id,
                    episode_id=episode_id,
                    step=step,
                    camera_name=camera_name,
                    fmt=self.config.image_format,
                )

                refs[camera_name] = {
                    "bucket": result["bucket"],
                    "key": result["key"],
                    "url": result["url"],
                }
            except Exception as e:
                logger.error(
                    f"Failed to upload observation {camera_name} for step {step}: {e}"
                )
                # Continue with other uploads even if one fails

        return refs

    def _upload_observations_async(
        self,
        observations: List[Tuple[np.ndarray, str]],
        episode_id: int,
        step: int,
    ) -> str:
        """Schedule observation uploads in background and return a placeholder key.

        Args:
            observations: List of (image, camera_name) tuples
            episode_id: Episode identifier
            step: Step number

        Returns:
            Placeholder key for the upload future
        """
        if not self._upload_executor:
            # Fallback to empty refs if executor not available
            return None

        # Create a unique key for this upload
        upload_key = f"{episode_id}_{step}"

        # Submit upload task to background thread pool
        future = self._upload_executor.submit(
            self._upload_observations_sync, observations, episode_id, step
        )

        # Store the future so we can retrieve results later
        self._upload_futures.append(future)
        self._pending_uploads[upload_key] = future

        logger.debug(
            f"Scheduled background upload for episode {episode_id}, step {step}"
        )

        return upload_key

    def _wait_for_uploads(self, timeout: float = 30.0) -> None:
        """Wait for all pending uploads to complete and update step data with results.

        Args:
            timeout: Maximum time to wait for uploads in seconds
        """
        if not self._pending_uploads:
            return

        logger.debug(f"Waiting for {len(self._pending_uploads)} uploads to complete")

        # Wait for all uploads to complete with timeout
        try:
            # Get all pending futures
            futures = list(self._pending_uploads.values())

            # Wait for completion with timeout
            done, not_done = wait(futures, timeout=timeout)

            if not_done:
                logger.warning(
                    f"{len(not_done)} uploads did not complete within {timeout}s timeout"
                )

            # Update step data with upload results
            for step_data in self._current_episode_steps:
                upload_key = step_data.get("upload_key")
                if upload_key and upload_key in self._pending_uploads:
                    future = self._pending_uploads[upload_key]
                    if future.done():
                        try:
                            # Get the upload results
                            refs = future.result(timeout=0.1)
                            step_data["observation_refs"] = refs
                            logger.debug(f"Retrieved upload results for {upload_key}")
                        except Exception as e:
                            logger.error(
                                f"Failed to get upload results for {upload_key}: {e}"
                            )
                            step_data["observation_refs"] = {}
                    else:
                        logger.warning(
                            f"Upload {upload_key} not completed, using empty refs"
                        )
                        step_data["observation_refs"] = {}

            # Clear pending uploads
            self._pending_uploads.clear()

        except FutureTimeoutError:
            logger.error(f"Upload wait timed out after {timeout}s")
            # Set empty refs for all incomplete uploads
            for step_data in self._current_episode_steps:
                if step_data.get("upload_key") and not step_data.get(
                    "observation_refs"
                ):
                    step_data["observation_refs"] = {}

    async def _queue_episode_data(self, episode_data: Dict[str, Any]) -> None:
        """Queue episode data to be sent to backend via pgqueuer.

        Args:
            episode_data: Episode data dictionary
        """
        try:
            # Convert datetime objects to ISO format for JSON serialization
            episode_data_copy = episode_data.copy()
            episode_data_copy["start_time"] = episode_data_copy[
                "start_time"
            ].isoformat()
            episode_data_copy["end_time"] = episode_data_copy["end_time"].isoformat()

            # Create message
            episode_msg = EpisodeDataMessage(**episode_data_copy)

            # Queue to pgqueuer
            conn = await asyncpg.connect(dsn=self.config.database_url)
            driver = AsyncpgDriver(conn)
            q = Queries(driver)

            message_json = episode_msg.model_dump_json()
            await q.enqueue(["episode_data"], [message_json.encode("utf-8")], [0])

            await conn.close()
            logger.debug(f"Queued episode {episode_data['episode_id']} for backend")

        except Exception as e:
            logger.error(f"Failed to queue episode data: {e}")

    async def _queue_step_data(self, step_data: Dict[str, Any]) -> None:
        """Queue step data to be sent to backend via pgqueuer.

        Args:
            step_data: Step data dictionary
        """
        try:
            # Convert datetime to ISO format
            step_data_copy = step_data.copy()

            # Remove internal fields that shouldn't be sent
            step_data_copy.pop("upload_key", None)
            step_data_copy.pop("should_log", None)

            # Only process if this step data has full information (not just basic summary)
            if "timestamp" not in step_data_copy:
                logger.error("Step data missing timestamp - skipping queue")
                return

            step_data_copy["step_timestamp"] = step_data_copy["timestamp"].isoformat()
            step_data_copy["submission_id"] = self.submission_id
            step_data_copy["task_id"] = self.task_id
            step_data_copy["episode_id"] = self._current_episode_id

            # Remove the original timestamp key since we renamed it
            del step_data_copy["timestamp"]

            # Create message
            step_msg = EpisodeStepDataMessage(**step_data_copy)

            # Queue to pgqueuer
            conn = await asyncpg.connect(dsn=self.config.database_url)
            driver = AsyncpgDriver(conn)
            q = Queries(driver)

            message_json = step_msg.model_dump_json()
            await q.enqueue(["episode_step_data"], [message_json.encode("utf-8")], [0])

            await conn.close()
            logger.debug(f"Queued step {step_data['step']} for backend")

        except Exception as e:
            logger.error(f"Failed to queue step data: {e}")

    def cleanup(self) -> None:
        """Clean up resources, shutdown upload executor."""
        if self._upload_executor:
            # Wait for any remaining uploads
            self._wait_for_uploads(timeout=OBS_UPLOAD_CLEANUP_TIMEOUT)

            # Shutdown the executor
            self._upload_executor.shutdown(wait=True, cancel_futures=True)
            self._upload_executor = None
            logger.info("Upload executor shutdown complete")

    def get_statistics(self) -> Dict[str, Any]:
        """Get current logging statistics.

        Returns:
            Dictionary with logging statistics
        """
        return {
            "episodes_tracked": self._episode_count,
            "current_episode_id": self._current_episode_id,
            "current_episode_steps": len(self._current_episode_steps),
            "storage_enabled": self._storage_client is not None,
        }
