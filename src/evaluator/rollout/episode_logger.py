"""
Episode and step logging system with S3 storage integration.

This module provides utilities for logging episode data and step-level
observations to both database and S3 storage with configurable intervals.
"""

import asyncio
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
from core.storage import S3Config, S3StorageClient

logger = logging.getLogger(__name__)

# Upload configuration
OBS_UPLOAD_CLEANUP_TIMEOUT = 5.0
OBS_UPLOAD_TIMEOUT = 20.0
MAX_UPLOAD_WORKERS = 4

# Database connection pool configuration
DB_POOL_MIN_SIZE = 1
DB_POOL_MAX_SIZE = 5
DB_POOL_MAX_QUERIES = 50000
DB_POOL_MAX_INACTIVE_LIFETIME = 300  # seconds
DB_POOL_COMMAND_TIMEOUT = 60  # seconds

# Retry configuration
DB_CONNECTION_MAX_RETRIES = 3
DB_CONNECTION_INITIAL_RETRY_DELAY = 1.0  # seconds
DB_CONNECTION_RETRY_BACKOFF = 2.0  # exponential backoff multiplier
DB_QUEUE_INITIAL_RETRY_DELAY = 0.5  # seconds


@dataclass
class LoggingConfig:
    """Configuration for episode and step logging."""

    # Logging intervals
    episode_log_interval: int = 1  # Log every N episodes
    step_log_interval: int = 10  # Log every N steps within an episode

    # Storage configuration
    enable_s3_upload: bool = True
    s3_config: Optional[S3Config] = None

    # Local storage fallback
    local_save_dir: Optional[Path] = None

    # Image settings
    image_format: ImageFormat = ImageFormat.PNG
    image_quality: int = 95  # For JPEG

    # Database URL for pgqueuer
    database_url: Optional[str] = None


@dataclass
class EpisodeLogger:
    """Logger for episode and step data with S3 integration."""

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
    _storage_client: Optional[S3StorageClient] = field(default=None, init=False)

    # Background upload system
    _upload_executor: Optional[ThreadPoolExecutor] = field(default=None, init=False)
    _upload_futures: List[Any] = field(default_factory=list, init=False)
    _pending_uploads: Dict[str, Any] = field(default_factory=dict, init=False)

    # Database connection pool
    _db_pool: Optional[asyncpg.Pool] = field(default=None, init=False)

    def __post_init__(self):
        """Initialize storage client and upload executor if S3 is enabled."""
        if self.config.enable_s3_upload and self.config.s3_config:
            try:
                self._storage_client = S3StorageClient(self.config.s3_config)
                # Create thread pool for background uploads
                self._upload_executor = ThreadPoolExecutor(
                    max_workers=MAX_UPLOAD_WORKERS, thread_name_prefix="s3_upload"
                )
                logger.info("S3 storage client initialized with background upload pool")
            except Exception as e:
                logger.error(f"Failed to initialize S3 storage: {e}")
                logger.warning("Falling back to local storage only")
                self._storage_client = None
                self._upload_executor = None

        # Initialize database pool asynchronously when needed
        # Pool will be created on first use

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
                "final_reward": final_reward,
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

    def _to_serializable(self, value: Any) -> Any:
        """Recursively normalize common numpy/pydantic types for JSON."""

        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, dict):
            return {str(key): self._to_serializable(val) for key, val in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._to_serializable(item) for item in value]
        if hasattr(value, "model_dump"):
            return self._to_serializable(value.model_dump())
        if hasattr(value, "__dict__"):
            return self._to_serializable(vars(value))
        return str(value)

    def _upload_observations_sync(
        self,
        observations: List[Tuple[np.ndarray, str]],
        episode_id: int,
        step: int,
    ) -> Dict[str, Dict[str, str]]:
        """Synchronously upload observations to S3.

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

        logger.info(f"Waiting for {len(self._pending_uploads)} uploads to complete")

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

    async def _ensure_db_pool(self) -> Optional[asyncpg.Pool]:
        """Ensure database pool is created with retry logic.

        Returns:
            Database connection pool or None if creation fails
        """
        if self._db_pool is None and self.config.database_url:
            retry_delay = DB_CONNECTION_INITIAL_RETRY_DELAY

            for attempt in range(DB_CONNECTION_MAX_RETRIES):
                try:
                    self._db_pool = await asyncpg.create_pool(
                        dsn=self.config.database_url,
                        min_size=DB_POOL_MIN_SIZE,
                        max_size=DB_POOL_MAX_SIZE,
                        max_queries=DB_POOL_MAX_QUERIES,
                        max_inactive_connection_lifetime=DB_POOL_MAX_INACTIVE_LIFETIME,
                        command_timeout=DB_POOL_COMMAND_TIMEOUT,
                        server_settings={
                            "jit": "off"  # Disable JIT to avoid potential issues
                        },
                    )
                    logger.info("Database connection pool created successfully")
                    return self._db_pool
                except Exception as e:
                    logger.warning(
                        f"Failed to create database pool (attempt {attempt + 1}/{DB_CONNECTION_MAX_RETRIES}): {e}"
                    )
                    if attempt < DB_CONNECTION_MAX_RETRIES - 1:
                        await asyncio.sleep(retry_delay)
                        retry_delay *= DB_CONNECTION_RETRY_BACKOFF
                    else:
                        logger.error("Failed to create database pool after all retries")
                        return None

        # Check if existing pool is healthy
        if self._db_pool:
            try:
                async with self._db_pool.acquire() as conn:
                    await conn.fetchval("SELECT 1")
                return self._db_pool
            except Exception as e:
                logger.warning(
                    f"Database pool health check failed: {e}. Recreating pool..."
                )
                await self._close_db_pool()
                self._db_pool = None
                return await self._ensure_db_pool()

        return self._db_pool

    async def _close_db_pool(self) -> None:
        """Close the database connection pool."""
        if self._db_pool:
            try:
                await self._db_pool.close()
                logger.info("Database connection pool closed")
            except Exception as e:
                logger.error(f"Error closing database pool: {e}")
            finally:
                self._db_pool = None

    async def _queue_with_retry(
        self, queue_name: str, message_json: str, max_retries: int = None
    ) -> bool:
        """Queue a message with retry logic.

        Args:
            queue_name: Name of the queue
            message_json: JSON message to queue
            max_retries: Maximum number of retry attempts (defaults to DB_CONNECTION_MAX_RETRIES)

        Returns:
            True if queued successfully, False otherwise
        """
        if max_retries is None:
            max_retries = DB_CONNECTION_MAX_RETRIES

        pool = await self._ensure_db_pool()
        if not pool:
            logger.error("No database pool available for queuing")
            return False

        retry_delay = DB_QUEUE_INITIAL_RETRY_DELAY

        for attempt in range(max_retries):
            try:
                async with pool.acquire() as conn:
                    driver = AsyncpgDriver(conn)
                    q = Queries(driver)
                    await q.enqueue([queue_name], [message_json.encode("utf-8")], [0])
                    return True

            except asyncpg.exceptions.InterfaceError as e:
                logger.warning(
                    f"Connection error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                # Connection is closed, recreate pool
                await self._close_db_pool()
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= DB_CONNECTION_RETRY_BACKOFF

            except Exception as e:
                logger.error(
                    f"Unexpected error queuing message (attempt {attempt + 1}/{max_retries}): {e}"
                )
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                    retry_delay *= DB_CONNECTION_RETRY_BACKOFF
                else:
                    return False

        return False

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

            episode_data_copy = {
                key: self._to_serializable(value)
                for key, value in episode_data_copy.items()
            }

            # Create message
            episode_msg = EpisodeDataMessage(**episode_data_copy)
            message_json = episode_msg.model_dump_json()

            # Queue with retry logic
            success = await self._queue_with_retry("episode_data", message_json)

            if success:
                logger.debug(f"Queued episode {episode_data['episode_id']} for backend")
            else:
                logger.error(
                    f"Failed to queue episode {episode_data['episode_id']} after all retries"
                )

        except Exception as e:
            logger.error(f"Failed to prepare episode data for queuing: {e}")

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

            step_data_copy = {
                key: self._to_serializable(value)
                for key, value in step_data_copy.items()
            }

            # Create message
            step_msg = EpisodeStepDataMessage(**step_data_copy)
            message_json = step_msg.model_dump_json()

            # Queue with retry logic
            success = await self._queue_with_retry("episode_step_data", message_json)

            if success:
                logger.debug(f"Queued step {step_data['step']} for backend")
            else:
                logger.error(
                    f"Failed to queue step {step_data['step']} after all retries"
                )

        except Exception as e:
            logger.error(f"Failed to prepare step data for queuing: {e}")

    def cleanup(self) -> None:
        """Clean up resources, shutdown upload executor."""
        if self._upload_executor:
            # Wait for any remaining uploads
            self._wait_for_uploads(timeout=OBS_UPLOAD_CLEANUP_TIMEOUT)

            # Shutdown the executor
            self._upload_executor.shutdown(wait=True, cancel_futures=True)
            self._upload_executor = None
            logger.info("Upload executor shutdown complete")

        # Close database pool
        if self._db_pool:
            # Create a new event loop if needed for async cleanup
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.run_until_complete(self._close_db_pool())
            except Exception as e:
                logger.error(f"Error during database pool cleanup: {e}")

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
