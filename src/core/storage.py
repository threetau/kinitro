"""
S3 storage client for managing observation data uploads.

This module provides utilities for uploading episode observations
(images, etc.) to S3-compatible storage.
"""

import io
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import boto3
import dotenv
import numpy as np
from botocore.exceptions import ClientError
from PIL import Image

from core.constants import PRESIGN_EXPIRY, ImageFormat

logger = logging.getLogger(__name__)
dotenv.load_dotenv()


@dataclass
class S3Config:
    """Configuration for S3-compatible storage client."""

    endpoint_url: str
    access_key_id: str
    secret_access_key: str
    bucket_name: str
    region: str = "auto"
    public_url_base: Optional[str] = None  # Base URL for public access if configured


def load_s3_config() -> Optional[S3Config]:
    """Load S3 configuration for submission vault from environment variables."""
    endpoint_url = os.environ.get("S3_ENDPOINT_URL")
    access_key_id = os.environ.get("S3_ACCESS_KEY_ID")
    secret_access_key = os.environ.get("S3_SECRET_ACCESS_KEY")
    bucket_name = os.environ.get("S3_BUCKET_NAME")
    region = os.environ.get("S3_REGION", "auto")
    public_url_base = os.environ.get("S3_PUBLIC_URL_BASE")

    if not all([endpoint_url, access_key_id, secret_access_key, bucket_name]):
        logger.warning("S3 credentials missing; direct submission uploads are disabled")
        return None

    return S3Config(
        endpoint_url=endpoint_url,
        access_key_id=access_key_id,
        secret_access_key=secret_access_key,
        bucket_name=bucket_name,
        region=region,
        public_url_base=public_url_base,
    )


class S3StorageClient:
    """Client for uploading observations to S3-compatible storage."""

    def __init__(self, config: S3Config):
        """Initialize S3 storage client.

        Args:
            config: S3 configuration with credentials and bucket info
        """
        self.config = config
        self.bucket_name = config.bucket_name

        # Initialize S3 client with endpoint
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name=config.region,
        )

        # Verify bucket exists
        self._verify_bucket()

    def _verify_bucket(self) -> None:
        """Verify that the configured bucket exists."""
        try:
            self.s3_client.head_bucket(Bucket=self.bucket_name)
            logger.info(
                f"Successfully connected to S3-compatible bucket: {self.bucket_name}"
            )
        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                logger.error(f"Bucket {self.bucket_name} does not exist")
                raise ValueError(f"Bucket {self.bucket_name} does not exist")
            else:
                logger.error(f"Error accessing bucket {self.bucket_name}: {e}")
                raise

    def upload_observation_image(
        self,
        image: np.ndarray,
        submission_id: str,
        task_id: str,
        episode_id: int,
        step: int,
        camera_name: str = "default",
        fmt: ImageFormat = ImageFormat.PNG,
    ) -> Dict[str, str]:
        """Upload a single observation image to S3-compatible storage.

        Args:
            image: Image array in HWC format (Height, Width, Channels)
            submission_id: Unique submission identifier
            task_id: Unique task identifier within the job
            episode_id: Episode number
            step: Step number within episode
            camera_name: Camera view name
            fmt: Image format (png, jpg, etc.)

        Returns:
            Dictionary with upload metadata including key and URL
        """
        # Generate object key with task_id to prevent overwriting between tasks
        key = f"observations/{submission_id}/{task_id}/episode_{episode_id:04d}/step_{step:06d}_{camera_name}.{fmt}"

        try:
            # Convert numpy array to PIL Image
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8)
            pil_image = Image.fromarray(image)

            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format=fmt.upper())
            buffer.seek(0)

            # Upload to S3-compatible storage
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=key,
                Body=buffer,
                ContentType=f"image/{fmt}",
                Metadata={
                    "submission_id": submission_id,
                    "task_id": task_id,
                    "episode_id": str(episode_id),
                    "step": str(step),
                    "camera_name": camera_name,
                },
            )

            # Generate URL
            url = self._generate_url(key)

            return {
                "key": key,
                "url": url,
                "bucket": self.bucket_name,
                "size": buffer.tell(),
                "fmt": fmt,
                "camera_name": camera_name,
            }

        except Exception as e:
            logger.error(f"Failed to upload observation image: {e}")
            raise

    def upload_observation_batch(
        self,
        images: List[Tuple[np.ndarray, str]],
        submission_id: str,
        task_id: str,
        episode_id: int,
        step: int,
        fmt: ImageFormat = ImageFormat.PNG,
    ) -> List[Dict[str, str]]:
        """Upload multiple observation images (e.g., different camera views).

        Args:
            images: List of (image_array, camera_name) tuples
            submission_id: Unique submission identifier
            task_id: Unique task identifier within the job
            episode_id: Episode number
            step: Step number within episode
            fmt: Image format

        Returns:
            List of upload metadata dictionaries
        """
        results = []
        for image, camera_name in images:
            result = self.upload_observation_image(
                image=image,
                submission_id=submission_id,
                task_id=task_id,
                episode_id=episode_id,
                step=step,
                camera_name=camera_name,
                fmt=fmt,
            )
            results.append(result)
        return results

    def _generate_url(self, key: str) -> str:
        """Generate URL for accessing an object.

        Args:
            key: Object key in bucket

        Returns:
            URL string
        """
        if self.config.public_url_base:
            # Use public URL if configured
            return f"{self.config.public_url_base}/{key}"
        else:
            # Generate presigned URL (valid for 7 days)
            return self.s3_client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self.bucket_name, "Key": key},
                ExpiresIn=PRESIGN_EXPIRY,  # 7 days
            )

    def delete_submission_data(self, submission_id: str) -> int:
        """Delete all data for a submission.

        Args:
            submission_id: Submission identifier

        Returns:
            Number of objects deleted
        """
        try:
            # List all observation objects with submission prefix
            prefix = f"observations/{submission_id}/"
            paginator = self.s3_client.get_paginator("list_objects_v2")
            pages = paginator.paginate(Bucket=self.bucket_name, Prefix=prefix)

            objects_to_delete = []
            for page in pages:
                if "Contents" in page:
                    for obj in page["Contents"]:
                        objects_to_delete.append({"Key": obj["Key"]})

            if objects_to_delete:
                # Delete in batches of 1000 (S3 limit)
                for i in range(0, len(objects_to_delete), 1000):
                    batch = objects_to_delete[i : i + 1000]
                    self.s3_client.delete_objects(
                        Bucket=self.bucket_name, Delete={"Objects": batch}
                    )

                logger.info(
                    f"Deleted {len(objects_to_delete)} objects for submission {submission_id}"
                )

            return len(objects_to_delete)

        except Exception as e:
            logger.error(f"Failed to delete submission data: {e}")
            raise
