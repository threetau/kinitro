"""
Submission storage helpers for the direct vault upload workflow.

This module provides thin wrappers around an S3-compatible (e.g. Cloudflare R2)
bucket so the backend can mint presigned upload/download URLs, check artifact
metadata, and manage lifecycle events for miner submissions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

import boto3
from botocore.exceptions import ClientError

from core.log import get_logger
from core.storage import S3Config

logger = get_logger(__name__)


@dataclass
class PresignedUpload:
    """Information about a presigned upload URL."""

    url: str
    method: str = "PUT"
    expires_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc) + timedelta(minutes=10)
    )
    headers: Dict[str, str] = field(default_factory=dict)


@dataclass
class ArtifactMetadata:
    """Metadata for an uploaded submission artifact."""

    object_key: str
    size_bytes: int
    last_modified: datetime
    etag: Optional[str] = None
    sha256: Optional[str] = None


class SubmissionStorage:
    """Manage submission artifacts stored in S3-compatible storage."""

    def __init__(self, config: S3Config, prefix: str = "submissions") -> None:
        self.config = config
        self.bucket_name = config.bucket_name
        self.prefix = prefix.rstrip("/")

        self._s3_client = boto3.client(
            "s3",
            endpoint_url=config.endpoint_url,
            aws_access_key_id=config.access_key_id,
            aws_secret_access_key=config.secret_access_key,
            region_name=config.region,
        )

    def build_object_key(self, submission_id: int, filename: str) -> str:
        safe_filename = filename.replace("..", "").lstrip("/")
        return f"{self.prefix}/{submission_id}/{safe_filename}"

    def create_presigned_upload(
        self,
        object_key: str,
        *,
        expires_in: int = 600,
        content_type: str = "application/gzip",
    ) -> PresignedUpload:
        """Generate a presigned PUT URL for uploading an artifact."""
        try:
            url = self._s3_client.generate_presigned_url(
                ClientMethod="put_object",
                Params={
                    "Bucket": self.bucket_name,
                    "Key": object_key,
                    "ContentType": content_type,
                },
                ExpiresIn=expires_in,
            )
        except Exception:
            logger.exception("Failed to generate presigned upload for %s", object_key)
            raise

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        return PresignedUpload(
            url=url,
            method="PUT",
            expires_at=expires_at,
            headers={"Content-Type": content_type},
        )

    def generate_download_url(
        self, object_key: str, expires_in: int
    ) -> tuple[str, datetime]:
        """Generate a presigned GET URL for downloading an artifact."""
        try:
            url = self._s3_client.generate_presigned_url(
                ClientMethod="get_object",
                Params={"Bucket": self.bucket_name, "Key": object_key},
                ExpiresIn=expires_in,
            )
        except Exception:
            logger.exception("Failed to generate download URL for %s", object_key)
            raise

        expires_at = datetime.now(timezone.utc) + timedelta(seconds=expires_in)
        return url, expires_at

    def head_object(self, object_key: str) -> Optional[ArtifactMetadata]:
        """Retrieve metadata for an uploaded artifact, if it exists."""
        try:
            response = self._s3_client.head_object(
                Bucket=self.bucket_name, Key=object_key
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] in {"404", "NoSuchKey", "NotFound"}:
                return None
            logger.exception("Failed to head object %s", object_key)
            raise

        metadata = response.get("Metadata", {})
        sha256 = metadata.get("sha256")
        etag = response.get("ETag")
        last_modified = response.get("LastModified")
        if isinstance(last_modified, datetime):
            last_modified = last_modified.astimezone(timezone.utc)
        else:
            last_modified = datetime.now(timezone.utc)

        return ArtifactMetadata(
            object_key=object_key,
            size_bytes=response.get("ContentLength", 0),
            last_modified=last_modified,
            etag=etag,
            sha256=sha256,
        )

    def delete_object(self, object_key: str) -> None:
        """Delete an artifact from storage."""
        try:
            self._s3_client.delete_object(Bucket=self.bucket_name, Key=object_key)
        except Exception:
            logger.exception("Failed to delete object %s", object_key)
            raise

    def public_url(self, object_key: str) -> Optional[str]:
        """Return a public URL for the artifact if a public base is configured."""
        if self.config.public_url_base:
            return f"{self.config.public_url_base.rstrip('/')}/{object_key}"
        return None
