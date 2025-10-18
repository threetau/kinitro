"""
Authentication system for Kinitro Backend API.
"""

import hashlib
import secrets
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from sqlalchemy import select

from backend.models import ApiKey


class UserRole(str, Enum):
    """User roles for access control."""

    ADMIN = "admin"
    VALIDATOR = "validator"
    VIEWER = "viewer"


def generate_api_key() -> str:
    """Generate a secure random API key."""
    return secrets.token_urlsafe(32)


def hash_api_key(api_key: str) -> str:
    """Hash an API key for secure storage."""
    return hashlib.sha256(api_key.encode()).hexdigest()


async def get_api_key_from_db(
    api_key: str,
    backend_service,
) -> Optional["ApiKey"]:
    """
    Look up an API key in the database.
    """
    if not backend_service.async_session:
        return None

    # Hash the provided key and look it up
    key_hash = hash_api_key(api_key)

    async with backend_service.async_session() as session:
        result = await session.execute(
            select(ApiKey).where(ApiKey.key_hash == key_hash, ApiKey.is_active)
        )
        api_key_obj = result.scalar_one_or_none()

        if api_key_obj:
            # Check expiration
            if api_key_obj.expires_at and api_key_obj.expires_at < datetime.now(
                timezone.utc
            ):
                return None

            # Update last used timestamp
            api_key_obj.last_used_at = datetime.now(timezone.utc)
            await session.commit()
            await session.refresh(api_key_obj)

        return api_key_obj
