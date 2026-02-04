"""Dependency injection for API routes."""

import secrets

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from kinitro.api.config import APIConfig
from kinitro.backend.storage import Storage

# Storage instance - set during app startup
_storage_state: dict[str, Storage | None] = {"storage": None}

# Config instance - set during app startup
_config_state: dict[str, APIConfig | None] = {"config": None}

# HTTP Bearer scheme (auto_error=False so we can handle missing auth ourselves)
_bearer_scheme = HTTPBearer(auto_error=False)


def set_storage(storage: Storage) -> None:
    """Set the storage instance for routes."""
    _storage_state["storage"] = storage


def get_storage() -> Storage:
    """Get the storage instance."""
    storage = _storage_state["storage"]
    if storage is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    return storage


async def get_session():
    """Dependency to get database session."""
    storage = get_storage()
    async with storage.session() as session:
        yield session


def set_config(config: APIConfig) -> None:
    """Set the config instance for routes."""
    _config_state["config"] = config


def get_config() -> APIConfig | None:
    """Get the config instance (may be None if not set)."""
    return _config_state["config"]


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
) -> None:
    """
    Verify the API key from the Authorization header.

    Authentication is enabled by default. The request must include a matching
    Bearer token in the Authorization header (from KINITRO_API_API_KEY env var).

    Auth can be disabled by passing --no-auth to the API command.

    Uses secrets.compare_digest to prevent timing attacks.
    """
    config = get_config()

    # If auth is explicitly disabled via --no-auth flag
    if config is not None and config.auth_disabled:
        return

    # Auth is enabled - API key must be configured
    if config is None or config.api_key is None:
        raise HTTPException(
            status_code=500,
            detail="API key not configured. Set KINITRO_API_API_KEY or use --no-auth.",
        )

    # Require authorization header
    if credentials is None:
        raise HTTPException(
            status_code=401,
            detail="Missing authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Compare using constant-time comparison to prevent timing attacks
    if not secrets.compare_digest(credentials.credentials, config.api_key):
        raise HTTPException(
            status_code=401,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
