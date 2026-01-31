"""Dependency injection for API routes."""

from fastapi import HTTPException

from kinitro.backend.storage import Storage

# Storage instance - set during app startup
_storage_state: dict[str, Storage | None] = {"storage": None}


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
