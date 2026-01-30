"""Shared utility functions for CLI commands."""


def normalize_database_url(database_url: str) -> str:
    """
    Normalize database URL to use asyncpg driver.

    Converts:
    - postgresql://... -> postgresql+asyncpg://...
    - postgres://... -> postgresql+asyncpg://...
    """
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return database_url


def parse_database_url(database_url: str) -> tuple[str, str, str, int, str]:
    """
    Parse a PostgreSQL database URL into components.

    Supports formats:
    - postgresql+asyncpg://user:password@host:port/dbname
    - postgresql+asyncpg://user:password@host/dbname (default port 5432)
    - postgresql://... (auto-converted to +asyncpg)
    - postgres://... (auto-converted to +asyncpg)

    Returns:
        Tuple of (user, password, host, port, dbname)
    """
    import re

    # Normalize URL to use asyncpg
    database_url = normalize_database_url(database_url)

    # Try with explicit port first
    pattern_with_port = r"postgresql\+asyncpg://([^:]+):([^@]+)@([^:/]+):(\d+)/(.+)"
    match = re.match(pattern_with_port, database_url)

    if match:
        user, password, host, port, dbname = match.groups()
        return user, password, host, int(port), dbname

    # Try without port (default to 5432)
    pattern_no_port = r"postgresql\+asyncpg://([^:]+):([^@]+)@([^:/]+)/(.+)"
    match = re.match(pattern_no_port, database_url)

    if match:
        user, password, host, dbname = match.groups()
        return user, password, host, 5432, dbname

    raise ValueError(
        "Invalid database URL format. Expected: postgresql://user:password@host[:port]/dbname"
    )
