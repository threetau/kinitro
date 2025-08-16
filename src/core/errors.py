class MinerError(Exception):
    """Base exception for miner operations."""

    pass


class ConfigurationError(MinerError):
    """Raised when configuration is invalid."""

    pass


class UploadError(MinerError):
    """Raised when upload operations fail."""

    pass


class CommitmentError(MinerError):
    """Raised when substrate commitment operations fail."""

    pass
