class Error(Exception):
    """Base exception for neuron errors."""

    pass


class ConfigurationError(Error):
    """Raised when configuration is invalid."""

    pass


class UploadError(Error):
    """Raised when upload operations fail."""

    pass


class CommitmentError(Error):
    """Raised when substrate commitment operations fail."""

    pass


class LocalEvaluationError(Error):
    """Raised when local evaluation sandbox runs fail."""

    pass
