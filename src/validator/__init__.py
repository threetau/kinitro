"""
Kinitro Validator - Polling-based weight setter for Bittensor.

The validator periodically fetches weights from the backend and sets them on chain.
"""

from .config import ValidatorConfig
from .lite_validator import LiteValidator

__all__ = ["ValidatorConfig", "LiteValidator"]
