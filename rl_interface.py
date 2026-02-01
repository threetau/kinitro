"""Compatibility wrapper for template imports."""

from kinitro.rl_interface import (
    CanonicalAction,
    CanonicalObservation,
    CanonicalStep,
    coerce_action,
    normalize_quaternion,
)

__all__ = [
    "CanonicalAction",
    "CanonicalObservation",
    "CanonicalStep",
    "coerce_action",
    "normalize_quaternion",
]
