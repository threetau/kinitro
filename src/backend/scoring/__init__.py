"""
Scoring strategy abstraction for Kinitro.

This package provides pluggable scoring strategies that allow different
task types to have their own eligibility, metrics extraction, and scoring logic.

It also re-exports the ScoringEngine and ScoringConfig for backward compatibility.
"""

from .registry import ScoringStrategyRegistry
from .strategies import (
    EligibilityResult,
    RLRolloutScoringStrategy,
    ScoringMetrics,
    ScoringStrategy,
    StrategyNotFoundError,
)

# Re-export from the scoring_engine module for backward compatibility
# The scoring_engine module was previously named scoring.py but was renamed
# to avoid conflicts with this package
from backend.scoring_engine import ScoringConfig, ScoringEngine

__all__ = [
    # Strategy abstractions
    "EligibilityResult",
    "RLRolloutScoringStrategy",
    "ScoringMetrics",
    "ScoringStrategy",
    "ScoringStrategyRegistry",
    "StrategyNotFoundError",
    # Backward compatible re-exports
    "ScoringConfig",
    "ScoringEngine",
]
