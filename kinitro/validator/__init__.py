"""
Validator module for the robotics subnet.

Orchestrates the evaluation cycle:
1. Fetch miner commitments
2. Evaluate policies on all environments
3. Compute Pareto scores
4. Set weights on chain
"""

from kinitro.validator.main import Validator

__all__ = ["Validator"]
