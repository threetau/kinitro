"""
Evaluation module for running miner policy rollouts.

Provides the core evaluation loop that:
1. Loads miner policies via Basilica containers
2. Runs episodes across all environments
3. Collects success rates and rewards
"""

from kinitro.evaluation.metrics import aggregate_results, compute_success_rate
from kinitro.evaluation.parallel import (
    MinerResult,
    evaluate_all_miners,
    evaluate_miner_on_environment,
)
from kinitro.evaluation.rollout import EpisodeResult, RolloutConfig, run_episode

__all__ = [
    "EpisodeResult",
    "RolloutConfig",
    "run_episode",
    "MinerResult",
    "evaluate_miner_on_environment",
    "evaluate_all_miners",
    "aggregate_results",
    "compute_success_rate",
]
