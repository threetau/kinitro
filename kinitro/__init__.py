"""
Kinitro - A Bittensor subnet for evaluating generalist robotics policies.

Miners submit RL policies that are evaluated across diverse robotics environments
(manipulation, locomotion, dexterous). Only policies that generalize across ALL
environments earn rewards via ε-Pareto dominance scoring.
"""

import os

# https://docs.learnbittensor.org/sdk/migration-guide#disabling-cli-argument-parsing
os.environ.setdefault("BT_NO_PARSE_CLI_ARGS", "1")

__version__ = "0.1.0"
