#!/usr/bin/env python
"""Helper script to run kinitro CLI bypassing bittensor argument hijacking."""

import sys

# Prevent bittensor from hijacking args by importing it first with empty args
original_argv = sys.argv.copy()
sys.argv = [""]
import bittensor as bt  # noqa: F401, E402

sys.argv = original_argv

from kinitro.cli import app  # noqa: E402

if __name__ == "__main__":
    app()
