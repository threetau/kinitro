"""Executor service for Kinitro evaluation backend."""

from kinitro.executor.config import ExecutorConfig
from kinitro.executor.main import run_executor

__all__ = ["ExecutorConfig", "run_executor"]
