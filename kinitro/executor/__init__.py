"""Executor service for Kinitro evaluation backend."""

from kinitro.executor.config import ExecutorConfig
from kinitro.executor.main import run_concurrent_executor, run_executor

__all__ = ["ExecutorConfig", "run_executor", "run_concurrent_executor"]
