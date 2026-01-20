"""Executor service for Kinitro evaluation backend."""

from kinitro.executor.main import run_executor
from kinitro.executor.config import ExecutorConfig

__all__ = ["run_executor", "ExecutorConfig"]
