"""Storb RL evaluator package.

Universal evaluator that supports any agent via the submission format.

Exports:
- agent_interface: ``AgentInterface`` - Interface that all agents must implement
- agent_loader: ``AgentLoader`` - Loads agent submissions dynamically
- envs: ``make_env``, ``EnvSpec`` - Environment creation and configuration
- runner: ``evaluate``, ``EvalConfig`` - Evaluation orchestration
"""

from .agent_interface import AgentInterface
from .agent_loader import AgentLoader
from .envs import EnvSpec, make_env
from .runner import EvalConfig, evaluate

__all__ = [
    # Core evaluation components
    "AgentInterface",
    "AgentLoader", 
    "EnvSpec",
    "make_env",
    "EvalConfig",
    "evaluate",
]
