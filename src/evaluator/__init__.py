"""kinitro evaluator package.

Exports:
- agent_interface: ``AgentInterface`` - Interface that all agents must implement
- envs: ``EnvSpec``, ``EnvManager`` - Environment configuration and management
- database: ``DatabaseManager`` - PostgreSQL database management
"""

from core.db import DatabaseManager

from .agent_interface import AgentInterface
from .roullout.envs import EnvManager, EnvSpec

__all__ = [
    # Core evaluation components
    "AgentInterface",
    "EnvSpec",
    "EnvManager",
    # Database components
    "DatabaseManager",
]
