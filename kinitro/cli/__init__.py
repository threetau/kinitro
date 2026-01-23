"""CLI commands for Kinitro."""

import typer

from .api import api
from .build import build
from .build_eval_env import build_eval_env
from .chutes_push import chutes_push
from .commit import commit
from .db_create import db_create
from .db_drop import db_drop
from .db_init import db_init
from .db_reset import db_reset
from .db_status import db_status
from .executor import executor
from .init_miner import init_miner
from .list_envs import list_envs
from .miner_deploy import miner_deploy
from .mock_miner import mock_miner
from .scheduler import scheduler
from .show_commitment import show_commitment
from .test_env import test_env
from .test_scoring import test_scoring
from .utils import normalize_database_url, parse_database_url
from .validate import validate

# Create main app
app = typer.Typer(
    name="kinitro",
    help="Kinitro: The Robotics Generalization Subnet",
    add_completion=False,
    no_args_is_help=True,
)

# Database commands subgroup
db_app = typer.Typer(help="Database management commands")
app.add_typer(db_app, name="db")

# Register database commands
db_app.command("init")(db_init)
db_app.command("create")(db_create)
db_app.command("drop")(db_drop)
db_app.command("reset")(db_reset)
db_app.command("status")(db_status)

# Register service commands
app.command()(api)
app.command()(scheduler)
app.command()(executor)

# Register validator commands
app.command()(validate)
app.command()(list_envs)

# Register miner commands
app.command()(commit)
app.command()(show_commitment)
app.command()(build)
app.command()(build_eval_env)
app.command()(init_miner)
app.command("chutes-push")(chutes_push)
app.command("miner-deploy")(miner_deploy)

# Register testing commands
app.command()(test_env)
app.command()(test_scoring)
app.command("mock-miner")(mock_miner)

__all__ = [
    "api",
    "app",
    "build",
    "build_eval_env",
    "chutes_push",
    "commit",
    "db_create",
    "db_drop",
    "db_init",
    "db_reset",
    "db_status",
    "executor",
    "init_miner",
    "list_envs",
    "miner_deploy",
    "mock_miner",
    "normalize_database_url",
    "parse_database_url",
    "scheduler",
    "show_commitment",
    "test_env",
    "test_scoring",
    "validate",
]
