"""Command-line interface for the robotics subnet."""

import asyncio
from typing import Optional

import typer

app = typer.Typer(
    name="robo",
    help="Robotics Generalization Subnet CLI",
    add_completion=False,
)

# Subcommand group for database management
db_app = typer.Typer(help="Database management commands")
app.add_typer(db_app, name="db")


def _normalize_database_url(database_url: str) -> str:
    """
    Normalize database URL to use asyncpg driver.

    Converts:
    - postgresql://... -> postgresql+asyncpg://...
    - postgres://... -> postgresql+asyncpg://...
    """
    if database_url.startswith("postgresql://"):
        return database_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif database_url.startswith("postgres://"):
        return database_url.replace("postgres://", "postgresql+asyncpg://", 1)
    return database_url


def _parse_database_url(database_url: str) -> tuple[str, str, str, int, str]:
    """
    Parse a PostgreSQL database URL into components.

    Supports formats:
    - postgresql+asyncpg://user:password@host:port/dbname
    - postgresql+asyncpg://user:password@host/dbname (default port 5432)
    - postgresql://... (auto-converted to +asyncpg)
    - postgres://... (auto-converted to +asyncpg)

    Returns:
        Tuple of (user, password, host, port, dbname)
    """
    import re

    # Normalize URL to use asyncpg
    database_url = _normalize_database_url(database_url)

    # Try with explicit port first
    pattern_with_port = r"postgresql\+asyncpg://([^:]+):([^@]+)@([^:/]+):(\d+)/(.+)"
    match = re.match(pattern_with_port, database_url)

    if match:
        user, password, host, port, dbname = match.groups()
        return user, password, host, int(port), dbname

    # Try without port (default to 5432)
    pattern_no_port = r"postgresql\+asyncpg://([^:]+):([^@]+)@([^:/]+)/(.+)"
    match = re.match(pattern_no_port, database_url)

    if match:
        user, password, host, dbname = match.groups()
        return user, password, host, 5432, dbname

    raise ValueError(
        "Invalid database URL format. Expected: postgresql://user:password@host[:port]/dbname"
    )


# =============================================================================
# DATABASE COMMANDS
# =============================================================================


@db_app.command("init")
def db_init(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/robo",
        help="PostgreSQL connection URL",
    ),
):
    """
    Initialize the database schema.

    Creates all tables if they don't exist.
    """
    # Normalize URL to use asyncpg driver
    normalized_url = _normalize_database_url(database_url)

    async def _init():
        from robo.backend.storage import Storage

        storage = Storage(normalized_url)
        await storage.initialize()
        await storage.close()

    typer.echo(f"Initializing database: {database_url.split('@')[-1]}")
    asyncio.run(_init())
    typer.echo("Database initialized successfully!")


@db_app.command("create")
def db_create(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/robo",
        help="PostgreSQL connection URL",
    ),
):
    """
    Create the database if it doesn't exist.

    Connects to the PostgreSQL server and creates the database.
    """
    try:
        user, password, host, port, dbname = _parse_database_url(database_url)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    async def _create():
        import asyncpg

        # Connect to the default 'postgres' database to create our database
        conn = await asyncpg.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database="postgres",
        )

        try:
            # Check if database exists
            exists = await conn.fetchval(
                "SELECT 1 FROM pg_database WHERE datname = $1",
                dbname,
            )

            if exists:
                typer.echo(f"Database '{dbname}' already exists.")
            else:
                # Create the database
                await conn.execute(f'CREATE DATABASE "{dbname}"')
                typer.echo(f"Database '{dbname}' created successfully!")
        finally:
            await conn.close()

    typer.echo(f"Creating database '{dbname}' on {host}:{port}...")
    asyncio.run(_create())


@db_app.command("drop")
def db_drop(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/robo",
        help="PostgreSQL connection URL",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Drop the database.

    WARNING: This will delete all data!
    """
    try:
        user, password, host, port, dbname = _parse_database_url(database_url)
    except ValueError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(
            f"Are you sure you want to drop database '{dbname}'? This cannot be undone!"
        )
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    async def _drop():
        import asyncpg

        conn = await asyncpg.connect(
            user=user,
            password=password,
            host=host,
            port=port,
            database="postgres",
        )

        try:
            # Terminate existing connections
            await conn.execute(f"""
                SELECT pg_terminate_backend(pg_stat_activity.pid)
                FROM pg_stat_activity
                WHERE pg_stat_activity.datname = '{dbname}'
                AND pid <> pg_backend_pid()
            """)

            # Drop the database
            await conn.execute(f'DROP DATABASE IF EXISTS "{dbname}"')
            typer.echo(f"Database '{dbname}' dropped successfully!")
        finally:
            await conn.close()

    typer.echo(f"Dropping database '{dbname}'...")
    asyncio.run(_drop())


@db_app.command("reset")
def db_reset(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/robo",
        help="PostgreSQL connection URL",
    ),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation"),
):
    """
    Reset the database (drop and recreate).

    WARNING: This will delete all data!
    """
    if not force:
        confirm = typer.confirm(
            "Are you sure you want to reset the database? All data will be lost!"
        )
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    # Drop
    db_drop(database_url=database_url, force=True)

    # Create
    db_create(database_url=database_url)

    # Init schema
    db_init(database_url=database_url)

    typer.echo("Database reset complete!")


@db_app.command("status")
def db_status(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/robo",
        help="PostgreSQL connection URL",
    ),
):
    """
    Show database status and statistics.
    """
    # Normalize URL to use asyncpg driver
    normalized_url = _normalize_database_url(database_url)

    async def _status():
        from robo.backend.storage import Storage

        storage = Storage(normalized_url)

        try:
            async with storage.session() as session:
                total_cycles = await storage.count_cycles(session)
                total_miners = await storage.count_unique_miners(session)
                latest_cycle = await storage.get_latest_cycle(session, completed_only=True)
                running_cycle = await storage.get_running_cycle(session)
                latest_weights = await storage.get_latest_weights(session)

                typer.echo("Database Status:")
                typer.echo(f"  Total evaluation cycles: {total_cycles}")
                typer.echo(f"  Unique miners evaluated: {total_miners}")

                if latest_cycle:
                    typer.echo(f"\nLatest completed cycle:")
                    typer.echo(f"  ID: {latest_cycle.id}")
                    typer.echo(f"  Block: {latest_cycle.block_number}")
                    typer.echo(f"  Status: {latest_cycle.status}")
                    typer.echo(f"  Miners: {latest_cycle.n_miners}")
                    typer.echo(
                        f"  Duration: {latest_cycle.duration_seconds:.1f}s"
                        if latest_cycle.duration_seconds
                        else "  Duration: N/A"
                    )
                else:
                    typer.echo("\nNo completed evaluation cycles yet.")

                if running_cycle:
                    typer.echo(f"\nCurrently running cycle:")
                    typer.echo(f"  ID: {running_cycle.id}")
                    typer.echo(f"  Block: {running_cycle.block_number}")
                    typer.echo(f"  Started: {running_cycle.started_at}")

                if latest_weights:
                    typer.echo(f"\nLatest weights:")
                    typer.echo(f"  Block: {latest_weights.block_number}")
                    typer.echo(f"  Miners: {len(latest_weights.weights_json)}")
                    typer.echo(f"  Created: {latest_weights.created_at}")

        except Exception as e:
            typer.echo(f"Error connecting to database: {e}", err=True)
            raise typer.Exit(1)
        finally:
            await storage.close()

    asyncio.run(_status())


# =============================================================================
# BACKEND COMMANDS
# =============================================================================


@app.command()
def backend(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/robo",
        help="PostgreSQL connection URL",
    ),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    eval_interval: int = typer.Option(3600, help="Seconds between evaluation cycles"),
    episodes_per_env: int = typer.Option(50, help="Episodes per environment"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the evaluation backend service.

    This service:
    - Discovers miners from the chain
    - Runs evaluations on robotics environments
    - Computes Pareto-based weights
    - Exposes REST API for validators to fetch weights
    """
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    from robo.backend.app import run_server
    from robo.backend.config import BackendConfig

    # Normalize database URL to use asyncpg driver
    normalized_db_url = _normalize_database_url(database_url)

    config = BackendConfig(
        host=host,
        port=port,
        database_url=normalized_db_url,
        network=network,
        netuid=netuid,
        eval_interval_seconds=eval_interval,
        episodes_per_env=episodes_per_env,
        log_level=log_level,
    )

    typer.echo(f"Starting backend on {host}:{port}")
    typer.echo(f"  Network: {network} (netuid={netuid})")
    typer.echo(f"  Database: {database_url.split('@')[-1]}")
    typer.echo(f"  Eval interval: {eval_interval}s")

    run_server(config)


# =============================================================================
# VALIDATOR COMMANDS
# =============================================================================


@app.command()
def validate(
    backend_url: str = typer.Option(..., help="URL of the evaluation backend"),
    network: str = typer.Option("finney", help="Network: finney, test, or local"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the validator.

    Polls the backend for weights and submits them to the chain.

    The validator is lightweight - all evaluation is done by the backend.
    """
    import structlog

    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    from robo.config import ValidatorConfig
    from robo.validator.main import run_validator

    config = ValidatorConfig(
        network=network,
        netuid=netuid,
        wallet_name=wallet_name,
        hotkey_name=hotkey_name,
        log_level=log_level,
    )

    typer.echo(f"Starting validator on {network} (netuid={netuid})")
    typer.echo(f"  Backend: {backend_url}")
    asyncio.run(run_validator(config, backend_url))


@app.command()
def list_envs():
    """List all available robotics environments."""
    from robo.environments.registry import (
        get_all_environment_ids,
        get_available_families,
        get_environments_by_family,
        is_family_available,
    )

    typer.echo("Available Robotics Environments:\n")

    all_families = ["metaworld", "dm_control", "maniskill"]
    available_families = get_available_families()

    for family in all_families:
        if is_family_available(family):
            envs = get_environments_by_family(family)
            if envs:
                typer.echo(f"  {family.upper()} (installed):")
                for env_id in envs:
                    typer.echo(f"    - {env_id}")
                typer.echo()
        else:
            typer.echo(f"  {family.upper()} (not installed):")
            if family == "dm_control":
                typer.echo("    Install with: pip install robo-subnet[dm-control]")
            elif family == "maniskill":
                typer.echo("    Install with: pip install robo-subnet[maniskill]")
            typer.echo()

    total = len(get_all_environment_ids())
    typer.echo(f"Total: {total} environments available")
    typer.echo(f"Families: {', '.join(available_families)}")


# =============================================================================
# MINER COMMANDS
# =============================================================================


@app.command()
def commit(
    repo: str = typer.Option(..., help="HuggingFace repo (user/model)"),
    revision: str = typer.Option(..., help="Commit SHA"),
    chute_id: str = typer.Option(..., help="Chutes deployment ID"),
    network: str = typer.Option("finney", help="Network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
):
    """
    Commit model to chain (for miners).

    Registers your policy so validators can evaluate it.
    """
    import bittensor as bt

    from robo.chain.commitments import commit_model

    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

    typer.echo(f"Committing model to {network} (netuid={netuid})")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision}")
    typer.echo(f"  Chute ID: {chute_id}")

    success = commit_model(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        repo=repo,
        revision=revision,
        chute_id=chute_id,
    )

    if success:
        typer.echo("Commitment successful!")
    else:
        typer.echo("Commitment failed!", err=True)
        raise typer.Exit(1)


@app.command()
def show_commitment(
    network: str = typer.Option("finney", help="Network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    wallet_name: str = typer.Option("default", help="Wallet name"),
    hotkey_name: str = typer.Option("default", help="Hotkey name"),
    uid: int = typer.Option(None, help="UID to query (if not using wallet)"),
    hotkey: str = typer.Option(None, help="Hotkey address to query (if not using wallet)"),
):
    """
    Show commitment for a miner.
    
    Query by wallet (default), UID, or hotkey address.
    """
    import bittensor as bt
    
    from robo.chain.commitments import _query_commitment_by_hotkey, parse_commitment
    
    subtensor = bt.Subtensor(network=network)
    
    # Determine which hotkey to query
    if hotkey:
        query_hotkey = hotkey
        typer.echo(f"Querying commitment for hotkey: {hotkey[:16]}...")
    elif uid is not None:
        neurons = subtensor.neurons(netuid=netuid)
        if uid >= len(neurons):
            typer.echo(f"UID {uid} not found on subnet {netuid}", err=True)
            raise typer.Exit(1)
        query_hotkey = neurons[uid].hotkey
        typer.echo(f"Querying commitment for UID {uid} ({query_hotkey[:16]}...)")
    else:
        wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)
        query_hotkey = wallet.hotkey.ss58_address
        typer.echo(f"Querying commitment for wallet {wallet_name}/{hotkey_name}")
        typer.echo(f"  Hotkey: {query_hotkey}")
    
    # Query the commitment
    raw = _query_commitment_by_hotkey(subtensor, netuid, query_hotkey)
    
    if not raw:
        typer.echo("\nNo commitment found.")
        raise typer.Exit(0)
    
    typer.echo(f"\nRaw commitment: {raw[:100]}{'...' if len(raw) > 100 else ''}")
    
    # Parse the commitment (supports both JSON and legacy formats)
    parsed = parse_commitment(raw)
    if parsed["huggingface_repo"]:
        typer.echo("\nParsed commitment:")
        typer.echo(f"  Repo: {parsed['huggingface_repo']}")
        typer.echo(f"  Revision: {parsed['revision_sha']}")
        typer.echo(f"  Chute ID: {parsed['chute_id']}")
        if parsed['docker_image']:
            typer.echo(f"  Docker Image: {parsed['docker_image']}")
    else:
        typer.echo("\nCould not parse commitment format.")


@app.command()
def build(
    env_path: str = typer.Argument(..., help="Path to env directory"),
    tag: str = typer.Option(..., help="Docker tag (e.g., user/repo:v1)"),
    push: bool = typer.Option(False, help="Push to registry after building"),
):
    """
    Build miner Docker image.

    Builds a Docker image from your policy directory.
    """
    import subprocess

    typer.echo(f"Building Docker image: {tag}")

    # Build
    result = subprocess.run(
        ["docker", "build", "-t", tag, env_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        typer.echo(f"Build failed:\n{result.stderr}", err=True)
        raise typer.Exit(1)

    typer.echo("Build successful!")

    # Push if requested
    if push:
        typer.echo(f"Pushing to registry: {tag}")
        result = subprocess.run(
            ["docker", "push", tag],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            typer.echo(f"Push failed:\n{result.stderr}", err=True)
            raise typer.Exit(1)

        typer.echo("Push successful!")


@app.command()
def build_eval_env(
    tag: str = typer.Option(
        "robo-subnet/eval-env:v1",
        help="Docker tag for eval environment image",
    ),
    push: bool = typer.Option(False, help="Push to registry after building"),
    registry: Optional[str] = typer.Option(None, help="Registry URL for pushing (e.g., docker.io/myuser)"),
    no_cache: bool = typer.Option(False, help="Build without using cache"),
    quiet: bool = typer.Option(False, help="Suppress build output"),
):
    """
    Build the evaluation environment Docker image using affinetes.

    This image is used by affinetes to run evaluations. It contains:
    - MuJoCo + MetaWorld simulation environment
    - HTTP client for calling miner policy endpoints
    - The robo package for environment management

    The built image is used by the backend scheduler when running evaluations.

    Examples:
        # Build locally
        robo build-eval-env --tag robo-subnet/eval-env:v1

        # Build and push to Docker Hub
        robo build-eval-env --tag eval-env:v1 --push --registry docker.io/myuser
    """
    from pathlib import Path

    import affinetes

    # Find the eval-env directory
    robo_package_dir = Path(__file__).parent
    root_dir = robo_package_dir.parent
    eval_env_path = root_dir / "eval-env"

    if not (eval_env_path / "env.py").exists():
        typer.echo(f"env.py not found at {eval_env_path}", err=True)
        typer.echo("Make sure you're running from within the robo-subnet package.")
        raise typer.Exit(1)

    if not (eval_env_path / "Dockerfile").exists():
        typer.echo(f"Dockerfile not found at {eval_env_path}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Building eval environment image: {tag}")
    typer.echo(f"  Environment path: {eval_env_path}")
    if push:
        typer.echo(f"  Push: True (registry: {registry or 'from tag'})")

    try:
        result_tag = affinetes.build_image_from_env(
            env_path=str(eval_env_path),
            image_tag=tag,
            nocache=no_cache,
            quiet=quiet,
            push=push,
            registry=registry,
        )
        typer.echo(f"\nBuild successful: {result_tag}")

        if push:
            typer.echo(f"Pushed to: {result_tag}")

    except Exception as e:
        typer.echo(f"Build failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\nTo use this image in the backend, ensure your config has:")
    typer.echo(f"  eval_image: {result_tag}")


@app.command()
def init_miner(
    output_dir: str = typer.Argument(".", help="Directory to create template in"),
):
    """
    Initialize a new miner policy from template.

    Creates the necessary files for building a policy container.
    """
    import shutil
    from pathlib import Path

    template_dir = Path(__file__).parent / "miner" / "template"
    output_path = Path(output_dir)

    if not template_dir.exists():
        typer.echo("Template directory not found!", err=True)
        raise typer.Exit(1)

    output_path.mkdir(parents=True, exist_ok=True)

    # Copy template files
    for file in template_dir.iterdir():
        dest = output_path / file.name
        if dest.exists():
            typer.echo(f"Skipping {file.name} (already exists)")
        else:
            shutil.copy(file, dest)
            typer.echo(f"Created {file.name}")

    typer.echo("\nMiner template initialized!")
    typer.echo("Next steps:")
    typer.echo("  1. Edit policy.py to implement your policy")
    typer.echo("  2. Add your model weights to the directory")
    typer.echo("  3. Test locally: uvicorn server:app --port 8001")
    typer.echo("  4. Upload to HuggingFace: huggingface-cli upload user/repo .")
    typer.echo("  5. Deploy to Chutes: robo chutes-push --repo user/repo --revision SHA")
    typer.echo("  6. Commit to chain: robo commit --repo ... --revision ... --chute-id ...")


@app.command("chutes-push")
def chutes_push(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    revision: str = typer.Option(..., "--revision", help="HuggingFace commit SHA"),
    chutes_api_key: Optional[str] = typer.Option(None, "--api-key", envvar="CHUTES_API_KEY", help="Chutes API key"),
    chute_user: Optional[str] = typer.Option(None, "--user", envvar="CHUTE_USER", help="Chutes username"),
    gpu_count: int = typer.Option(1, "--gpu-count", help="Number of GPUs"),
    min_vram_gb: int = typer.Option(8, "--min-vram", help="Minimum VRAM in GB"),
):
    """
    Deploy policy to Chutes.

    Generates a Chute configuration and deploys your policy server.
    Requires CHUTES_API_KEY and CHUTE_USER environment variables.

    Example:
        robo chutes-push --repo user/robo-policy --revision abc123
    """
    import os
    import subprocess
    import textwrap
    from pathlib import Path

    # Validate required credentials
    api_key = chutes_api_key or os.environ.get("CHUTES_API_KEY")
    user = chute_user or os.environ.get("CHUTE_USER")

    if not api_key:
        typer.echo("Error: CHUTES_API_KEY not configured", err=True)
        typer.echo("Set it via --api-key or CHUTES_API_KEY environment variable")
        raise typer.Exit(1)

    if not user:
        typer.echo("Error: CHUTE_USER not configured", err=True)
        typer.echo("Set it via --user or CHUTE_USER environment variable")
        raise typer.Exit(1)

    typer.echo(f"Deploying to Chutes:")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision[:12]}...")
    typer.echo(f"  User: {user}")
    typer.echo(f"  GPU: {gpu_count}x (min {min_vram_gb}GB VRAM)")

    # Generate Chute configuration for robotics policy server
    chute_config = textwrap.dedent(f'''
import os
from chutes import Chute
from chutes.chute import NodeSelector

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = Chute(
    name="{repo.replace("/", "-")}",
    readme="{repo}",
    node_selector=NodeSelector(
        gpu_count={gpu_count},
        min_vram_gb={min_vram_gb},
    ),
)

# The policy server files are loaded from HuggingFace
chute.from_huggingface("{repo}", revision="{revision}")

# Install dependencies
chute.add_pip_requirements("requirements.txt")

# Set the entrypoint for the FastAPI policy server
chute.entrypoint = "uvicorn server:app --host 0.0.0.0 --port 8000"
''')

    tmp_file = Path("tmp_robo_chute.py")
    tmp_file.write_text(chute_config)

    try:
        # Deploy to Chutes
        cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
        env = {**os.environ, "CHUTES_API_KEY": api_key}

        typer.echo("\nDeploying...")
        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            typer.echo("Chutes deployment failed!", err=True)
            raise typer.Exit(1)

        typer.echo("\nDeployment submitted!")
        typer.echo("Check the Chutes dashboard for your chute_id")
        typer.echo("\nNext step:")
        typer.echo(f"  robo commit --repo {repo} --revision {revision} --chute-id YOUR_CHUTE_ID --netuid ...")

    finally:
        tmp_file.unlink(missing_ok=True)


@app.command("miner-deploy")
def miner_deploy(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    policy_path: Optional[str] = typer.Option(None, "--path", "-p", help="Path to local policy directory"),
    revision: Optional[str] = typer.Option(None, "--revision", help="HuggingFace commit SHA (required if --skip-upload)"),
    chute_id: Optional[str] = typer.Option(None, "--chute-id", help="Chutes deployment ID (required if --skip-chutes)"),
    message: str = typer.Option("Model update", "--message", "-m", help="Commit message for HuggingFace upload"),
    skip_upload: bool = typer.Option(False, "--skip-upload", help="Skip HuggingFace upload"),
    skip_chutes: bool = typer.Option(False, "--skip-chutes", help="Skip Chutes deployment"),
    skip_commit: bool = typer.Option(False, "--skip-commit", help="Skip on-chain commit"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Show what would be done without executing"),
    network: str = typer.Option("finney", "--network", help="Bittensor network"),
    netuid: int = typer.Option(..., "--netuid", help="Subnet UID"),
    wallet_name: str = typer.Option("default", "--wallet-name", help="Wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey-name", help="Hotkey name"),
    chutes_api_key: Optional[str] = typer.Option(None, "--chutes-api-key", envvar="CHUTES_API_KEY"),
    chute_user: Optional[str] = typer.Option(None, "--chute-user", envvar="CHUTE_USER"),
    hf_token: Optional[str] = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
):
    """
    One-command deployment: Upload -> Deploy -> Commit.

    Combines the miner deployment workflow into a single command:
    1. Upload policy to HuggingFace (skip with --skip-upload)
    2. Deploy to Chutes (skip with --skip-chutes)
    3. Commit on-chain (skip with --skip-commit)

    Examples:
        # Full deployment
        robo miner-deploy -r user/policy -p ./my-policy --netuid 123

        # Skip upload (already on HuggingFace)
        robo miner-deploy -r user/policy --skip-upload --revision abc123 --netuid 123

        # Dry run to see what would happen
        robo miner-deploy -r user/policy -p ./my-policy --netuid 123 --dry-run
    """
    import json
    import os
    import subprocess
    import textwrap
    from pathlib import Path

    # Validate arguments
    if not skip_upload and not policy_path:
        typer.echo("Error: --path is required unless --skip-upload is set", err=True)
        raise typer.Exit(1)

    if skip_upload and not revision:
        typer.echo("Error: --revision is required when --skip-upload is set", err=True)
        raise typer.Exit(1)

    if skip_chutes and not chute_id:
        typer.echo("Error: --chute-id is required when --skip-chutes is set", err=True)
        raise typer.Exit(1)

    # Get credentials from env if not provided
    api_key = chutes_api_key or os.environ.get("CHUTES_API_KEY")
    user = chute_user or os.environ.get("CHUTE_USER")
    hf = hf_token or os.environ.get("HF_TOKEN")

    # Validate credentials
    if not dry_run:
        if not skip_upload and not hf:
            typer.echo("Error: HF_TOKEN not configured", err=True)
            raise typer.Exit(1)
        if not skip_chutes and not api_key:
            typer.echo("Error: CHUTES_API_KEY not configured", err=True)
            raise typer.Exit(1)
        if not skip_chutes and not user:
            typer.echo("Error: CHUTE_USER not configured", err=True)
            raise typer.Exit(1)

    # Determine steps
    steps = []
    if not skip_upload:
        steps.append("upload")
    if not skip_chutes:
        steps.append("chutes")
    if not skip_commit:
        steps.append("commit")

    typer.echo("=" * 60)
    typer.echo("ROBOTICS SUBNET DEPLOYMENT")
    typer.echo("=" * 60)
    typer.echo(f"  Repository: {repo}")
    if policy_path:
        typer.echo(f"  Policy Path: {policy_path}")
    if revision:
        typer.echo(f"  Revision: {revision}")
    if chute_id:
        typer.echo(f"  Chute ID: {chute_id}")
    typer.echo(f"  Steps: {' -> '.join(steps) if steps else 'none'}")
    if dry_run:
        typer.echo("  Mode: DRY RUN")
    typer.echo("=" * 60)

    # Step 1: Upload to HuggingFace
    if not skip_upload:
        typer.echo(f"\n[1/{len(steps)}] Uploading to HuggingFace ({repo})...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would upload {policy_path} to {repo}")
            revision = "dry-run-revision"
        else:
            try:
                from huggingface_hub import HfApi

                api = HfApi(token=hf)

                # Create repo if it doesn't exist
                try:
                    api.create_repo(repo, exist_ok=True, repo_type="model")
                except Exception:
                    pass  # Repo may already exist

                # Upload folder
                typer.echo(f"  Uploading {policy_path}...")
                api.upload_folder(
                    folder_path=policy_path,
                    repo_id=repo,
                    commit_message=message,
                )

                # Get latest commit SHA
                info = api.repo_info(repo, repo_type="model")
                revision = info.sha

                typer.echo(f"  Upload complete. Revision: {revision[:12]}...")

            except Exception as e:
                typer.echo(f"HuggingFace upload failed: {e}", err=True)
                raise typer.Exit(1)
    else:
        typer.echo(f"\nSkipping upload, using revision: {revision[:12]}...")

    # Step 2: Deploy to Chutes
    if not skip_chutes:
        step_num = 2 if not skip_upload else 1
        typer.echo(f"\n[{step_num}/{len(steps)}] Deploying to Chutes...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would deploy {repo}@{revision[:12]}...")
            chute_id = "dry-run-chute-id"
        else:
            # Generate Chute config
            chute_config = textwrap.dedent(f'''
import os
from chutes import Chute
from chutes.chute import NodeSelector

os.environ["NO_PROXY"] = "localhost,127.0.0.1"

chute = Chute(
    name="{repo.replace("/", "-")}",
    readme="{repo}",
    node_selector=NodeSelector(gpu_count=1, min_vram_gb=8),
)

chute.from_huggingface("{repo}", revision="{revision}")
chute.add_pip_requirements("requirements.txt")
chute.entrypoint = "uvicorn server:app --host 0.0.0.0 --port 8000"
''')

            tmp_file = Path("tmp_robo_chute.py")
            tmp_file.write_text(chute_config)

            try:
                cmd = ["chutes", "deploy", f"{tmp_file.stem}:chute", "--accept-fee"]
                env = {**os.environ, "CHUTES_API_KEY": api_key}

                result = subprocess.run(cmd, env=env, capture_output=True, text=True)

                if result.returncode != 0:
                    typer.echo(f"Chutes deployment failed: {result.stderr}", err=True)
                    raise typer.Exit(1)

                # Try to get chute_id from Chutes API
                try:
                    import aiohttp

                    async def get_chute():
                        async with aiohttp.ClientSession() as session:
                            async with session.get(
                                "https://api.chutes.ai/chutes/",
                                headers={"Authorization": api_key}
                            ) as r:
                                if r.status == 200:
                                    data = await r.json()
                                    chutes = data.get("items", data) if isinstance(data, dict) else data
                                    for c in reversed(chutes):
                                        if c.get("readme") == repo or c.get("model_name") == repo:
                                            return c.get("chute_id")
                        return None

                    chute_id = asyncio.run(get_chute())
                except Exception:
                    chute_id = None

                if chute_id:
                    typer.echo(f"  Chute ID: {chute_id}")
                else:
                    typer.echo("  Deployment submitted. Check Chutes dashboard for chute_id.")
                    typer.echo("  You may need to run commit separately with --chute-id")
                    if not skip_commit:
                        typer.echo("  Skipping commit step (no chute_id)")
                        skip_commit = True

            finally:
                tmp_file.unlink(missing_ok=True)
    else:
        typer.echo(f"\nSkipping Chutes deployment, using chute_id: {chute_id}")

    # Step 3: Commit on-chain
    if not skip_commit and chute_id:
        step_num = len(steps)
        typer.echo(f"\n[{step_num}/{len(steps)}] Committing on-chain...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would commit {repo}@{revision[:12]}... with chute {chute_id}")
        else:
            import bittensor as bt

            from robo.chain.commitments import commit_model

            subtensor = bt.Subtensor(network=network)
            wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

            typer.echo(f"  Wallet: {wallet.hotkey.ss58_address[:16]}...")

            success = commit_model(
                subtensor=subtensor,
                wallet=wallet,
                netuid=netuid,
                repo=repo,
                revision=revision,
                chute_id=chute_id,
            )

            if success:
                typer.echo("  Commit successful!")
            else:
                typer.echo("  Commit failed!", err=True)
                raise typer.Exit(1)

    # Summary
    typer.echo("\n" + "=" * 60)
    if dry_run:
        typer.echo("DRY RUN COMPLETE - No changes were made")
    else:
        typer.echo("DEPLOYMENT COMPLETE")
    typer.echo("=" * 60)
    typer.echo(f"  Repository: {repo}")
    typer.echo(f"  Revision: {revision[:12] if revision else 'N/A'}...")
    typer.echo(f"  Chute ID: {chute_id or 'N/A'}")
    typer.echo("=" * 60)


# =============================================================================
# TESTING COMMANDS
# =============================================================================


@app.command()
def test_env(
    env_id: str = typer.Argument(..., help="Environment ID to test"),
    episodes: int = typer.Option(5, help="Number of episodes to run"),
    render: bool = typer.Option(False, help="Render environment (if supported)"),
):
    """
    Test an environment with random actions.

    Useful for verifying environment setup.
    """
    import numpy as np

    from robo.environments import get_environment

    typer.echo(f"Testing environment: {env_id}")

    env = get_environment(env_id)
    typer.echo(f"  Proprioceptive observation shape: {env.observation_shape}")
    typer.echo(f"  Action shape: {env.action_shape}")

    # Check for camera support
    if hasattr(env, "num_cameras"):
        typer.echo(f"  Number of cameras: {env.num_cameras}")
        if hasattr(env, "image_shape"):
            typer.echo(f"  Image shape: {env.image_shape}")

    successes = 0
    total_reward = 0.0

    for ep in range(episodes):
        task_config = env.generate_task(seed=ep)
        obs = env.reset(task_config)

        typer.echo(f"  Episode {ep + 1} initial obs: {obs}")

        ep_reward = 0.0
        steps = 0

        for _ in range(500):
            # Random action
            low, high = env.action_bounds
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1

            if done:
                break

        success = env.get_success()
        successes += int(success)
        total_reward += ep_reward

        status = "SUCCESS" if success else "FAIL"
        typer.echo(f"  Episode {ep + 1}: {status} | Reward: {ep_reward:.2f} | Steps: {steps}")

    env.close()

    typer.echo(f"\nResults: {successes}/{episodes} successful")
    typer.echo(f"Average reward: {total_reward / episodes:.2f}")


@app.command()
def test_scoring(
    n_miners: int = typer.Option(5, help="Number of simulated miners"),
    n_envs: int = typer.Option(3, help="Number of environments"),
):
    """
    Test the Pareto scoring mechanism with simulated data.
    """
    import numpy as np

    from robo.scoring.pareto import compute_pareto_frontier
    from robo.scoring.winners_take_all import compute_full_scoring

    typer.echo(f"Testing Pareto scoring with {n_miners} miners, {n_envs} environments\n")

    # Generate random scores
    env_ids = [f"env_{i}" for i in range(n_envs)]
    miner_scores = {}

    for uid in range(n_miners):
        miner_scores[uid] = {env_id: float(np.random.uniform(0.3, 0.9)) for env_id in env_ids}

    # Display scores
    typer.echo("Miner scores:")
    for uid, scores in miner_scores.items():
        scores_str = ", ".join(f"{k}: {v:.2f}" for k, v in scores.items())
        typer.echo(f"  UID {uid}: {scores_str}")

    # Compute Pareto frontier
    pareto = compute_pareto_frontier(miner_scores, env_ids, n_samples_per_env=50)
    typer.echo(f"\nPareto frontier: {pareto.frontier_uids}")

    # Compute weights
    weights = compute_full_scoring(miner_scores, env_ids)
    typer.echo("\nFinal weights:")
    for uid, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        typer.echo(f"  UID {uid}: {weight:.4f}")


@app.command("mock-miner")
def mock_miner(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8001, help="Port to bind to"),
    random_actions: bool = typer.Option(True, help="Return random actions (won't solve tasks)"),
):
    """
    Run a mock miner policy server for testing.

    This starts a FastAPI server that implements the miner policy API
    with random actions. Useful for testing the evaluation pipeline
    without a real trained policy.

    Examples:
        # Start mock miner on default port
        robo mock-miner

        # Start on specific port
        robo mock-miner --port 8002
    """
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel
    import numpy as np

    mock_app = FastAPI(title="Mock Miner Policy Server")

    class TaskConfig(BaseModel):
        env_id: str | None = None
        env_name: str | None = None
        task_name: str | None = None
        seed: int | None = None
        task_id: int | None = None

    class ResetRequest(BaseModel):
        task_config: TaskConfig

    class ResetResponse(BaseModel):
        status: str = "ok"
        episode_id: str | None = None

    class ActRequest(BaseModel):
        observation: list[float]
        images: dict[str, list] | None = None

    class ActResponse(BaseModel):
        action: list[float]

    class HealthResponse(BaseModel):
        status: str
        model_loaded: bool
        uptime_seconds: float = 0.0

    import time
    _start_time = time.time()

    @mock_app.get("/health", response_model=HealthResponse)
    async def health():
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            uptime_seconds=time.time() - _start_time,
        )

    @mock_app.post("/reset", response_model=ResetResponse)
    async def reset(request: ResetRequest):
        typer.echo(f"Reset: env_id={request.task_config.env_id}, seed={request.task_config.seed}")
        return ResetResponse(status="ok", episode_id="mock-episode")

    @mock_app.post("/act", response_model=ActResponse)
    async def act(request: ActRequest):
        if random_actions:
            action = np.random.uniform(-1, 1, size=4).tolist()
        else:
            # Zero action (no movement)
            action = [0.0, 0.0, 0.0, 0.0]
        return ActResponse(action=action)

    typer.echo(f"Starting mock miner server on {host}:{port}")
    typer.echo(f"  Random actions: {random_actions}")
    typer.echo(f"  Health: http://{host}:{port}/health")
    typer.echo(f"  Reset:  POST http://{host}:{port}/reset")
    typer.echo(f"  Act:    POST http://{host}:{port}/act")
    typer.echo("\nPress Ctrl+C to stop\n")

    uvicorn.run(mock_app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
