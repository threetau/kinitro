"""Command-line interface for the robotics subnet."""

import asyncio

import typer

app = typer.Typer(
    name="kinitro",
    help="Kinitro - Robotics Generalization Subnet CLI",
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
        "postgresql://postgres:postgres@localhost:5432/kinitro",
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
        from kinitro.backend.storage import Storage

        storage = Storage(normalized_url)
        await storage.initialize()
        await storage.close()

    typer.echo(f"Initializing database: {database_url.split('@')[-1]}")
    asyncio.run(_init())
    typer.echo("Database initialized successfully!")


@db_app.command("create")
def db_create(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
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
        "postgresql://postgres:postgres@localhost:5432/kinitro",
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
        "postgresql://postgres:postgres@localhost:5432/kinitro",
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
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
):
    """
    Show database status and statistics.
    """
    # Normalize URL to use asyncpg driver
    normalized_url = _normalize_database_url(database_url)

    async def _status():
        from kinitro.backend.storage import Storage

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
                    typer.echo("\nLatest completed cycle:")
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
                    typer.echo("\nCurrently running cycle:")
                    typer.echo(f"  ID: {running_cycle.id}")
                    typer.echo(f"  Block: {running_cycle.block_number}")
                    typer.echo(f"  Started: {running_cycle.started_at}")

                if latest_weights:
                    typer.echo("\nLatest weights:")
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
# SERVICE COMMANDS (api, scheduler, executor)
# =============================================================================


@app.command()
def api(
    host: str = typer.Option("0.0.0.0", help="Host to bind to"),
    port: int = typer.Option(8000, help="Port to bind to"),
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the API service (lightweight REST API).

    The API service provides:
    - Task pool endpoints for executors (fetch/submit)
    - Weights endpoint for validators
    - Scores and status endpoints

    This is the stateless API layer of the split architecture.
    Run separately from scheduler and executor.
    """
    import structlog

    log_level = log_level.upper()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    from kinitro.api import APIConfig, run_server

    normalized_db_url = _normalize_database_url(database_url)

    config = APIConfig(
        host=host,
        port=port,
        database_url=normalized_db_url,
        log_level=log_level,
    )

    typer.echo(f"Starting API service on {host}:{port}")
    typer.echo(f"  Database: {database_url.split('@')[-1]}")

    run_server(config)


@app.command()
def scheduler(
    database_url: str = typer.Option(
        "postgresql://postgres:postgres@localhost:5432/kinitro",
        help="PostgreSQL connection URL",
    ),
    network: str = typer.Option("finney", help="Bittensor network"),
    netuid: int = typer.Option(..., help="Subnet UID"),
    eval_interval: int = typer.Option(3600, help="Seconds between evaluation cycles"),
    episodes_per_env: int = typer.Option(50, help="Episodes per environment"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the scheduler service.

    The scheduler is responsible for:
    - Reading miner commitments from the chain
    - Creating evaluation tasks in the task pool
    - Waiting for executors to complete tasks
    - Computing Pareto scores and weights
    - Storing results in the database

    Requires the API service to be running for executors to fetch tasks.
    """
    import structlog

    log_level = log_level.upper()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    from kinitro.scheduler import SchedulerConfig, run_scheduler

    normalized_db_url = _normalize_database_url(database_url)

    config = SchedulerConfig(
        database_url=normalized_db_url,
        network=network,
        netuid=netuid,
        eval_interval_seconds=eval_interval,
        episodes_per_env=episodes_per_env,
        log_level=log_level,
    )

    typer.echo("Starting scheduler service")
    typer.echo(f"  Network: {network} (netuid={netuid})")
    typer.echo(f"  Database: {database_url.split('@')[-1]}")
    typer.echo(f"  Eval interval: {eval_interval}s")

    asyncio.run(run_scheduler(config))


@app.command()
def executor(
    api_url: str = typer.Option(
        "http://localhost:8000",
        help="URL of the Kinitro API service",
    ),
    executor_id: str | None = typer.Option(
        None,
        help="Unique ID for this executor (auto-generated if not provided)",
    ),
    batch_size: int = typer.Option(10, help="Number of tasks to fetch at a time"),
    poll_interval: int = typer.Option(5, help="Seconds between polling for tasks"),
    eval_image: str = typer.Option(
        "kinitro/eval-env:v1",
        help="Docker image for evaluation environment",
    ),
    eval_mode: str = typer.Option("docker", help="Evaluation mode: docker or basilica"),
    log_level: str = typer.Option("INFO", help="Logging level"),
):
    """
    Run the executor service.

    The executor fetches tasks from the API and runs MuJoCo evaluations
    using affinetes. It calls miner policy endpoints to get actions.

    Multiple executors can run in parallel on different GPU machines
    to scale evaluation throughput.

    Examples:
        # Run with default settings
        kinitro executor --api-url http://api.kinitro.io:8000

        # Run on specific GPU with custom batch size
        CUDA_VISIBLE_DEVICES=0 kinitro executor --batch-size 20
    """
    import structlog

    log_level = log_level.upper()
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
    )

    from kinitro.executor import ExecutorConfig, run_executor

    config_kwargs = {
        "api_url": api_url,
        "batch_size": batch_size,
        "poll_interval_seconds": poll_interval,
        "eval_image": eval_image,
        "eval_mode": eval_mode,
        "log_level": log_level,
    }

    if executor_id:
        config_kwargs["executor_id"] = executor_id

    config = ExecutorConfig(**config_kwargs)

    typer.echo("Starting executor service")
    typer.echo(f"  Executor ID: {config.executor_id}")
    typer.echo(f"  API URL: {api_url}")
    typer.echo(f"  Batch size: {batch_size}")
    typer.echo(f"  Eval image: {eval_image}")

    asyncio.run(run_executor(config))


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

    from kinitro.config import ValidatorConfig
    from kinitro.validator.main import run_validator

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
    from kinitro.environments.registry import get_all_environment_ids

    typer.echo("Available Robotics Environments:\n")

    typer.echo("  METAWORLD (Manipulation):")
    for env_id in get_all_environment_ids():
        typer.echo(f"    - {env_id}")

    typer.echo()
    total = len(get_all_environment_ids())
    typer.echo(f"Total: {total} environments available")


# =============================================================================
# MINER COMMANDS
# =============================================================================


@app.command()
def commit(
    repo: str = typer.Option(..., help="HuggingFace repo (user/model)"),
    revision: str = typer.Option(..., help="Commit SHA"),
    deployment_id: str = typer.Option(
        ..., "--deployment-id", "-d", help="Basilica deployment ID (UUID only)"
    ),
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

    from kinitro.chain.commitments import commit_model

    subtensor = bt.Subtensor(network=network)
    wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

    typer.echo(f"Committing model to {network} (netuid={netuid})")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision}")
    typer.echo(f"  Deployment ID: {deployment_id}")

    success = commit_model(
        subtensor=subtensor,
        wallet=wallet,
        netuid=netuid,
        repo=repo,
        revision=revision,
        deployment_id=deployment_id,
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

    from kinitro.chain.commitments import _query_commitment_by_hotkey, parse_commitment

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
        typer.echo(f"  Deployment ID: {parsed['deployment_id']}")
        if parsed["docker_image"]:
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
        "kinitro/eval-env:v1",
        help="Docker tag for eval environment image",
    ),
    push: bool = typer.Option(False, help="Push to registry after building"),
    registry: str | None = typer.Option(
        None, help="Registry URL for pushing (e.g., docker.io/myuser)"
    ),
    no_cache: bool = typer.Option(False, help="Build without using cache"),
    quiet: bool = typer.Option(False, help="Suppress build output"),
):
    """
    Build the evaluation environment Docker image using affinetes.

    This image is used by affinetes to run evaluations. It contains:
    - MuJoCo + MetaWorld simulation environment
    - HTTP client for calling miner policy endpoints
    - The kinitro environments module

    The built image is used by the backend scheduler when running evaluations.

    Examples:
        # Build locally
        kinitro build-eval-env --tag kinitro/eval-env:v1

        # Build and push to Docker Hub
        kinitro build-eval-env --tag eval-env:v1 --push --registry docker.io/myuser
    """
    import shutil
    from pathlib import Path

    import affinetes

    # Find the eval-env directory and kinitro package
    kinitro_package_dir = Path(__file__).parent
    root_dir = kinitro_package_dir.parent
    eval_env_path = root_dir / "eval-env"
    environments_src = kinitro_package_dir / "environments"

    if not (eval_env_path / "env.py").exists():
        typer.echo(f"env.py not found at {eval_env_path}", err=True)
        typer.echo("Make sure you're running from within the kinitro package.")
        raise typer.Exit(1)

    if not (eval_env_path / "Dockerfile").exists():
        typer.echo(f"Dockerfile not found at {eval_env_path}", err=True)
        raise typer.Exit(1)

    if not environments_src.exists():
        typer.echo(f"environments module not found at {environments_src}", err=True)
        raise typer.Exit(1)

    typer.echo(f"Building eval environment image: {tag}")
    typer.echo(f"  Environment path: {eval_env_path}")
    if push:
        typer.echo(f"  Push: True (registry: {registry or 'from tag'})")

    # Copy kinitro/environments to eval-env/kinitro/environments for the build
    # This avoids duplicating the code in the repo
    eval_env_kinitro = eval_env_path / "kinitro"
    eval_env_environments = eval_env_kinitro / "environments"

    try:
        # Create kinitro package structure in eval-env
        eval_env_kinitro.mkdir(exist_ok=True)

        # Create __init__.py for kinitro package
        (eval_env_kinitro / "__init__.py").write_text(
            '"""Kinitro package subset for eval environment."""\n'
        )

        # Copy environments module
        if eval_env_environments.exists():
            shutil.rmtree(eval_env_environments)
        shutil.copytree(
            environments_src,
            eval_env_environments,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        typer.echo("  Copied environments module to build context")

        # Build the image
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

    finally:
        # Clean up the copied kinitro directory
        if eval_env_kinitro.exists():
            shutil.rmtree(eval_env_kinitro)
            typer.echo("  Cleaned up temporary build files")

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

    # Copy template files (skip directories and pycache)
    for file in template_dir.iterdir():
        # Skip directories and pycache
        if file.is_dir() or file.name.startswith("__"):
            continue
        # Skip .pyc files
        if file.suffix == ".pyc":
            continue

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
    typer.echo("  5. Deploy to Basilica: kinitro basilica-push --repo user/repo --revision SHA")
    typer.echo(
        "  6. Or use one-command deploy: kinitro miner-deploy -r user/repo -p . --netuid ..."
    )


@app.command("basilica-push")
def basilica_push(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    revision: str = typer.Option(..., "--revision", help="HuggingFace commit SHA"),
    deployment_name: str | None = typer.Option(
        None, "--name", "-n", help="Deployment name (default: derived from repo)"
    ),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Number of GPUs (0 for CPU-only)"),
    min_gpu_memory_gb: int | None = typer.Option(None, "--min-vram", help="Minimum GPU VRAM in GB"),
    memory: str = typer.Option("512Mi", "--memory", help="Memory allocation"),
    basilica_api_token: str | None = typer.Option(
        None, "--api-token", envvar="BASILICA_API_TOKEN", help="Basilica API token"
    ),
    hf_token: str | None = typer.Option(
        None, "--hf-token", envvar="HF_TOKEN", help="HuggingFace token for private repos"
    ),
    timeout: int = typer.Option(600, "--timeout", help="Deployment timeout in seconds"),
):
    """
    Deploy policy to Basilica.

    Deploys your robotics policy server to Basilica's GPU serverless platform.
    The policy is downloaded from HuggingFace and served via FastAPI.

    Requires BASILICA_API_TOKEN environment variable or --api-token.

    Example:
        kinitro basilica-push --repo user/policy --revision abc123

        # With custom name
        kinitro basilica-push --repo user/policy --revision abc123 --name my-policy

        # With more GPU memory
        kinitro basilica-push --repo user/policy --revision abc123 --min-vram 24
    """
    import os

    # Validate required credentials
    api_token = basilica_api_token or os.environ.get("BASILICA_API_TOKEN")

    if not api_token:
        typer.echo("Error: BASILICA_API_TOKEN not configured", err=True)
        typer.echo("Set it via --api-token or BASILICA_API_TOKEN environment variable")
        typer.echo("\nTo get a token, run: basilica tokens create")
        raise typer.Exit(1)

    # Import Basilica SDK
    try:
        from basilica import BasilicaClient
    except ImportError:
        typer.echo("Error: basilica-sdk not installed", err=True)
        typer.echo("Run: pip install basilica-sdk")
        raise typer.Exit(1)

    # Derive deployment name from repo
    name = deployment_name or repo.replace("/", "-").lower()
    # Ensure name is DNS-safe (lowercase, alphanumeric and hyphens only)
    name = "".join(c if c.isalnum() or c == "-" else "-" for c in name).strip("-")[:63]

    typer.echo("Deploying to Basilica:")
    typer.echo(f"  Repo: {repo}")
    typer.echo(f"  Revision: {revision[:12]}...")
    typer.echo(f"  Deployment Name: {name}")
    vram_str = f" (min {min_gpu_memory_gb}GB VRAM)" if min_gpu_memory_gb else ""
    typer.echo(f"  GPU: {gpu_count}x{vram_str}")
    typer.echo(f"  Memory: {memory}")

    # Create client
    client = BasilicaClient(api_key=api_token)

    # Generate deployment source code
    hf_token_str = hf_token or ""
    source_code = f"""
import os
import sys
import subprocess

print("Starting Kinitro Policy Server...")
print(f"HF_REPO: {{os.environ.get('HF_REPO', 'not set')}}")
print(f"HF_REVISION: {{os.environ.get('HF_REVISION', 'not set')}}")

# Download model from HuggingFace
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN") or None
print("Downloading model from HuggingFace...")
snapshot_download(
    "{repo}",
    revision="{revision}",
    local_dir="/app",
    token=hf_token,
)
print("Model downloaded successfully!")

# Change to /app directory and add to Python path
os.chdir("/app")
sys.path.insert(0, "/app")

# Start the FastAPI server from /app directory
print("Starting uvicorn server on port 8000...")
subprocess.run([
    sys.executable, "-m", "uvicorn",
    "server:app",
    "--host", "0.0.0.0",
    "--port", "8000",
], cwd="/app")
"""

    # Choose image based on GPU requirement
    if gpu_count > 0:
        image = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    else:
        image = "python:3.11-slim"

    # Build deployment configuration
    deploy_kwargs = {
        "name": name,
        "source": source_code,
        "image": image,
        "port": 8000,
        "memory": memory,
        "cpu": "500m",
        "pip_packages": [
            "fastapi",
            "uvicorn",
            "numpy",
            "huggingface-hub",
            "pydantic",
            "pillow",
        ],
        "env": {
            "HF_REPO": repo,
            "HF_REVISION": revision,
        },
        "timeout": timeout,
    }

    # Add GPU settings only if GPU is requested
    if gpu_count > 0:
        deploy_kwargs["gpu_count"] = gpu_count
        if min_gpu_memory_gb:
            deploy_kwargs["min_gpu_memory_gb"] = min_gpu_memory_gb

    # Add HF token if provided
    if hf_token_str:
        deploy_kwargs["env"]["HF_TOKEN"] = hf_token_str

    # Deploy
    typer.echo("\nDeploying to Basilica (this may take several minutes)...")
    try:
        deployment = client.deploy(**deploy_kwargs)
    except Exception as e:
        typer.echo(f"\nDeployment failed: {e}", err=True)
        raise typer.Exit(1)

    typer.echo("\n" + "=" * 60)
    typer.echo("DEPLOYMENT SUCCESSFUL")
    typer.echo("=" * 60)
    typer.echo(f"  Name: {deployment.name}")
    typer.echo(f"  URL: {deployment.url}")
    typer.echo(f"  State: {deployment.state}")
    typer.echo("=" * 60)
    # Extract deployment ID from URL
    deploy_id = deployment.url.split("//")[1].split(".")[0] if deployment.url else deployment.name
    typer.echo("\nNext step - commit on-chain:")
    typer.echo(f"  kinitro commit --repo {repo} --revision {revision} \\")
    typer.echo(f"    --deployment-id {deploy_id} --netuid YOUR_NETUID")


@app.command("miner-deploy")
def miner_deploy(
    repo: str = typer.Option(..., "--repo", "-r", help="HuggingFace repository ID"),
    policy_path: str | None = typer.Option(
        None, "--path", "-p", help="Path to local policy directory"
    ),
    revision: str | None = typer.Option(
        None, "--revision", help="HuggingFace commit SHA (required if --skip-upload)"
    ),
    deployment_id: str | None = typer.Option(
        None, "--deployment-id", "-d", help="Basilica deployment ID (required if --skip-deploy)"
    ),
    deployment_name: str | None = typer.Option(
        None, "--name", "-n", help="Deployment name (default: derived from repo)"
    ),
    message: str = typer.Option(
        "Model update", "--message", "-m", help="Commit message for HuggingFace upload"
    ),
    skip_upload: bool = typer.Option(False, "--skip-upload", help="Skip HuggingFace upload"),
    skip_deploy: bool = typer.Option(False, "--skip-deploy", help="Skip Basilica deployment"),
    skip_commit: bool = typer.Option(False, "--skip-commit", help="Skip on-chain commit"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Show what would be done without executing"
    ),
    network: str = typer.Option("finney", "--network", help="Bittensor network"),
    netuid: int = typer.Option(..., "--netuid", help="Subnet UID"),
    wallet_name: str = typer.Option("default", "--wallet-name", help="Wallet name"),
    hotkey_name: str = typer.Option("default", "--hotkey-name", help="Hotkey name"),
    basilica_api_token: str | None = typer.Option(
        None, "--api-token", envvar="BASILICA_API_TOKEN", help="Basilica API token"
    ),
    hf_token: str | None = typer.Option(None, "--hf-token", envvar="HF_TOKEN"),
    gpu_count: int = typer.Option(0, "--gpu-count", help="Number of GPUs (0 for CPU-only)"),
    min_gpu_memory_gb: int | None = typer.Option(None, "--min-vram", help="Minimum GPU VRAM in GB"),
    memory: str = typer.Option("512Mi", "--memory", help="Memory allocation"),
):
    """
    One-command deployment: Upload -> Deploy -> Commit.

    Combines the miner deployment workflow into a single command:
    1. Upload policy to HuggingFace (skip with --skip-upload)
    2. Deploy to Basilica (skip with --skip-deploy)
    3. Commit on-chain (skip with --skip-commit)

    Examples:
        # Full deployment
        kinitro miner-deploy -r user/policy -p ./my-policy --netuid 123

        # Skip upload (already on HuggingFace)
        kinitro miner-deploy -r user/policy --skip-upload --revision abc123 --netuid 123

        # Skip deployment (already deployed)
        kinitro miner-deploy -r user/policy --skip-upload --revision abc123 \\
            --skip-deploy --endpoint https://my-policy.basilica.ai --netuid 123

        # Dry run to see what would happen
        kinitro miner-deploy -r user/policy -p ./my-policy --netuid 123 --dry-run
    """
    import os

    # Validate arguments
    if not skip_upload and not policy_path:
        typer.echo("Error: --path is required unless --skip-upload is set", err=True)
        raise typer.Exit(1)

    if skip_upload and not revision:
        typer.echo("Error: --revision is required when --skip-upload is set", err=True)
        raise typer.Exit(1)

    if skip_deploy and not deployment_id:
        typer.echo("Error: --deployment-id is required when --skip-deploy is set", err=True)
        raise typer.Exit(1)

    # Get credentials from env if not provided
    api_token = basilica_api_token or os.environ.get("BASILICA_API_TOKEN")
    hf = hf_token or os.environ.get("HF_TOKEN")

    # Validate credentials
    if not dry_run:
        if not skip_upload and not hf:
            typer.echo("Error: HF_TOKEN not configured", err=True)
            raise typer.Exit(1)
        if not skip_deploy and not api_token:
            typer.echo("Error: BASILICA_API_TOKEN not configured", err=True)
            typer.echo("Set it via --api-token or BASILICA_API_TOKEN environment variable")
            typer.echo("\nTo get a token, run: basilica tokens create")
            raise typer.Exit(1)

    # Determine steps
    steps = []
    if not skip_upload:
        steps.append("upload")
    if not skip_deploy:
        steps.append("deploy")
    if not skip_commit:
        steps.append("commit")

    typer.echo("=" * 60)
    typer.echo("KINITRO DEPLOYMENT")
    typer.echo("=" * 60)
    typer.echo(f"  Repository: {repo}")
    if policy_path:
        typer.echo(f"  Policy Path: {policy_path}")
    if revision:
        typer.echo(f"  Revision: {revision}")
    if deployment_id:
        typer.echo(f"  Deployment ID: {deployment_id}")
    typer.echo(f"  Steps: {' -> '.join(steps) if steps else 'none'}")
    if dry_run:
        typer.echo("  Mode: DRY RUN")
    typer.echo("=" * 60)

    # Maximum allowed repo size (same as verification limit, configurable via env var)
    max_repo_size_gb = float(
        os.environ.get("KINITRO_MAX_REPO_SIZE_GB", 5.0)
    )
    max_repo_size_bytes = int(max_repo_size_gb * 1024 * 1024 * 1024)

    # Step 1: Upload to HuggingFace
    if not skip_upload:
        typer.echo(f"\n[1/{len(steps)}] Uploading to HuggingFace ({repo})...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would upload {policy_path} to {repo}")
            revision = "dry-run-revision"
        else:
            try:
                import os as _os

                from huggingface_hub import HfApi

                # Check local folder size before uploading
                total_size = 0
                for dirpath, dirnames, filenames in _os.walk(policy_path):
                    # Skip ignored patterns
                    dirnames[:] = [d for d in dirnames if d not in ["__pycache__", ".git"]]
                    for filename in filenames:
                        if not filename.endswith(".pyc"):
                            filepath = _os.path.join(dirpath, filename)
                            total_size += _os.path.getsize(filepath)

                if total_size > max_repo_size_bytes:
                    size_gb = total_size / (1024 * 1024 * 1024)
                    typer.echo(
                        f"Error: Policy folder size ({size_gb:.2f}GB) exceeds maximum "
                        f"allowed ({max_repo_size_gb}GB)",
                        err=True,
                    )
                    raise typer.Exit(1)

                typer.echo(
                    f"  Folder size: {total_size / (1024 * 1024):.2f}MB (max: {max_repo_size_gb}GB)"
                )

                api = HfApi(token=hf)

                # Create repo if it doesn't exist
                typer.echo(f"  Creating/checking repository {repo}...")
                api.create_repo(repo, exist_ok=True, repo_type="model", private=False)
                typer.echo("  Repository ready.")

                # Upload folder
                typer.echo(f"  Uploading {policy_path}...")
                api.upload_folder(
                    folder_path=policy_path,
                    repo_id=repo,
                    commit_message=message,
                    ignore_patterns=["__pycache__/*", "*.pyc", ".git/*"],
                )

                # Get latest commit SHA
                info = api.repo_info(repo, repo_type="model")
                revision = info.sha

                typer.echo(f"  Upload complete. Revision: {revision[:12]}...")

            except typer.Exit:
                # Re-raise Exit exceptions (e.g., from size check) without misleading message
                raise
            except Exception as e:
                typer.echo(f"HuggingFace upload failed: {e}", err=True)
                raise typer.Exit(1)
    else:
        typer.echo(f"\nSkipping upload, using revision: {revision[:12]}...")

    # Step 2: Deploy to Basilica
    if not skip_deploy:
        step_num = 2 if not skip_upload else 1
        typer.echo(f"\n[{step_num}/{len(steps)}] Deploying to Basilica...")

        if dry_run:
            typer.echo(f"  [DRY RUN] Would deploy {repo}@{revision[:12]}...")
            deployment_id = "dry-run-deployment-id"
        else:
            # Import Basilica SDK
            try:
                from basilica import BasilicaClient
            except ImportError:
                typer.echo("Error: basilica-sdk not installed", err=True)
                typer.echo("Run: pip install basilica-sdk")
                raise typer.Exit(1)

            # Derive deployment name
            name = deployment_name or repo.replace("/", "-").lower()
            name = "".join(c if c.isalnum() or c == "-" else "-" for c in name).strip("-")[:63]

            typer.echo(f"  Deployment Name: {name}")

            # Create client
            client = BasilicaClient(api_key=api_token)

            # Generate deployment source code
            hf_token_str = hf or ""
            source_code = f"""
import os
import sys
import subprocess

print("Starting Kinitro Policy Server...")
print(f"HF_REPO: {{os.environ.get('HF_REPO', 'not set')}}")
print(f"HF_REVISION: {{os.environ.get('HF_REVISION', 'not set')}}")

# Download model from HuggingFace
from huggingface_hub import snapshot_download

hf_token = os.environ.get("HF_TOKEN") or None
print("Downloading model from HuggingFace...")
snapshot_download(
    "{repo}",
    revision="{revision}",
    local_dir="/app",
    token=hf_token,
)
print("Model downloaded successfully!")

# Change to /app directory and add to Python path
os.chdir("/app")
sys.path.insert(0, "/app")

# Start the FastAPI server from /app directory
print("Starting uvicorn server on port 8000...")
subprocess.run([
    sys.executable, "-m", "uvicorn",
    "server:app",
    "--host", "0.0.0.0",
    "--port", "8000",
], cwd="/app")
"""

            # Choose image based on GPU requirement
            if gpu_count > 0:
                image = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
            else:
                image = "python:3.11-slim"

            # Build deployment configuration
            deploy_kwargs = {
                "name": name,
                "source": source_code,
                "image": image,
                "port": 8000,
                "memory": memory,
                "cpu": "500m",
                "pip_packages": [
                    "fastapi",
                    "uvicorn",
                    "numpy",
                    "huggingface-hub",
                    "pydantic",
                    "pillow",
                ],
                "env": {
                    "HF_REPO": repo,
                    "HF_REVISION": revision,
                },
                "timeout": 600,
            }

            # Add GPU settings only if GPU is requested
            if gpu_count > 0:
                deploy_kwargs["gpu_count"] = gpu_count
                if min_gpu_memory_gb:
                    deploy_kwargs["min_gpu_memory_gb"] = min_gpu_memory_gb

            # Add HF token if provided
            if hf_token_str:
                deploy_kwargs["env"]["HF_TOKEN"] = hf_token_str

            # Deploy
            typer.echo("  Deploying (this may take several minutes)...")
            try:
                deployment = client.deploy(**deploy_kwargs)
                endpoint_url = deployment.url
                # Extract deployment ID from URL (e.g., https://UUID.deployments.basilica.ai -> UUID)
                if endpoint_url and ".deployments.basilica.ai" in endpoint_url:
                    deployment_id = endpoint_url.split("//")[1].split(".")[0]
                else:
                    # Fallback: use the full URL if format is unexpected
                    deployment_id = endpoint_url
                typer.echo("  Deployment successful!")
                typer.echo(f"  URL: {endpoint_url}")
                typer.echo(f"  Deployment ID: {deployment_id}")
                typer.echo(f"  State: {deployment.state}")
            except Exception as e:
                typer.echo(f"  Deployment failed: {e}", err=True)
                raise typer.Exit(1)
    else:
        typer.echo(f"\nSkipping deployment, using deployment ID: {deployment_id}")

    # Step 3: Commit on-chain
    if not skip_commit and deployment_id:
        step_num = len(steps)
        typer.echo(f"\n[{step_num}/{len(steps)}] Committing on-chain...")

        if dry_run:
            typer.echo(
                f"  [DRY RUN] Would commit {repo}@{revision[:12]}... with deployment_id {deployment_id}"
            )
        else:
            import bittensor as bt

            from kinitro.chain.commitments import commit_model

            subtensor = bt.Subtensor(network=network)
            wallet = bt.Wallet(name=wallet_name, hotkey=hotkey_name)

            typer.echo(f"  Wallet: {wallet.hotkey.ss58_address[:16]}...")

            success = commit_model(
                subtensor=subtensor,
                wallet=wallet,
                netuid=netuid,
                repo=repo,
                revision=revision,
                deployment_id=deployment_id,
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
    typer.echo(f"  Deployment ID: {deployment_id or 'N/A'}")
    typer.echo("=" * 60)


# =============================================================================
# TESTING COMMANDS
# =============================================================================


@app.command()
def test_env(
    env_id: str = typer.Argument(..., help="Environment ID to test"),
    episodes: int = typer.Option(5, help="Number of episodes to run"),
    record_dir: str | None = typer.Option(
        None, "--record-dir", "-r", help="Directory to save recordings (enables recording)"
    ),
    save_images: bool = typer.Option(
        False, "--save-images", help="Save camera images (can be large)"
    ),
    max_steps: int = typer.Option(500, "--max-steps", help="Maximum steps per episode"),
):
    """
    Test an environment with random actions.

    Useful for verifying environment setup.

    Examples:
        # Basic test
        kinitro test-env metaworld/pick-place-v3

        # Record trajectories to disk
        kinitro test-env metaworld/push-v3 --record-dir ./recordings

        # Record with camera images
        kinitro test-env metaworld/push-v3 --record-dir ./recordings --save-images
    """
    import json
    from datetime import datetime
    from pathlib import Path

    import numpy as np

    from kinitro.environments import get_environment

    typer.echo(f"Testing environment: {env_id}")

    env = get_environment(env_id)
    typer.echo(f"  Canonical observation shape: {env.observation_shape}")
    typer.echo(f"  Canonical action shape: {env.action_shape}")

    # Check for camera support
    has_cameras = hasattr(env, "num_cameras") and env.num_cameras > 0
    if has_cameras:
        typer.echo(f"  Number of cameras: {env.num_cameras}")
        if hasattr(env, "image_shape"):
            typer.echo(f"  Image shape: {env.image_shape}")

    # Setup recording directory
    recording = record_dir is not None
    if recording:
        record_path = Path(record_dir)
        record_path.mkdir(parents=True, exist_ok=True)

        # Save run metadata
        metadata = {
            "env_id": env_id,
            "episodes": episodes,
            "max_steps": max_steps,
            "save_images": save_images,
            "timestamp": datetime.now().isoformat(),
            "observation_shape": list(env.observation_shape),
            "action_shape": list(env.action_shape),
        }
        if has_cameras and hasattr(env, "image_shape"):
            metadata["image_shape"] = list(env.image_shape)
            metadata["num_cameras"] = env.num_cameras

        with open(record_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        typer.echo(f"  Recording to: {record_path}")

    successes = 0
    total_reward = 0.0

    for ep in range(episodes):
        task_config = env.generate_task(seed=ep)
        obs = env.reset(task_config)

        typer.echo(f"  Episode {ep + 1} initial obs: {obs}")

        # Storage for trajectory
        observations = [obs.to_payload(include_images=True)]
        actions = []
        rewards = []
        dones = []
        infos = []
        images: dict[str, list[np.ndarray]] = {}

        # Capture initial images if recording
        if recording and save_images and has_cameras and hasattr(env, "get_observation"):
            typer.echo("    Capturing initial images...")
            full_obs = env.get_observation()
            for cam_name, img in full_obs.rgb.items():
                if cam_name not in images:
                    images[cam_name] = []
                images[cam_name].append(np.array(img))
            typer.echo("    Done.")

        ep_reward = 0.0
        steps = 0

        for step_idx in range(max_steps):
            # Random action
            low, high = env.action_bounds
            action = np.random.uniform(low, high)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            steps += 1

            # Record trajectory data
            if recording:
                observations.append(obs.to_payload(include_images=True))
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                # Convert info to serializable format
                infos.append(
                    {k: v for k, v in info.items() if isinstance(v, (int, float, bool, str))}
                )

                # Capture images
                if save_images and has_cameras and hasattr(env, "get_observation"):
                    if step_idx % 100 == 0:
                        typer.echo(f"    Step {step_idx}...")
                    full_obs = env.get_observation()
                    for cam_name, img in full_obs.rgb.items():
                        if cam_name not in images:
                            images[cam_name] = []
                        images[cam_name].append(np.array(img))

            if done:
                break

        success = env.get_success()
        successes += int(success)
        total_reward += ep_reward

        status = "SUCCESS" if success else "FAIL"
        typer.echo(f"  Episode {ep + 1}: {status} | Reward: {ep_reward:.2f} | Steps: {steps}")

        # Save episode data
        if recording:
            ep_dir = record_path / f"episode_{ep:03d}"
            ep_dir.mkdir(exist_ok=True)

            # Save task config
            with open(ep_dir / "task_config.json", "w") as f:
                json.dump(task_config.to_dict(), f, indent=2)

            # Save episode result
            result = {
                "success": success,
                "total_reward": ep_reward,
                "timesteps": steps,
            }
            with open(ep_dir / "result.json", "w") as f:
                json.dump(result, f, indent=2)

            # Save trajectory as numpy archive
            np.savez_compressed(
                ep_dir / "trajectory.npz",
                observations=np.array(observations, dtype=object),
                actions=np.array(actions, dtype=object),
                rewards=np.array(rewards),
                dones=np.array(dones),
            )

            # Save images
            if save_images and images:
                from PIL import Image

                images_dir = ep_dir / "images"
                images_dir.mkdir(exist_ok=True)

                for cam_name, cam_images in images.items():
                    for i, img in enumerate(cam_images):
                        img_path = images_dir / f"{cam_name}_{i:04d}.png"
                        Image.fromarray(img).save(img_path)

                typer.echo(f"    Saved {sum(len(v) for v in images.values())} images")

    env.close()

    typer.echo(f"\nResults: {successes}/{episodes} successful")
    typer.echo(f"Average reward: {total_reward / episodes:.2f}")

    if recording:
        typer.echo(f"\nRecordings saved to: {record_path}")


@app.command()
def test_scoring(
    n_miners: int = typer.Option(5, help="Number of simulated miners"),
    n_envs: int = typer.Option(3, help="Number of environments"),
):
    """
    Test the Pareto scoring mechanism with simulated data.
    """
    import numpy as np

    from kinitro.scoring.pareto import compute_pareto_frontier
    from kinitro.scoring.winners_take_all import compute_full_scoring

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
        kinitro mock-miner

        # Start on specific port
        kinitro mock-miner --port 8002
    """
    import numpy as np
    import uvicorn
    from fastapi import FastAPI
    from pydantic import BaseModel

    from kinitro.rl_interface import CanonicalAction, CanonicalObservation

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
        obs: CanonicalObservation

    class ActResponse(BaseModel):
        action: CanonicalAction

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
            twist = np.random.uniform(-1, 1, size=6)
            gripper = float(np.random.uniform(0, 1))
        else:
            twist = np.zeros(6)
            gripper = 0.0
        action = CanonicalAction(twist_ee_norm=twist.tolist(), gripper_01=gripper)
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
