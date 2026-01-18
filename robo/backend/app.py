"""FastAPI application for the backend service."""

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from robo.backend.config import BackendConfig
from robo.backend.routes import router, set_scheduler, set_storage
from robo.backend.scheduler import EvaluationScheduler
from robo.backend.storage import Storage

logger = structlog.get_logger()


def create_app(config: BackendConfig | None = None) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        config: Backend configuration. If None, loads from environment.

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = BackendConfig()

    # Create storage and scheduler
    storage = Storage(config.database_url)
    scheduler = EvaluationScheduler(config, storage)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        logger.info("starting_backend", port=config.port)

        # Initialize database
        await storage.initialize()

        # Set globals for routes
        set_storage(storage)
        set_scheduler(scheduler)

        # Start background evaluation scheduler
        scheduler_task = asyncio.create_task(scheduler.start())

        yield

        # Shutdown
        logger.info("stopping_backend")
        await scheduler.stop()
        scheduler_task.cancel()
        await storage.close()

    app = FastAPI(
        title="Robo Subnet Backend",
        description="Evaluation backend for the Bittensor robotics subnet",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include routes
    app.include_router(router)

    return app


def run_server(config: BackendConfig | None = None) -> None:
    """
    Run the backend server.

    Args:
        config: Backend configuration
    """
    import uvicorn

    if config is None:
        config = BackendConfig()

    app = create_app(config)

    # Use asyncio loop instead of uvloop to allow nest_asyncio (required by affinetes)
    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
        loop="asyncio",
    )
