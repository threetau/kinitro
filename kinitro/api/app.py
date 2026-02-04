"""FastAPI application for the API service."""

from contextlib import asynccontextmanager

import structlog
import uvicorn
from fastapi import FastAPI

from kinitro.api.config import APIConfig
from kinitro.api.deps import set_config, set_storage
from kinitro.api.routes import (
    health_router,
    miners_router,
    scores_router,
    tasks_router,
    weights_router,
)
from kinitro.backend.storage import Storage

logger = structlog.get_logger()


def create_app(config: APIConfig | None = None) -> FastAPI:
    """
    Create the FastAPI application.

    Args:
        config: API configuration. If None, loads from environment.

    Returns:
        Configured FastAPI application
    """
    if config is None:
        config = APIConfig()

    # Create storage
    storage = Storage(config.database_url)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Application lifespan manager."""
        # Startup
        logger.info("starting_api", port=config.port)

        # Initialize database
        await storage.initialize()

        # Set storage and config for routes
        set_storage(storage)
        set_config(config)

        yield

        # Shutdown
        logger.info("stopping_api")
        await storage.close()

    app = FastAPI(
        title="Kinitro API",
        description="Evaluation backend API for the Bittensor robotics subnet",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Include routes
    app.include_router(health_router)
    app.include_router(weights_router)
    app.include_router(scores_router)
    app.include_router(tasks_router)
    app.include_router(miners_router)

    return app


def run_server(config: APIConfig | None = None) -> None:
    """
    Run the API server.

    Args:
        config: API configuration
    """
    if config is None:
        config = APIConfig()

    app = create_app(config)

    uvicorn.run(
        app,
        host=config.host,
        port=config.port,
        log_level=config.log_level.lower(),
    )
