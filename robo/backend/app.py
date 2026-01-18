"""FastAPI application for the backend service."""

import asyncio
import signal
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from robo.backend.config import BackendConfig
from robo.backend.routes import router, set_scheduler, set_storage
from robo.backend.scheduler import EvaluationScheduler
from robo.backend.storage import Storage

logger = structlog.get_logger()

# Global reference for signal handler
_scheduler: EvaluationScheduler | None = None


_signal_received = False


def _force_cleanup_handler(signum, frame):
    """Signal handler to force cleanup docker containers on interrupt."""
    import subprocess
    
    global _signal_received
    
    container_name = "robo-eval-env"
    
    if _signal_received:
        # Second signal - force cleanup and exit immediately
        logger.info("force_exit", signal=signum)
        # Kill container synchronously before exiting
        try:
            subprocess.run(["docker", "rm", "-f", container_name], 
                          capture_output=True, timeout=3)
        except Exception:
            pass
        import os
        os._exit(1)
    
    _signal_received = True
    logger.info("received_signal_cleanup", signal=signum)
    
    # Kill container immediately on first signal too
    try:
        subprocess.run(["docker", "rm", "-f", container_name], 
                      capture_output=True, timeout=3)
        logger.info("container_killed_on_signal", container=container_name)
    except Exception as e:
        logger.warning("container_kill_failed", error=str(e))


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
    
    # Set global for signal handler
    global _scheduler
    _scheduler = scheduler
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, _force_cleanup_handler)
    signal.signal(signal.SIGTERM, _force_cleanup_handler)

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
        
        # Cancel the scheduler task first
        scheduler_task.cancel()
        try:
            await asyncio.wait_for(scheduler_task, timeout=5.0)
        except (asyncio.CancelledError, asyncio.TimeoutError):
            pass
        
        # Now cleanup scheduler resources (with timeout)
        try:
            await asyncio.wait_for(scheduler.stop(), timeout=10.0)
        except asyncio.TimeoutError:
            logger.warning("scheduler_stop_timeout", msg="Force stopping scheduler")
        
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
