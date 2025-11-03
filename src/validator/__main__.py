"""
Main entry point for WebSocket-based validator.
"""

import asyncio
import signal
import sys

from core.log import get_logger

from .config import ValidatorConfig, ValidatorMode
from .lite_validator import LiteValidator
from .websocket_validator import WebSocketValidator

logger = get_logger(__name__)


class ValidatorService:
    """Service wrapper for the WebSocket validator."""

    def __init__(self):
        self.config = ValidatorConfig()
        self.validator = None
        self._shutdown_event = asyncio.Event()
        mode_value = self.config.settings.get(
            "validator_mode", ValidatorMode.FULL.value
        )
        self.mode = ValidatorMode(mode_value)

    def setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating shutdown...")
            self._shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

    async def run(self):
        """Run the validator service."""
        try:
            logger.info("Starting Kinitro Validator Service (mode=%s)", self.mode.value)

            match self.mode:
                case ValidatorMode.LITE:
                    self.validator = LiteValidator(self.config)
                case ValidatorMode.FULL:
                    self.validator = WebSocketValidator(self.config)
                case _:
                    raise ValueError(f"Unknown validator mode: {self.mode}")

            # Start validator in background
            validator_task = asyncio.create_task(self.validator.start())

            # Wait for shutdown signal
            logger.info("Validator service is running. Press Ctrl+C to stop.")
            await self._shutdown_event.wait()

            # Graceful shutdown
            logger.info("Shutting down validator service...")
            if self.validator:
                await self.validator.stop()

            # Cancel validator task
            validator_task.cancel()
            try:
                await validator_task
            except asyncio.CancelledError:
                pass

            logger.info("Validator service stopped")

        except Exception as e:
            logger.error(f"Fatal error in validator service: {e}")
            sys.exit(1)


async def main():
    """Main entry point."""
    service = ValidatorService()
    service.setup_signal_handlers()
    await service.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unhandled exception: {e}")
        sys.exit(1)
