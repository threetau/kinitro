import asyncio
import time

from core.log import get_logger

# from core.validator import Validator

logger = get_logger(__name__)


class Validator: ...


async def main():
    validator = Validator()
    # await validator.start()

    logger.info(f"Validator running... timestamp: {time.time()}")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        ...
        # await validator.stop()


if __name__ == "__main__":
    asyncio.run(main())
