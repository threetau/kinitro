import asyncio

import asyncpg
from pgqueuer import PgQueuer
from pgqueuer.db import AsyncpgDriver
from pgqueuer.models import Job

from core.db.db_manager import create_database_manager
from core.log import get_logger
from evaluator.config import EvaluatorConfig

logger = get_logger(__name__)


class Orchestrator:
    def __init__(self, config: EvaluatorConfig):
        self.config = config
        print(f"Orchestrator initialized with db: {self.config.pg_database}")  # pyright: ignore[reportAttributeAccessIssue]
        self.db = create_database_manager(self.config.pg_database, self.config.duck_db)  # pyright: ignore[reportAttributeAccessIssue]
        # Add additional initialization here
        logger.info(f"Orchestrator initialized with config: {self.config}")

    async def start(self):
        logger.info("Starting orchestrator...")
        conn = await asyncpg.connect()

        driver = AsyncpgDriver(conn)
        pgq = PgQueuer(driver)

        @pgq.entrypoint("add_job")
        async def process(job: Job) -> None:
            print(f"Processed: {job!r}")

        await pgq.run()
        return pgq

    def stop(self):
        logger.info("Stopping orchestrator...")
        # Add cleanup logic here
        pass


if __name__ == "__main__":
    orc = Orchestrator(EvaluatorConfig())
    asyncio.run(orc.start())
