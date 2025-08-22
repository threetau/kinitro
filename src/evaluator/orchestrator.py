import asyncio
import datetime
import time

import asyncpg
from pgqueuer import PgQueuer
from pgqueuer.db import AsyncpgDriver
from pgqueuer.models import Job
from pgqueuer.queries import Queries
from snowflake import SnowflakeGenerator

from core.db.db_manager import create_database_manager
from core.db.models import EvaluationJob, EvaluationStatus
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

        @pgq.entrypoint("fetch")
        async def process(job: Job) -> None:
            print(f"Processed: {job!r}")

        try:
            print(f"Orchestrator running... timestamp: {time.time()}")
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            logger.info("Stopping orchestrator...")

        return pgq

    def stop(self):
        logger.info("Stopping orchestrator...")
        # Add cleanup logic here
        pass


async def main():
    logger.info(f"Orchestrator running... timestamp: {time.time()}")
    gen = SnowflakeGenerator(42)
    job_id = next(gen)
    sub_id = next(gen)

    job = EvaluationJob(
        created_at=datetime.datetime.now(),
        updated_at=datetime.datetime.now(),
        status=EvaluationStatus.QUEUED,
        submission_id=sub_id,  # type: ignore
        miner_hotkey="5CyY97KCfwRC5UZN58A1cLpZnMgSZAKWtqaaggUfzYiJ6B8d",
        hf_repo_id="rishiad/default_submission",
        hf_repo_commit="93a2aa6de1069bcc37c60e80954d3a2c6e202678",
        env_provider="metaworld",
        env_name="MT10",
        id=job_id,  # type: ignore
        container_id=None,
        ray_worker_id=None,
        retry_count=0,
        max_retries=3,
        logs_path="./data/logs",
        random_seed=None,
        eval_start=None,
        eval_end=None,
    )

    conn = await asyncpg.connect()
    driver = AsyncpgDriver(conn)
    q = Queries(driver)
    job_bytes = job.to_bytes()
    await q.enqueue(["fetch"], [job_bytes], [0])

    # try:
    #     while True:
    #         await asyncio.sleep(1)
    # except KeyboardInterrupt:
    #     logger.info("Stopping orchestrator...")


if __name__ == "__main__":
    orc = Orchestrator(EvaluatorConfig())
    asyncio.run(main())
