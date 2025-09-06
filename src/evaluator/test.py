import asyncio
from datetime import datetime

import asyncpg
from pgqueuer import AsyncpgDriver, Queries
from snowflake import SnowflakeGenerator

from validator.db.models import EvaluationJob, EvaluationStatus


async def main():
    gen = SnowflakeGenerator(42)
    job_id = next(gen)
    sub_id = next(gen)
    job = EvaluationJob(
        created_at=datetime.now(),
        updated_at=datetime.now(),
        status=EvaluationStatus.QUEUED,
        submission_id=sub_id,  # type: ignore
        miner_hotkey="5CyY97KCfwRC5UZN58A1cLpZnMgSZAKWtqaaggUfzYiJ6B8d",
        hf_repo_id="rishiad/default_submission",
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

    conn = await asyncpg.connect(
        dsn="postgresql://myuser:mypassword@localhost:5432/kinitrodb"
    )
    driver = AsyncpgDriver(conn)
    q = Queries(driver)
    job_bytes = job.to_bytes()
    print(f"Enqueuing job: {job!r}")
    await q.enqueue(["add_job"], [job_bytes], [0])
    print("Job enqueued successfully.")


if __name__ == "__main__":
    asyncio.run(main())
