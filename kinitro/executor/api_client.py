"""HTTP client for communicating with the API service."""

import aiohttp
import structlog

from kinitro.backend.models import Task, TaskResult

logger = structlog.get_logger()


class APIClient:
    """Client for communicating with the Kinitro API service."""

    def __init__(self, api_url: str, executor_id: str):
        self.api_url = api_url.rstrip("/")
        self.executor_id = executor_id
        self._session: aiohttp.ClientSession | None = None

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close the HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()

    async def fetch_tasks(
        self,
        batch_size: int = 10,
        env_ids: list[str] | None = None,
    ) -> list[Task]:
        """
        Fetch tasks from the API.

        Args:
            batch_size: Maximum number of tasks to fetch
            env_ids: Optional filter by environment IDs

        Returns:
            List of tasks
        """
        session = await self._get_session()

        payload = {
            "executor_id": self.executor_id,
            "batch_size": batch_size,
        }
        if env_ids:
            payload["env_ids"] = env_ids

        try:
            async with session.post(
                f"{self.api_url}/v1/tasks/fetch",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error("fetch_tasks_error", status=resp.status, error=error)
                    return []

                data = await resp.json()
                tasks = [Task(**t) for t in data["tasks"]]
                logger.info(
                    "tasks_fetched",
                    count=len(tasks),
                    total_pending=data["total_pending"],
                )
                return tasks

        except Exception as e:
            logger.error("fetch_tasks_exception", error=str(e))
            return []

    async def submit_results(
        self,
        results: list[TaskResult],
    ) -> tuple[int, int]:
        """
        Submit task results to the API.

        Args:
            results: List of task results

        Returns:
            Tuple of (accepted, rejected) counts
        """
        if not results:
            return 0, 0

        session = await self._get_session()

        payload = {
            "executor_id": self.executor_id,
            "results": [r.model_dump() for r in results],
        }

        try:
            async with session.post(
                f"{self.api_url}/v1/tasks/submit",
                json=payload,
            ) as resp:
                if resp.status != 200:
                    error = await resp.text()
                    logger.error("submit_results_error", status=resp.status, error=error)
                    return 0, len(results)

                data = await resp.json()
                logger.info(
                    "results_submitted",
                    accepted=data["accepted"],
                    rejected=data["rejected"],
                )
                return data["accepted"], data["rejected"]

        except Exception as e:
            logger.error("submit_results_exception", error=str(e))
            return 0, len(results)

    async def health_check(self) -> bool:
        """Check if the API is healthy."""
        session = await self._get_session()

        try:
            async with session.get(f"{self.api_url}/health") as resp:
                return resp.status == 200
        except Exception:
            return False
