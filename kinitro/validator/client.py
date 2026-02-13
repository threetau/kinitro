"""HTTP client for the backend API."""

from dataclasses import dataclass
from typing import Any

import httpx
import structlog

logger = structlog.get_logger()


@dataclass
class WeightsData:
    """Weights data from backend."""

    cycle_id: int
    block_number: int
    weights: dict[int, float]
    uids: list[int]
    values_u16: list[int]
    metadata: dict[str, Any]  # Any: open-ended backend metadata


class BackendClient:
    """
    HTTP client for communicating with the evaluation backend.

    Usage:
        client = BackendClient("http://localhost:8000")
        weights = await client.get_latest_weights()
        if weights:
            print(f"Block: {weights.block_number}")
            print(f"Weights: {weights.weights}")
    """

    def __init__(
        self,
        backend_url: str,
        timeout: float = 30.0,
    ):
        """
        Initialize client.

        Args:
            backend_url: Base URL of the backend service
            timeout: Request timeout in seconds
        """
        self.backend_url = backend_url.rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                base_url=self.backend_url,
                timeout=self.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """
        Check if backend is healthy.

        Returns:
            True if backend is healthy
        """
        try:
            client = await self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning("backend_health_check_failed", error=str(e))
            return False

    async def get_latest_weights(self) -> WeightsData | None:
        """
        Fetch the latest computed weights from backend.

        Returns:
            WeightsData if available, None otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get("/v1/weights/latest")

            if response.status_code == 404:
                logger.info("no_weights_available")
                return None

            response.raise_for_status()
            data = response.json()

            return WeightsData(
                cycle_id=data["cycle_id"],
                block_number=data["block_number"],
                weights={int(k): float(v) for k, v in data["weights"].items()},
                uids=data["weights_u16"]["uids"],
                values_u16=data["weights_u16"]["values"],
                metadata=data.get("metadata", {}),
            )

        except httpx.HTTPStatusError as e:
            logger.error(
                "backend_request_failed",
                status_code=e.response.status_code,
                detail=e.response.text,
            )
            return None
        except Exception as e:
            logger.error("backend_request_error", error=str(e))
            return None

    async def get_weights_for_block(self, block_number: int) -> WeightsData | None:
        """
        Fetch weights for a specific block.

        Args:
            block_number: Block number to fetch weights for

        Returns:
            WeightsData if available, None otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get(f"/v1/weights/{block_number}")

            if response.status_code == 404:
                return None

            response.raise_for_status()
            data = response.json()

            return WeightsData(
                cycle_id=data["cycle_id"],
                block_number=data["block_number"],
                weights={int(k): float(v) for k, v in data["weights"].items()},
                uids=data["weights_u16"]["uids"],
                values_u16=data["weights_u16"]["values"],
                metadata=data.get("metadata", {}),
            )

        except Exception as e:
            logger.error("backend_request_error", error=str(e))
            return None

    async def get_status(self) -> dict[str, Any] | None:  # Any: backend status schema is open-ended
        """
        Get backend status.

        Returns:
            Status dict if available, None otherwise
        """
        try:
            client = await self._get_client()
            response = await client.get("/v1/status")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("backend_status_error", error=str(e))
            return None
