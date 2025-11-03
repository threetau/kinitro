"""
Lite validator that polls the backend HTTP weights endpoint and writes weights on-chain.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Optional

import httpx
from pydantic import ValidationError

from backend.models import WeightsSnapshot
from core.chain import set_node_weights
from core.log import get_logger
from core.neuron import Neuron

from .config import ValidatorConfig

logger = get_logger(__name__)


class LiteValidator(Neuron):
    """
    Minimal validator implementation that periodically polls the backend weights endpoint
    and calls `set_node_weights` on the Bittensor chain.
    """

    def __init__(self, config: ValidatorConfig):
        super().__init__(config)
        self.hotkey = self.keypair.ss58_address
        self.weights_url = config.settings.get(
            "weights_url", "https://api.kinitro.ai/weights"
        )
        self.poll_interval = float(config.settings.get("weights_poll_interval", 30.0))
        self.request_timeout = float(
            config.settings.get("weights_request_timeout", 10.0)
        )
        self.stale_threshold = float(
            config.settings.get("weights_stale_threshold", 180.0)
        )
        self._running = False
        self._stop_event = asyncio.Event()
        self._last_snapshot_timestamp: Optional[datetime] = None
        self._last_weights_signature: Optional[tuple[tuple[int, float], ...]] = None
        self._last_backend_timestamp: Optional[datetime] = None
        self._last_success_at: Optional[datetime] = None
        self._max_backoff = max(self.poll_interval * 10.0, 300.0)
        self._node_resync_interval = max(self.stale_threshold, 300.0)
        self._last_resync_at: Optional[datetime] = None
        self._http_client: Optional[httpx.AsyncClient] = None

        logger.info(
            "Lite validator initialized (hotkey=%s, weights_url=%s, poll_interval=%.1fs, stale_threshold=%.1fs)",
            self.hotkey,
            self.weights_url,
            self.poll_interval,
            self.stale_threshold,
        )

    async def start(self) -> None:
        """Begin the polling loop."""
        if self._running:
            logger.warning("Lite validator already running")
            return

        logger.info("Starting lite validator polling loop")
        self._running = True
        self._stop_event.clear()
        backoff = self.poll_interval
        self._http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(self.request_timeout)
        )

        try:
            while self._running:
                success = await self._poll_once()
                backoff = (
                    self.poll_interval
                    if success
                    else min(backoff * 2.0, self._max_backoff)
                )

                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=backoff)
                except asyncio.TimeoutError:
                    continue
        finally:
            self._running = False
            self._stop_event.set()
            await self._close_http_client()
            logger.info("Lite validator loop exited")

    async def stop(self) -> None:
        """Stop the polling loop."""
        if not self._running:
            return

        logger.info("Stopping lite validator")
        self._running = False
        self._stop_event.set()
        await self._close_http_client()

    async def _close_http_client(self) -> None:
        client = self._http_client
        if client is None:
            return
        self._http_client = None
        try:
            await client.aclose()
        except Exception as exc:  # pragma: no cover
            logger.debug("Error closing HTTP client: %s", exc)

    async def _poll_once(self) -> bool:
        """Fetch and process a single weight snapshot."""
        try:
            snapshot = await self._fetch_weights_snapshot()
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Failed to fetch weights snapshot: %s", exc)
            return False

        if snapshot is None:
            logger.debug("No weights snapshot available yet")
            if self._last_backend_timestamp:
                age = (
                    datetime.now(timezone.utc) - self._last_backend_timestamp
                ).total_seconds()
                if age > self.stale_threshold:
                    logger.warning(
                        "No fresh weight snapshots received for %.1fs (threshold=%.1fs)",
                        age,
                        self.stale_threshold,
                    )
            return False

        if self._stop_event.is_set():
            logger.debug("Shutdown requested; skipping snapshot processing")
            return False

        now = datetime.now(timezone.utc)

        try:
            await self._handle_snapshot(snapshot, now)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error handling weights snapshot: %s", exc)
            return False

        return True

    async def _fetch_weights_snapshot(self) -> Optional[WeightsSnapshot]:
        """Retrieve the latest weights via HTTP."""

        if self._stop_event.is_set():
            return None

        client = self._http_client
        if client is None:
            raise RuntimeError("HTTP client not initialized")

        try:
            response = await client.get(self.weights_url)
        except httpx.RequestError as exc:
            if self._running:
                logger.warning("Weight endpoint request failed: %s", exc)
            return None

        if response.status_code == 404:
            return None

        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error("Unexpected response from weight endpoint: %s", exc)
            return None

        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("Failed to decode weights JSON payload: %s", exc)
            return None

        try:
            return WeightsSnapshot.model_validate(payload)
        except ValidationError as exc:
            logger.error("Failed to validate weights snapshot payload: %s", exc)
            return None

    async def _handle_snapshot(self, snapshot: WeightsSnapshot, now: datetime) -> bool:
        """Validate and, if new, apply the weight snapshot."""

        timestamp = snapshot.updated_at
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)

        snapshot_label = timestamp.isoformat()
        self._last_backend_timestamp = timestamp
        age = (now - timestamp).total_seconds()
        if age > self.stale_threshold:
            logger.warning(
                "Weights snapshot (updated_at=%s) is stale by %.1fs (threshold=%.1fs)",
                snapshot_label,
                age,
                self.stale_threshold,
            )

        weights_payload = snapshot.weights
        if not weights_payload:
            logger.error("No valid weights found in snapshot (%s)", snapshot_label)
            return False

        node_ids = []
        node_weights = []
        for node_id, weight in weights_payload.items():
            node_ids.append(int(node_id))
            node_weights.append(float(weight))

        weights_signature = tuple(
            sorted(
                (int(node_id), float(weight))
                for node_id, weight in weights_payload.items()
            )
        )
        if self._last_snapshot_timestamp:
            if timestamp <= self._last_snapshot_timestamp and (
                self._last_weights_signature == weights_signature
            ):
                logger.debug(
                    "Skipping snapshot updated_at=%s (duplicate or older than last applied)",
                    snapshot_label,
                )
                return False
        elif self._last_weights_signature == weights_signature:
            logger.debug(
                "Skipping snapshot (%s) with identical weights payload",
                snapshot_label,
            )
            return False

        # Periodically refresh node metadata to stay aligned with on-chain state.
        if self._should_resync_nodes(now):
            logger.info("Refreshing node metadata from chain before setting weights")
            self.sync_nodes()
            self._last_resync_at = now

        total_weight = sum(node_weights)
        logger.info(
            "Applying weights snapshot updated_at=%s to chain (%s miners, total_weight=%.6f)",
            snapshot_label,
            len(node_ids),
            total_weight,
        )
        logger.debug(
            "Weight payload for updated_at=%s: %s",
            snapshot_label,
            ", ".join(
                f"{uid}:{weight:.6f}" for uid, weight in zip(node_ids, node_weights)
            ),
        )

        success = set_node_weights(
            substrate=self.substrate,
            keypair=self.keypair,
            node_ids=node_ids,
            node_weights=node_weights,
            netuid=self.netuid,
            validator_node_id=self.uid,
            version_key=0,
            wait_for_inclusion=True,
            wait_for_finalization=False,
        )

        if not success:
            logger.error(
                "Failed to set weights on-chain for snapshot updated_at=%s",
                snapshot_label,
            )
            return False

        self._last_snapshot_timestamp = timestamp
        self._last_weights_signature = weights_signature
        logger.info(
            "Successfully set weights on-chain for snapshot updated_at=%s (processed %s miners)",
            snapshot_label,
            len(node_ids),
        )
        return True

    def _should_resync_nodes(self, now: datetime) -> bool:
        if self.nodes is None:
            return True
        if self._last_resync_at is None:
            return True
        return (
            now - self._last_resync_at
        ).total_seconds() >= self._node_resync_interval
