"""
Lite validator that polls the backend HTTP weights endpoint and writes weights on-chain.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

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
        self._last_sequence: Optional[int] = None
        self._last_backend_timestamp: Optional[datetime] = None
        self._last_success_at: Optional[datetime] = None
        self._max_backoff = max(self.poll_interval * 10.0, 300.0)
        self._node_resync_interval = max(self.stale_threshold, 300.0)
        self._last_resync_at: Optional[datetime] = None

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
            logger.info("Lite validator loop exited")

    async def stop(self) -> None:
        """Stop the polling loop."""
        if not self._running:
            return

        logger.info("Stopping lite validator")
        self._running = False
        self._stop_event.set()

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

        now = datetime.now(timezone.utc)

        try:
            processed = await self._handle_snapshot(snapshot, now)
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.error("Error handling weights snapshot: %s", exc)
            return False

        if processed:
            self._last_success_at = now
        else:
            logger.debug("Snapshot processed without chain update (sequence unchanged)")

        return True

    async def _fetch_weights_snapshot(self) -> Optional[Dict[str, Any]]:
        """Retrieve the latest weights via HTTP."""

        def _fetch() -> Optional[Dict[str, Any]]:
            try:
                response = requests.get(self.weights_url, timeout=self.request_timeout)
            except requests.RequestException as exc:
                logger.warning("Weight endpoint request failed: %s", exc)
                return None

            if response.status_code == 404:
                return None

            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                logger.error("Unexpected response from weight endpoint: %s", exc)
                return None

            try:
                return response.json()
            except ValueError as exc:
                logger.error("Failed to decode weights JSON payload: %s", exc)
                return None

        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _fetch)

    async def _handle_snapshot(self, snapshot: Dict[str, Any], now: datetime) -> bool:
        """Validate and, if new, apply the weight snapshot."""
        sequence = snapshot.get("sequence")
        if sequence is None:
            logger.error("Weights snapshot missing 'sequence' field: %s", snapshot)
            return False

        weights_payload = snapshot.get("weights")
        if not isinstance(weights_payload, dict):
            logger.error("Weights snapshot has invalid 'weights' payload: %s", snapshot)
            return False

        backend_timestamp = self._parse_timestamp(snapshot.get("updated_at"))
        if backend_timestamp:
            self._last_backend_timestamp = backend_timestamp
            age = (now - backend_timestamp).total_seconds()
            if age > self.stale_threshold:
                logger.warning(
                    "Weights snapshot (seq=%s) is stale by %.1fs (threshold=%.1fs)",
                    sequence,
                    age,
                    self.stale_threshold,
                )
        else:
            logger.warning(
                "Weights snapshot missing or invalid timestamp: %s", snapshot
            )

        if self._last_sequence is not None and sequence <= self._last_sequence:
            # Nothing new to apply; treat as success so we continue polling normally.
            return False

        node_ids: list[int] = []
        node_weights: list[float] = []
        malformed_keys = 0

        for raw_node_id, weight in weights_payload.items():
            try:
                node_id = int(raw_node_id)
            except (TypeError, ValueError):
                malformed_keys += 1
                continue

            try:
                weight_value = float(weight)
            except (TypeError, ValueError):
                malformed_keys += 1
                continue

            node_ids.append(node_id)
            node_weights.append(weight_value)

        if malformed_keys:
            logger.warning(
                "Snapshot sequence %s contained %s malformed weight entries that were skipped",
                sequence,
                malformed_keys,
            )

        if not node_ids:
            logger.error("No valid weights found in snapshot sequence %s", sequence)
            return False

        # Periodically refresh node metadata to stay aligned with on-chain state.
        if self._should_resync_nodes(now):
            logger.info("Refreshing node metadata from chain before setting weights")
            self.sync_nodes()
            self._last_resync_at = now

        logger.info(
            "Applying weights snapshot seq=%s to chain (%s miners, total_weight=%.6f)",
            sequence,
            len(node_ids),
            sum(node_weights),
        )
        logger.debug(
            "Weight payload for seq=%s: %s",
            sequence,
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

        if success:
            self._last_sequence = sequence
            logger.info(
                "Successfully set weights on-chain for snapshot seq=%s (processed %s miners)",
                sequence,
                len(node_ids),
            )
            return True

        logger.error("Failed to set weights on-chain for snapshot seq=%s", sequence)
        return False

    def _should_resync_nodes(self, now: datetime) -> bool:
        if self.nodes is None:
            return True
        if self._last_resync_at is None:
            return True
        return (
            now - self._last_resync_at
        ).total_seconds() >= self._node_resync_interval

    @staticmethod
    def _parse_timestamp(raw_value: Any) -> Optional[datetime]:
        if isinstance(raw_value, datetime):
            return (
                raw_value
                if raw_value.tzinfo
                else raw_value.replace(tzinfo=timezone.utc)
            )
        if isinstance(raw_value, str):
            try:
                # fromisoformat supports offsets like "+00:00"
                parsed = datetime.fromisoformat(raw_value.replace("Z", "+00:00"))
                return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
            except ValueError:
                return None
        return None
