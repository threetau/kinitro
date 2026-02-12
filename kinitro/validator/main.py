"""Simplified validator that polls backend for weights and submits to chain."""

from __future__ import annotations

import asyncio

import structlog
from bittensor import Subtensor
from bittensor_wallet import Wallet

from kinitro.chain.weights import set_weights, verify_weight_setting_eligibility
from kinitro.config import ValidatorConfig
from kinitro.types import MinerUID
from kinitro.validator.client import BackendClient

logger = structlog.get_logger()


class Validator:
    """
    Simplified validator that:
    1. Polls the evaluation backend for computed weights
    2. Submits weights to the Bittensor chain

    The heavy lifting (evaluation, scoring) is done by the backend service.
    """

    def __init__(
        self,
        config: ValidatorConfig,
        backend_url: str,
    ):
        """
        Initialize validator.

        Args:
            config: Validator configuration
            backend_url: URL of the evaluation backend
        """
        self.config = config
        self.backend_url = backend_url
        self.client = BackendClient(backend_url)
        self.subtensor: Subtensor | None = None
        self.wallet = Wallet(
            name=config.wallet_name,
            hotkey=config.hotkey_name,
        )

        # State
        self._last_submitted_block = 0

        logger.info(
            "validator_initialized",
            network=config.network,
            netuid=config.netuid,
            backend_url=backend_url,
        )

    async def run(self) -> None:
        """
        Main validator loop.

        Polls backend for weights and submits to chain.
        """
        logger.info("starting_validator_loop")

        if self.subtensor is None:
            raise RuntimeError("subtensor is not initialized")

        # Check eligibility
        eligible, reason = await asyncio.to_thread(
            verify_weight_setting_eligibility,
            self.subtensor,
            self.wallet,
            self.config.netuid,
        )
        if not eligible:
            logger.error("validator_not_eligible", reason=reason)
            return

        logger.info("validator_eligible", hotkey=self.wallet.hotkey.ss58_address)

        # Check backend health
        if not await self.client.health_check():
            logger.error("backend_not_reachable", url=self.backend_url)
            return

        logger.info("backend_connected", url=self.backend_url)

        # Main loop
        while True:
            try:
                await self._weight_setting_cycle()
            except KeyboardInterrupt:
                logger.info("validator_interrupted")
                break
            except Exception as e:
                logger.error("weight_setting_cycle_failed", error=str(e))

            # Poll interval (shorter than evaluation interval since we're just checking)
            await asyncio.sleep(60)  # Check every minute

    async def _weight_setting_cycle(self) -> None:
        """Single cycle: fetch weights from backend and submit if new."""
        if self.subtensor is None:
            raise RuntimeError("subtensor is not initialized")

        # Get latest weights from backend
        weights_data = await self.client.get_latest_weights()

        if weights_data is None:
            logger.debug("no_weights_available")
            return

        # Check if we already submitted these weights
        if weights_data.block_number <= self._last_submitted_block:
            logger.debug(
                "weights_already_submitted",
                weights_block=weights_data.block_number,
                last_submitted=self._last_submitted_block,
            )
            return

        logger.info(
            "new_weights_available",
            block=weights_data.block_number,
            cycle_id=weights_data.cycle_id,
            n_miners=len(weights_data.weights),
        )

        # Submit weights to chain
        success = await asyncio.to_thread(
            set_weights,
            self.subtensor,
            self.wallet,
            self.config.netuid,
            {MinerUID(k): v for k, v in weights_data.weights.items()},
        )

        if success:
            self._last_submitted_block = weights_data.block_number
            logger.info(
                "weights_submitted",
                block=weights_data.block_number,
                n_miners=len(weights_data.weights),
            )
        else:
            logger.error("weights_submission_failed", block=weights_data.block_number)

    async def close(self) -> None:
        """Clean up resources."""
        await self.client.close()


async def run_validator(
    config: ValidatorConfig,
    backend_url: str,
) -> None:
    """
    Entry point for running the validator.

    Args:
        config: Validator configuration
        backend_url: URL of the evaluation backend
    """
    validator = Validator(config, backend_url)
    try:
        validator.subtensor = await asyncio.to_thread(Subtensor, network=config.network)
        await validator.run()
    finally:
        if validator.subtensor is not None:
            try:
                await asyncio.to_thread(validator.subtensor.close)
            except Exception:
                logger.warning("subtensor_close_failed", exc_info=True)
        await validator.close()
