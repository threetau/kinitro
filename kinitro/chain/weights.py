"""Weight setting for validators."""

import numpy as np
import structlog
from bittensor import Subtensor
from bittensor_wallet import Wallet

from kinitro.types import EligibilityResult, MinerUID

logger = structlog.get_logger()


def weights_to_u16(weights: dict[MinerUID, float]) -> tuple[list[MinerUID], list[int]]:
    """
    Convert float weights to u16 format for chain submission.

    Bittensor expects weights as u16 integers that sum to 65535.

    Args:
        weights: Dict mapping uid -> weight (should sum to 1.0)

    Returns:
        Tuple of (uids, weights_u16)
    """
    if not weights:
        return [], []

    uids = list(weights.keys())
    values = np.array([weights[uid] for uid in uids], dtype=np.float64)

    # Normalize to sum to 1
    total = values.sum()
    if total > 0:
        values = values / total
    else:
        # Uniform if all zeros
        values = np.ones_like(values) / len(values)

    # Convert to u16 (0-65535)
    weights_u16 = (values * 65535).astype(np.uint16).tolist()

    return uids, weights_u16


def set_weights(
    subtensor: Subtensor,
    wallet: Wallet,
    netuid: int,
    weights: dict[MinerUID, float],
    wait_for_inclusion: bool = True,
    wait_for_finalization: bool = False,
) -> bool:
    """
    Submit weights to chain.

    Args:
        subtensor: Bittensor subtensor connection
        wallet: Validator's wallet
        netuid: Subnet UID
        weights: Dict mapping uid -> weight
        wait_for_inclusion: Wait for transaction inclusion
        wait_for_finalization: Wait for finalization

    Returns:
        True if weights were set successfully
    """
    uids, weights_u16 = weights_to_u16(weights)

    if not uids:
        logger.warning("no_weights_to_set")
        return False

    try:
        # Log weight distribution
        top_weights = sorted(zip(uids, weights_u16), key=lambda x: x[1], reverse=True)[:5]
        logger.info(
            "setting_weights",
            n_miners=len(uids),
            top_weights=[(uid, w) for uid, w in top_weights],
        )

        result = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=[int(uid) for uid in uids],
            weights=weights_u16,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if result.success:
            logger.info("weights_set_successfully")
        else:
            logger.error("weights_set_failed", message=result.message)

        return result.success

    except Exception as e:
        logger.error("weights_set_exception", error=str(e))
        return False


def verify_weight_setting_eligibility(
    subtensor: Subtensor,
    wallet: Wallet,
    netuid: int,
) -> EligibilityResult:
    """
    Check if validator can set weights.

    Args:
        subtensor: Bittensor subtensor connection
        wallet: Validator's wallet
        netuid: Subnet UID

    Returns:
        Tuple of (eligible, reason)
    """
    try:
        # Use neurons() instead of metagraph() to avoid runtime API compatibility issues
        # with certain substrate node versions
        neurons = subtensor.neurons(netuid=netuid)

        if not neurons:
            return EligibilityResult(False, "No neurons found on subnet")

        # Check if hotkey is registered
        hotkey = wallet.hotkey.ss58_address
        hotkeys = [n.hotkey for n in neurons]

        if hotkey not in hotkeys:
            return EligibilityResult(False, "Hotkey not registered on subnet")

        uid = hotkeys.index(hotkey)
        _neuron = neurons[uid]  # noqa: F841 - kept for future validation

        # NOTE: this has been disabled for now do not check permit and stake
        # # Check if has validator permit
        # if not neuron.validator_permit:
        #     return EligibilityResult(False, "No validator permit")

        # # Check stake (neuron.stake is a Balance object, compare raw value)
        # stake_tao = float(neuron.stake.tao)
        # if stake_tao < 1.0:  # Minimum stake threshold
        #     return EligibilityResult(False, f"Insufficient stake: {stake_tao}")

        return EligibilityResult(True, "Eligible")

    except Exception as e:
        return EligibilityResult(False, f"Error checking eligibility: {e}")
