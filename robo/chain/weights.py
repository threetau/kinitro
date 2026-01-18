"""Weight setting for validators."""

import numpy as np
import structlog

logger = structlog.get_logger()


def weights_to_u16(weights: dict[int, float]) -> tuple[list[int], list[int]]:
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
    subtensor,  # bt.Subtensor
    wallet,  # bt.Wallet
    netuid: int,
    weights: dict[int, float],
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

        success = subtensor.set_weights(
            wallet=wallet,
            netuid=netuid,
            uids=uids,
            weights=weights_u16,
            wait_for_inclusion=wait_for_inclusion,
            wait_for_finalization=wait_for_finalization,
        )

        if success:
            logger.info("weights_set_successfully")
        else:
            logger.error("weights_set_failed")

        return success

    except Exception as e:
        logger.error("weights_set_exception", error=str(e))
        return False


def verify_weight_setting_eligibility(
    subtensor,  # bt.Subtensor
    wallet,  # bt.Wallet
    netuid: int,
) -> tuple[bool, str]:
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
            return False, "No neurons found on subnet"

        # Check if hotkey is registered
        hotkey = wallet.hotkey.ss58_address
        hotkeys = [n.hotkey for n in neurons]

        if hotkey not in hotkeys:
            return False, "Hotkey not registered on subnet"

        uid = hotkeys.index(hotkey)
        neuron = neurons[uid]

        # NOTE: this has been disbaled for now do not check permit and stake
        # # Check if has validator permit
        # if not neuron.validator_permit:
        #     return False, "No validator permit"

        # # Check stake (neuron.stake is a Balance object, compare raw value)
        # stake_tao = float(neuron.stake.tao)
        # if stake_tao < 1.0:  # Minimum stake threshold
        #     return False, f"Insufficient stake: {stake_tao}"

        return True, "Eligible"

    except Exception as e:
        return False, f"Error checking eligibility: {e}"
