"""
Configuration options for Storb
"""

from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dynaconf import Dynaconf, Validator

from .constants import NeuronType
from .log import get_logger

logger = get_logger(__name__)


@dataclass
class ConfigOpts:
    neuron_name: str
    neuron_type: NeuronType
    settings_files: Optional[list[str]] = None


class Config:
    """Configurations

    The CLI arguments correspond to the options in the TOML file.
    """

    def __init__(self, opts: ConfigOpts):
        settings_files = ["settings.toml", ".secrets.toml"]
        if opts.settings_files:
            settings_files.extend(opts.settings_files)

        self.settings: Dynaconf = Dynaconf(
            settings_files=settings_files,
            validators=[
                Validator("wallet_name", must_exist=True, default="default_wallet"),
            ],
        )
        self._parser = ArgumentParser()

        # Add arguments
        self.add_args()
        match opts.neuron_type:
            case NeuronType.Miner:
                neuron_name = "miner"
            case NeuronType.Validator:
                neuron_name = "validator"
            case _:
                raise ValueError(
                    f"The provided neuron_type ({self.settings.neuron_type}) is not valid."
                )

        options = self._parser.parse_args()
        logger.info(f"Current config: {vars(options)}")
        self.add_full_path(options, neuron_name)

        # Update settings with parsed options
        option_vars = vars(options)
        for key, value in option_vars.items():
            self.settings[key] = value

    @classmethod
    def add_full_path(cls, options, neuron_name: str):
        """Validates that the neuron config path exists"""

        base_path = Path.home() / ".bittensor" / "storb_neurons"
        full_path = (
            base_path
            / options.wallet_name
            / options.hotkey_name
            / f"netuid{options.netuid}"
            / neuron_name
        )
        logger.info(f"Full path: {full_path}")

        options.full_path = full_path
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)

    def add_args(self):
        """Add command line arguments"""
        self._parser.add_argument(
            "--wallet-name",
            type=str,
            help="Name of the wallet",
            default=self.settings.get("wallet_name", "default"),
        )
        self._parser.add_argument(
            "--hotkey-name",
            type=str,
            help="Name of the hotkey",
            default=self.settings.get("hotkey_name", "default"),
        )
        self._parser.add_argument(
            "--netuid",
            type=int,
            help="Subnet netuid",
            default=self.settings.get("subtensor", {}).get("netuid", 1),
        )
