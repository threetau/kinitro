from dynaconf import Dynaconf

from core.config import Config, ConfigOpts
from core.constants import NeuronType


class MinerConfig(Config):
    def __init__(self):
        opts = ConfigOpts(
            neuron_name="miner",
            neuron_type=NeuronType.Miner,
            settings_files=["miner.toml"],
        )
        super().__init__(opts)
        self._parser = super()._parser
