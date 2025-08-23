from core.config import Config, ConfigOpts
from core.constants import NeuronType


class ValidatorConfig(Config):
    def __init__(self):
        opts = ConfigOpts(
            neuron_name="validator",
            neuron_type=NeuronType.Validator,
            settings_files=["validator.toml"],
        )
        super().__init__(opts)

    def add_args(self):
        """Add command line arguments"""
        super().add_args()
        # pg database
        self._parser.add_argument(
            "--pg-database",
            type=str,
            help="PostgreSQL database URL",
            default=self.settings.get(
                "pg_database", "postgresql://user:password@localhost/dbname"
            ),  # type: ignore
        )
