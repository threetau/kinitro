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

        # parent-child validator configuration
        self._parser.add_argument(
            "--is-parent-validator",
            action="store_true",
            help="Run as parent validator (distributes jobs to child validators)",
            default=self.settings.get("is_parent_validator", False),
        )

        self._parser.add_argument(
            "--parent-host",
            type=str,
            help="Parent validator host (for child validators)",
            default=self.settings.get("parent_host", "localhost"),
        )

        self._parser.add_argument(
            "--parent-port",
            type=int,
            help="Parent validator RPC port (for child validators)",
            default=self.settings.get("parent_port", 8001),
        )

        self._parser.add_argument(
            "--child-port",
            type=int,
            help="Child validator RPC port (for receiving jobs)",
            default=self.settings.get("child_port", 8002),
        )

        self._parser.add_argument(
            "--validator-id",
            type=str,
            help="Unique validator ID for parent-child communication",
            default=self.settings.get("validator_id", None),
        )
