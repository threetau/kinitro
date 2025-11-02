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

        self._parser.add_argument(
            "--backend-url",
            type=str,
            help="Backend WebSocket URL for validator connections",
            default=self.settings.get(
                "backend_url", "ws://localhost:8080/ws/validator"
            ),
        )

        self._parser.add_argument(
            "--reconnect-interval",
            type=int,
            help="Seconds to wait before reconnecting to backend",
            default=self.settings.get("reconnect_interval", 5),
        )

        self._parser.add_argument(
            "--heartbeat-interval",
            type=int,
            help="Seconds between heartbeat messages to backend",
            default=self.settings.get("heartbeat_interval", 30),
        )

        self._parser.add_argument(
            "--validator-mode",
            type=str,
            choices=("websocket", "lite"),
            help="Validator service mode",
            default=self.settings.get("validator_mode", "websocket"),
        )

        self._parser.add_argument(
            "--weights-url",
            type=str,
            help="HTTP endpoint that exposes weight snapshots",
            default=self.settings.get("weights_url", "http://api.kinitro.ai/weights"),
        )

        self._parser.add_argument(
            "--weights-poll-interval",
            type=float,
            help="Seconds between weight snapshot polls in lite mode",
            default=self.settings.get("weights_poll_interval", 30.0),
        )

        self._parser.add_argument(
            "--weights-request-timeout",
            type=float,
            help="Timeout (seconds) for weight snapshot HTTP requests in lite mode",
            default=self.settings.get("weights_request_timeout", 10.0),
        )

        self._parser.add_argument(
            "--weights-stale-threshold",
            type=float,
            help="Warn if backend weight snapshot is older than this many seconds",
            default=self.settings.get("weights_stale_threshold", 180.0),
        )
