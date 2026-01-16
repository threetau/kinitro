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

        self._parser.add_argument(
            "--backend-url",
            type=str,
            help="Backend HTTP URL for fetching weights",
            default=self.settings.get("backend_url", "http://localhost:8080"),
        )

        self._parser.add_argument(
            "--weight-poll-interval",
            type=int,
            help="Seconds between weight polling cycles",
            default=self.settings.get("weight_poll_interval", 300),
        )

        self._parser.add_argument(
            "--weights-url",
            type=str,
            help="HTTP endpoint that exposes weight snapshots (deprecated, use backend-url)",
            default=self.settings.get("weights_url", "https://api.kinitro.ai/weights"),
        )

        self._parser.add_argument(
            "--weights-poll-interval",
            type=float,
            help="Seconds between weight snapshot polls (deprecated, use weight-poll-interval)",
            default=self.settings.get("weights_poll_interval", 300.0),
        )

        self._parser.add_argument(
            "--weights-request-timeout",
            type=float,
            help="Timeout (seconds) for weight snapshot HTTP requests",
            default=self.settings.get("weights_request_timeout", 30.0),
        )

        self._parser.add_argument(
            "--weights-stale-threshold",
            type=float,
            help="Warn if backend weight snapshot is older than this many seconds",
            default=self.settings.get("weights_stale_threshold", 900.0),
        )
