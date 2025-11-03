from enum import Enum

from core.config import Config, ConfigOpts
from core.constants import NeuronType


class ValidatorMode(str, Enum):
    FULL = "full"
    LITE = "lite"


class ValidatorConfig(Config):
    def __init__(self):
        opts = ConfigOpts(
            neuron_name="validator",
            neuron_type=NeuronType.Validator,
            settings_files=["validator.toml"],
        )
        super().__init__(opts)
        self._normalize_validator_mode()

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
            choices=tuple(mode.value for mode in ValidatorMode),
            help="Validator service mode",
            default=self.settings.get("validator_mode", ValidatorMode.FULL.value),
        )

        self._parser.add_argument(
            "--weights-url",
            type=str,
            help="HTTP endpoint that exposes weight snapshots",
            default=self.settings.get("weights_url", "https://api.kinitro.ai/weights"),
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

    def _normalize_validator_mode(self) -> None:
        """Ensure validator_mode is set to a supported value."""

        raw_mode = self.settings.get("validator_mode", ValidatorMode.FULL.value)
        if isinstance(raw_mode, ValidatorMode):
            normalized = raw_mode.value
        else:
            normalized = str(raw_mode).lower()

        try:
            ValidatorMode(normalized)
        except ValueError as exc:
            raise ValueError(f"Invalid validator_mode '{raw_mode}'") from exc

        self.settings["validator_mode"] = normalized
