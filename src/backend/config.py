import sys
from pathlib import Path

from core.config import Config, ConfigOpts
from core.constants import NeuronType


class BackendConfig(Config):
    def __init__(self):
        # First, do a preliminary parse to check for --config argument
        custom_config_file = self._get_config_file_from_args()

        # Set up the settings files list
        settings_files = ["backend.toml"]
        if custom_config_file:
            settings_files = [custom_config_file]

        opts = ConfigOpts(
            neuron_name="backend",
            neuron_type=NeuronType.Validator,  # Backend uses validator-like chain access
            settings_files=settings_files,
        )
        super().__init__(opts)

    def _get_config_file_from_args(self):
        """Extract config file from command line arguments before full parsing."""
        config_file = None

        # Look for --config or -c in sys.argv
        for i, arg in enumerate(sys.argv):
            if arg == "--config" or arg == "-c":
                if i + 1 < len(sys.argv):
                    config_file = sys.argv[i + 1]
                    break
            elif arg.startswith("--config="):
                config_file = arg.split("=", 1)[1]
                break

        if config_file:
            config_path = Path(config_file)
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_file}")
            print(f"Using custom config file: {config_file}")

        return config_file

    def add_args(self):
        """Add command line arguments"""
        super().add_args()

        # config file
        self._parser.add_argument(
            "--config",
            "-c",
            type=str,
            help="Path to configuration file (TOML format)",
            default=None,
        )

        # database configuration
        self._parser.add_argument(
            "--database-url",
            type=str,
            help="PostgreSQL database URL for backend",
            default=self.settings.get(
                "database_url",
                "postgresql+asyncpg://postgres@localhost/kinitro_backend",
            ),
        )

        # websocket server configuration
        self._parser.add_argument(
            "--websocket-host",
            type=str,
            help="WebSocket server host to bind to",
            default=self.settings.get("websocket_host", "0.0.0.0"),
        )

        self._parser.add_argument(
            "--websocket-port",
            type=int,
            help="WebSocket server port to bind to",
            default=self.settings.get("websocket_port", 8080),
        )

        # chain monitoring configuration
        self._parser.add_argument(
            "--max-commitment-lookback",
            type=int,
            help="Maximum blocks to look back for commitments",
            default=self.settings.get("max_commitment_lookback", 360),
        )

        self._parser.add_argument(
            "--chain-sync-interval",
            type=int,
            help="Seconds between chain sync operations",
            default=self.settings.get("chain_sync_interval", 30),
        )

        # miner filtering
        self._parser.add_argument(
            "--min-stake-threshold",
            type=float,
            help="Minimum stake threshold for miners to be queried",
            default=self.settings.get("min_stake_threshold", 0.0),
        )
