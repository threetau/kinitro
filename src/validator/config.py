import sys
from pathlib import Path

from dynaconf import Dynaconf
from core.config import Config, ConfigOpts
from core.constants import NeuronType


class ValidatorConfig(Config):
    def __init__(self):
        # First, do a preliminary parse to check for --config argument
        custom_config_file = self._get_config_file_from_args()

        # Set up the settings files list
        settings_files = ["validator.toml"]
        if custom_config_file:
            settings_files = [custom_config_file]

        opts = ConfigOpts(
            neuron_name="validator",
            neuron_type=NeuronType.Validator,
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

        # pg database
        self._parser.add_argument(
            "--pg-database",
            type=str,
            help="PostgreSQL database URL",
            default=self.settings.get(
                "pg_database", "postgresql://user:password@localhost/dbname"
            ),  # type: ignore
        )

        # duckdb database
        self._parser.add_argument(
            "--duckdb-path",
            type=str,
            help="Path to DuckDB database file",
            default=self.settings.get("duckdb_path", "evaluation_data.duckdb"),
        )

        # parent/child validator configuration
        self._parser.add_argument(
            "--is-parent",
            action="store_true",
            help="Run as parent validator that distributes jobs to children",
            default=self.settings.get("is_parent", False),
        )

        self._parser.add_argument(
            "--parent-host",
            type=str,
            help="Parent validator hostname (for child validators)",
            default=self.settings.get("parent_host", "localhost"),
        )

        self._parser.add_argument(
            "--parent-port",
            type=int,
            help="Parent validator WebSocket port (for child validators)",
            default=self.settings.get("parent_port", 8765),
        )

        self._parser.add_argument(
            "--broadcast-host",
            type=str,
            help="Host to bind WebSocket server (for parent validators)",
            default=self.settings.get("broadcast_host", "localhost"),
        )

        self._parser.add_argument(
            "--broadcast-port",
            type=int,
            help="Port to bind WebSocket server (for parent validators)",
            default=self.settings.get("broadcast_port", 8765),
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
