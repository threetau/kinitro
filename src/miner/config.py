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

    def add_args(self):
        """Add miner-specific CLI arguments"""
        super().add_args()

        # Backend configuration
        self._parser.add_argument(
            "--backend-url",
            type=str,
            help="Base URL for the Kinitro backend (e.g., http://localhost:8080)",
            default=self.settings.get("backend_url", "http://localhost:8080"),
        )

        # Submission metadata
        self._parser.add_argument(
            "--submission-version",
            type=str,
            help="Version identifier for this submission (defaults to timestamp)",
            default=self.settings.get("submission_version"),
        )
        self._parser.add_argument(
            "--holdout-seconds",
            type=int,
            help="Optional hold-out duration in seconds for this submission",
            default=self.settings.get("holdout_seconds"),
        )

        self._parser.add_argument(
            "--submission-id",
            type=str,
            help="Submission ID returned by the backend upload endpoint",
            default=self.settings.get("submission_id"),
        )
        self._parser.add_argument(
            "--artifact-sha256",
            type=str,
            help="SHA-256 digest of the submission artifact",
            default=self.settings.get("artifact_sha256"),
        )
        self._parser.add_argument(
            "--artifact-size-bytes",
            type=int,
            help="Size of the submission artifact in bytes",
            default=self.settings.get("artifact_size_bytes"),
        )

        # Substrate/Subtensor configuration
        self._parser.add_argument(
            "--subtensor-network",
            type=str,
            help="Subtensor network to connect to",
            default=self.settings.get("subtensor", {}).get("network", "finney"),
        )
        self._parser.add_argument(
            "--subtensor-address",
            type=str,
            help="Subtensor websocket address",
            default=self.settings.get("subtensor", {}).get("address"),
        )

        # Submission configuration
        self._parser.add_argument(
            "--submission-dir",
            type=str,
            help="Path to submission directory to upload",
            default=self.settings.get(
                "submission_dir", "./evaluator/submissions/default_submission"
            ),
        )

        # CLI commands
        self._parser.add_argument(
            "command",
            choices=["upload", "commit"],
            help="Command to execute: 'upload' to send submission artifact, 'commit' to commit to the substrate chain",
        )
