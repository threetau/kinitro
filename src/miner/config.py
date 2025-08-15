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

        # Hugging Face configuration
        self._parser.add_argument(
            "--hf-repo-id",
            type=str,
            help="Hugging Face repository ID (e.g., username/repo-name)",
            default=self.settings.get("hf_repo_id"),
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
            help="Command to execute: 'upload' to upload submission to HF, 'commit' to commit to substrate chain",
        )
