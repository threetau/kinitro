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
            choices=["upload", "commit", "local-eval"],
            help=(
                "Command to execute: 'upload' to send submission artifact, "
                "'commit' to commit to the substrate chain, "
                "'local-eval' to run a local benchmark against your agent"
            ),
        )

        # Local evaluation settings
        local_eval = self.settings.get("local_eval", {}) or {}
        agent_port_default = local_eval.get("agent_port", 8000)
        if agent_port_default is None:
            agent_port_default = 8000
        agent_start_timeout_default = local_eval.get("agent_start_timeout", 30)
        if agent_start_timeout_default is None:
            agent_start_timeout_default = 30
        ray_num_cpus_default = local_eval.get("ray_num_cpus", 2)
        if ray_num_cpus_default is None:
            ray_num_cpus_default = 2
        ray_num_gpus_default = local_eval.get("ray_num_gpus", 0)
        if ray_num_gpus_default is None:
            ray_num_gpus_default = 0

        self._parser.add_argument(
            "--agent-host",
            type=str,
            help="Hostname or IP address where your agent RPC server listens",
            default=local_eval.get("agent_host", "127.0.0.1"),
        )
        self._parser.add_argument(
            "--agent-port",
            type=int,
            help="Port where your agent RPC server listens",
            default=int(agent_port_default),
        )
        self._parser.add_argument(
            "--agent-start-cmd",
            type=str,
            help="Optional shell command to launch your agent server before evaluation",
            default=local_eval.get("agent_start_cmd"),
        )
        self._parser.add_argument(
            "--agent-start-timeout",
            type=float,
            help="Seconds to wait for the agent RPC server to accept connections",
            default=float(agent_start_timeout_default),
        )
        self._parser.add_argument(
            "--benchmark-spec-file",
            type=str,
            help=(
                "Path to a JSON or TOML file containing benchmark specs. "
                "This is required for local evaluation."
            ),
            default=local_eval.get("benchmark_spec_file"),
        )
        self._parser.add_argument(
            "--ray-num-cpus",
            type=float,
            help="Number of Ray CPUs to reserve for the local rollout worker",
            default=ray_num_cpus_default,
        )
        self._parser.add_argument(
            "--ray-num-gpus",
            type=float,
            help="Number of Ray GPUs to reserve for the local rollout worker",
            default=ray_num_gpus_default,
        )
        self._parser.add_argument(
            "--local-results-dir",
            type=str,
            help="Directory to store local evaluation artifacts and summaries",
            default=local_eval.get("local_results_dir", ".kinitro/miner_runs"),
        )
