import os
import uuid

import dotenv

from core.config import Config, ConfigOpts
from core.constants import NeuronType
from core.storage import load_s3_config

dotenv.load_dotenv()

DEFAULT_RPC_HANDSHAKE_MAX_ATTEMPTS = 5
DEFAULT_RPC_HANDSHAKE_RETRY_SECONDS = 2.0

# Backend connection defaults
DEFAULT_BACKEND_WS_URL = "ws://localhost:8080/ws/evaluator"
DEFAULT_RECONNECT_INTERVAL = 5.0
DEFAULT_MAX_RECONNECT_INTERVAL = 60.0
DEFAULT_HEARTBEAT_INTERVAL = 30.0


class EvaluatorConfig(Config):
    def __init__(self):
        opts = ConfigOpts(
            neuron_type=NeuronType.Validator,
            neuron_name="evaluator",
            settings_files=["evaluator.toml"],
        )
        super().__init__(opts)
        self.pg_database = self.settings.get("pg_database")  # type: ignore
        self.log_file = self._normalize_log_file(self.settings.get("log_file"))

        # Backend WebSocket connection settings
        self.backend_ws_url = self.settings.get(
            "backend_ws_url", DEFAULT_BACKEND_WS_URL
        )
        self.evaluator_id = self.settings.get(
            "evaluator_id",
            os.environ.get("EVALUATOR_ID", f"evaluator-{uuid.uuid4().hex[:8]}"),
        )
        self.api_key = os.environ.get("KINITRO_API_KEY")
        self.reconnect_interval = float(
            self.settings.get("reconnect_interval", DEFAULT_RECONNECT_INTERVAL)
        )
        self.max_reconnect_interval = float(
            self.settings.get("max_reconnect_interval", DEFAULT_MAX_RECONNECT_INTERVAL)
        )
        self.heartbeat_interval = float(
            self.settings.get("heartbeat_interval", DEFAULT_HEARTBEAT_INTERVAL)
        )

        # Connection mode: "direct" for WebSocket to backend, "pgqueuer" for legacy
        self.connection_mode = self.settings.get("connection_mode", "direct")

        # S3 storage configuration
        self.s3_config = load_s3_config()

        # Episode logging configuration
        self.episode_log_interval = self.settings.get("episode_log_interval", 1)
        self.step_log_interval = self.settings.get("step_log_interval", 1)

        # Concurrent job execution configuration
        self.max_concurrent_jobs = self.settings.get("max_concurrent_jobs", 4)

        # Ray cluster resource hints
        self.ray_num_cpus = self.settings.get("ray_num_cpus", 4)
        self.ray_num_gpus = self.settings.get("ray_num_gpus", 0)
        self.ray_memory = self._maybe_to_bytes(
            self.settings.get("ray_memory_gb"),
        )
        self.ray_object_store_memory = self._maybe_to_bytes(
            self.settings.get("ray_object_store_memory_gb"),
        )

        # Rollout worker actor resource tuning
        self.worker_remote_options = self._build_worker_remote_options()

        # RPC handshake/backoff behavior
        handshake_attempts_value = self.settings.get("rpc_handshake_max_attempts")
        try:
            self.rpc_handshake_max_attempts = max(
                1,
                int(
                    handshake_attempts_value
                    if handshake_attempts_value is not None
                    else DEFAULT_RPC_HANDSHAKE_MAX_ATTEMPTS
                ),
            )
        except (TypeError, ValueError):
            self.rpc_handshake_max_attempts = DEFAULT_RPC_HANDSHAKE_MAX_ATTEMPTS

        handshake_retry_value = self.settings.get("rpc_handshake_retry_seconds")
        try:
            retry_seconds = float(
                handshake_retry_value
                if handshake_retry_value is not None
                else DEFAULT_RPC_HANDSHAKE_RETRY_SECONDS
            )
        except (TypeError, ValueError):
            retry_seconds = DEFAULT_RPC_HANDSHAKE_RETRY_SECONDS
        self.rpc_handshake_retry_seconds = max(0.0, retry_seconds)

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
            "--log-file",
            type=str,
            help="File path to write evaluator logs (and keep stdout logging). Leave empty to disable.",
            default=self.settings.get("log_file", "logs/evaluator.log"),
        )

    def _build_worker_remote_options(self) -> dict:
        """Create Ray.remote options for rollout workers based on settings."""

        options = {
            "max_restarts": self.settings.get("worker_max_restarts", 1),
            "max_task_retries": self.settings.get("worker_max_task_retries", 0),
        }

        worker_num_cpus = self.settings.get("worker_num_cpus")
        if worker_num_cpus is not None:
            options["num_cpus"] = worker_num_cpus

        worker_num_gpus = self.settings.get("worker_num_gpus")
        if worker_num_gpus is not None:
            options["num_gpus"] = worker_num_gpus

        worker_memory = self._maybe_to_bytes(self.settings.get("worker_memory_gb"))
        if worker_memory is not None:
            options["memory"] = worker_memory

        return options

    @staticmethod
    def _maybe_to_bytes(value):
        """Convert a size in GB (float/int) to bytes, returning None if unset."""

        if value is None:
            return None

        try:
            numeric = float(value)
            if numeric <= 0:
                return None
            return int(numeric * (1024**3))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize_log_file(value) -> str | None:
        if value is None:
            return None
        string_value = str(value).strip()
        return string_value or None
