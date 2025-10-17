import dotenv

from core.config import Config, ConfigOpts
from core.constants import NeuronType
from core.storage import load_s3_config

dotenv.load_dotenv()


class EvaluatorConfig(Config):
    def __init__(self):
        opts = ConfigOpts(
            neuron_type=NeuronType.Validator,
            neuron_name="evaluator",
            settings_files=["evaluator.toml"],
        )
        super().__init__(opts)
        self.pg_database = self.settings.get("pg_database")  # type: ignore

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
