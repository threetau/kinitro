from core.config import Config, ConfigOpts
from core.constants import NeuronType


class EvaluatorConfig(Config):
    def __init__(self):
        opts = ConfigOpts(
            neuron_type=NeuronType.Validator,
            neuron_name="evaluator",
            settings_files=["evaluator.toml"],
        )
        super().__init__(opts)
        self.pg_database = self.settings.get("pg_database")  # type: ignore
        self.duck_db = self.settings.get("duck_db")  # type: ignore

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

        # duck db
        self._parser.add_argument(
            "--duck-db",
            type=str,
            help="DuckDB file path",
            default=self.settings.get("duck_db", "data/evaluator.duckdb"),  # type: ignore
        )
