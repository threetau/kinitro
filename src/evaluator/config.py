from core.config import Config, ConfigOpts


class EvaluatorConfig(Config):
    def __init__(self):
        opts = ConfigOpts(
            settings_files=["evaluator.toml"],
        )
        super().__init__(opts)
        self._parser = super()._parser
