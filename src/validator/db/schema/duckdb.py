import duckdb


class DuckDBSchema:
    """Manages DuckDB schema creation and maintenance."""

    EPISODES_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS episodes (
        id BIGINT PRIMARY KEY,
        evaluation_id BIGINT NOT NULL,
        episode_index INTEGER NOT NULL,

        start_time TIMESTAMPTZ,
        end_time TIMESTAMPTZ,
        duration_seconds DOUBLE GENERATED ALWAYS AS (
            CASE
                WHEN start_time IS NOT NULL AND end_time IS NOT NULL
                THEN EXTRACT(EPOCH FROM (end_time - start_time))
                ELSE NULL
            END
        ) VIRTUAL,

        total_reward DOUBLE,
        success BOOLEAN NOT NULL DEFAULT false,
        num_steps INTEGER DEFAULT 0,

        initial_state JSON,
        final_state JSON,

        memory_peak_mb INTEGER,    -- Peak memory usage

        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
    """

    EPISODE_STEPS_TABLE_SQL = """
    CREATE TABLE IF NOT EXISTS episode_steps (
        id BIGINT PRIMARY KEY,
        episode_id BIGINT NOT NULL,  -- Links to episodes.id
        step_index INTEGER NOT NULL,

        reward DOUBLE NOT NULL,

        action JSON NOT NULL,
        observation STRING NOT NULL,

        step_timestamp TIMESTAMPTZ,
        duration FLOAT,

        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

        partition_date DATE AS (DATE(created_at))
    )
    """

    # Index creation SQL
    EPISODES_INDEXES = [
        "CREATE INDEX IF NOT EXISTS ix_episodes_evaluation_id ON episodes(evaluation_id)",
        "CREATE INDEX IF NOT EXISTS ix_episodes_success_reward ON episodes(success, total_reward)",
        "CREATE INDEX IF NOT EXISTS ix_episodes_created_at ON episodes(created_at)",
        "CREATE INDEX IF NOT EXISTS ix_episodes_duration ON episodes(duration_seconds)",
        "CREATE INDEX IF NOT EXISTS ix_episodes_eval_success_reward ON episodes(evaluation_id, success, total_reward)",
        "CREATE INDEX IF NOT EXISTS ix_episodes_eval_episode_idx ON episodes(evaluation_id, episode_index)",
    ]

    EPISODE_STEPS_INDEXES = [
        "CREATE INDEX IF NOT EXISTS ix_episode_steps_episode_id ON episode_steps(episode_id)",
        "CREATE INDEX IF NOT EXISTS ix_episode_steps_step_index ON episode_steps(step_index)",
        "CREATE INDEX IF NOT EXISTS ix_episode_steps_reward ON episode_steps(reward)",
        "CREATE INDEX IF NOT EXISTS ix_episode_steps_episode_step ON episode_steps(episode_id, step_index)",
    ]

    @classmethod
    def create_database(cls, db_path: str = ":memory:") -> duckdb.DuckDBPyConnection:
        """Create and initialize DuckDB database with schema."""
        conn = duckdb.connect(db_path)

        # Install required extensions
        conn.execute("INSTALL json")
        conn.execute("LOAD json")

        return conn

    @classmethod
    def initialize_schema(cls, conn: duckdb.DuckDBPyConnection) -> None:
        """Initialize all tables, indexes, and views."""

        # Create main tables
        conn.execute(cls.EPISODES_TABLE_SQL)
        conn.execute(cls.EPISODE_STEPS_TABLE_SQL)

        # Create indexes for performance
        for index_sql in cls.EPISODES_INDEXES:
            conn.execute(index_sql)

        for index_sql in cls.EPISODE_STEPS_INDEXES:
            conn.execute(index_sql)

    @classmethod
    def optimize_tables(cls, conn: duckdb.DuckDBPyConnection) -> None:
        """Run optimization commands for better performance."""

        # Analyze tables for query optimization
        conn.execute("ANALYZE episodes")
        conn.execute("ANALYZE episode_steps")
        conn.execute("ANALYZE evaluation_episode_stats")

        # Checkpoint to flush data to disk
        conn.execute("CHECKPOINT")


def setup_duckdb_database(
    db_path: str = "evaluation_data.duckdb",
) -> duckdb.DuckDBPyConnection:
    """Set up a new DuckDB database with the evaluation schema."""

    # Create database and connection
    conn = DuckDBSchema.create_database(db_path)

    # Initialize schema
    DuckDBSchema.initialize_schema(conn)

    return conn
