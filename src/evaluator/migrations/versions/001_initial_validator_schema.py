"""Initial validator database schema

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create evaluation_status enum (if not exists)
    connection = op.get_bind()

    # Check if enum already exists
    result = connection.execute(
        sa.text("""
        SELECT EXISTS (
            SELECT 1 FROM pg_type
            WHERE typname = 'evaluationstatus'
        )
    """)
    )
    enum_exists = result.scalar()

    if not enum_exists:
        evaluation_status = postgresql.ENUM(
            "QUEUED",
            "STARTING",
            "RUNNING",
            "COMPLETED",
            "FAILED",
            "CANCELLED",
            "TIMEOUT",
            name="evaluationstatus",
        )
        evaluation_status.create(connection)

    # Create validator_evaluation_jobs table
    op.create_table(
        "validator_evaluation_jobs",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("submission_id", sa.BigInteger(), nullable=False),
        sa.Column("competition_id", sa.String(length=128), nullable=False),
        sa.Column("miner_hotkey", sa.String(length=48), nullable=False),
        sa.Column("hf_repo_id", sa.String(length=256), nullable=False),
        sa.Column("env_provider", sa.String(length=128), nullable=False),
        sa.Column("benchmark_name", sa.String(length=128), nullable=False),
        sa.Column("config", sa.JSON(), nullable=True),
        sa.Column(
            "status",
            postgresql.ENUM(name="evaluationstatus", create_type=False),
            nullable=False,
            server_default="QUEUED",
        ),
        sa.Column(
            "received_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("max_memory_mb", sa.Integer(), nullable=True),
        sa.Column("max_cpu_percent", sa.Float(), nullable=True),
        sa.Column("max_retries", sa.Integer(), nullable=False, server_default="3"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("container_id", sa.String(length=128), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("random_seed", sa.Integer(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_validator_evaluation_jobs_id"),
        "validator_evaluation_jobs",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_jobs_submission_id"),
        "validator_evaluation_jobs",
        ["submission_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_jobs_competition_id"),
        "validator_evaluation_jobs",
        ["competition_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_jobs_miner_hotkey"),
        "validator_evaluation_jobs",
        ["miner_hotkey"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_jobs_status"),
        "validator_evaluation_jobs",
        ["status"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_jobs_created_at"),
        "validator_evaluation_jobs",
        ["created_at"],
        unique=False,
    )

    # Create validator_evaluation_results table
    op.create_table(
        "validator_evaluation_results",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("job_id", sa.BigInteger(), nullable=False),
        sa.Column("benchmark", sa.String(length=128), nullable=False),
        sa.Column("validator_hotkey", sa.String(length=48), nullable=False),
        sa.Column("miner_hotkey", sa.String(length=48), nullable=False),
        sa.Column("competition_id", sa.String(length=128), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("success_rate", sa.Float(), nullable=True),
        sa.Column("avg_reward", sa.Float(), nullable=True),
        sa.Column("total_episodes", sa.Integer(), nullable=True),
        sa.Column(
            "result_time", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column("computation_time_seconds", sa.Float(), nullable=True),
        sa.Column("logs", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("extra_data", sa.JSON(), nullable=True),
        sa.Column(
            "submitted_to_backend", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("submission_error", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.ForeignKeyConstraint(
            ["job_id"],
            ["validator_evaluation_jobs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_validator_evaluation_results_benchmark"),
        "validator_evaluation_results",
        ["benchmark"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_results_competition_id"),
        "validator_evaluation_results",
        ["competition_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_results_job_id"),
        "validator_evaluation_results",
        ["job_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_results_miner_hotkey"),
        "validator_evaluation_results",
        ["miner_hotkey"],
        unique=False,
    )
    op.create_index(
        op.f("ix_validator_evaluation_results_validator_hotkey"),
        "validator_evaluation_results",
        ["validator_hotkey"],
        unique=False,
    )

    # Create validator_state table
    op.create_table(
        "validator_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("validator_hotkey", sa.String(length=48), nullable=False),
        sa.Column("backend_url", sa.String(length=512), nullable=True),
        sa.Column(
            "connected_to_backend", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column("last_backend_connection", sa.DateTime(), nullable=True),
        sa.Column("last_heartbeat_sent", sa.DateTime(), nullable=True),
        sa.Column("last_heartbeat_ack", sa.DateTime(), nullable=True),
        sa.Column(
            "total_jobs_received", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_jobs_completed", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_jobs_failed", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("validator_version", sa.String(length=32), nullable=True),
        sa.Column("config_hash", sa.String(length=64), nullable=True),
        sa.Column("avg_job_duration_seconds", sa.Float(), nullable=True),
        sa.Column("last_performance_update", sa.DateTime(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.Column(
            "updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("validator_hotkey"),
    )
    op.create_index(
        op.f("ix_validator_state_validator_hotkey"),
        "validator_state",
        ["validator_hotkey"],
        unique=False,
    )


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_table("validator_state")
    op.drop_table("validator_evaluation_results")
    op.drop_table("validator_evaluation_jobs")

    # Check if we should drop the enum (only if no other tables are using it)
    connection = op.get_bind()

    # Check if any backend tables are using the enum
    result = connection.execute(
        sa.text("""
        SELECT EXISTS (
            SELECT 1 FROM information_schema.columns
            WHERE udt_name = 'evaluationstatus'
            AND table_name NOT LIKE 'validator_%'
        )
    """)
    )
    other_tables_using_enum = result.scalar()

    if not other_tables_using_enum:
        # Safe to drop the enum
        evaluation_status = postgresql.ENUM(name="evaluationstatus")
        evaluation_status.drop(connection)
