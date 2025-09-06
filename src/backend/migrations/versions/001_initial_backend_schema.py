"""Initial backend schema for Kinitro

Revision ID: 001
Revises:
Create Date: 2024-01-01 00:00:00.000000

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Create competitions table
    op.create_table(
        "competitions",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=256), nullable=False),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("benchmarks", sa.JSON(), nullable=False),
        sa.Column("points", sa.Integer(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column("start_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("end_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("points > 0", name="ck_competition_points_positive"),
        sa.CheckConstraint(
            "end_time IS NULL OR start_time IS NULL OR end_time > start_time",
            name="ck_competition_times_ordered",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name"),
    )

    # Create indexes for competitions
    op.create_index("ix_competitions_active", "competitions", ["active"])
    op.create_index("ix_competitions_created_at", "competitions", ["created_at"])
    op.create_index("ix_competitions_points", "competitions", ["points"])

    # Create miner_submissions table
    op.create_table(
        "miner_submissions",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("miner_hotkey", sa.String(length=48), nullable=False),
        sa.Column("competition_id", sa.String(length=64), nullable=False),
        sa.Column("hf_repo_id", sa.String(length=256), nullable=False),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("commitment_block", sa.BigInteger(), nullable=False),
        sa.Column("commitment_hash", sa.String(length=128), nullable=True),
        sa.Column(
            "submission_time",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competitions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "miner_hotkey",
            "competition_id",
            "version",
            name="uq_miner_competition_version",
        ),
    )

    # Create indexes for miner_submissions
    op.create_index(
        "ix_miner_submissions_miner_hotkey", "miner_submissions", ["miner_hotkey"]
    )
    op.create_index(
        "ix_miner_submissions_competition_id", "miner_submissions", ["competition_id"]
    )
    op.create_index(
        "ix_miner_submissions_miner_competition",
        "miner_submissions",
        ["miner_hotkey", "competition_id"],
    )
    op.create_index(
        "ix_miner_submissions_block", "miner_submissions", ["commitment_block"]
    )
    op.create_index(
        "ix_miner_submissions_time", "miner_submissions", ["submission_time"]
    )

    # Create backend_evaluation_jobs table
    op.create_table(
        "backend_evaluation_jobs",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("submission_id", sa.BigInteger(), nullable=False),
        sa.Column("competition_id", sa.String(length=64), nullable=False),
        sa.Column("miner_hotkey", sa.String(length=48), nullable=False),
        sa.Column("hf_repo_id", sa.String(length=256), nullable=False),
        sa.Column("env_provider", sa.String(length=64), nullable=False),
        sa.Column("benchmark_name", sa.String(length=128), nullable=False),
        sa.Column("config", sa.JSON(), nullable=False),
        sa.Column("broadcast_time", sa.DateTime(timezone=True), nullable=True),
        sa.Column("validators_sent", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "validators_completed", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "validators_sent >= 0", name="ck_validators_sent_non_negative"
        ),
        sa.CheckConstraint(
            "validators_completed >= 0", name="ck_validators_completed_non_negative"
        ),
        sa.CheckConstraint(
            "validators_completed <= validators_sent",
            name="ck_validators_completed_within_sent",
        ),
        sa.ForeignKeyConstraint(
            ["competition_id"],
            ["competitions.id"],
        ),
        sa.ForeignKeyConstraint(
            ["submission_id"],
            ["miner_submissions.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    # Create indexes for backend_evaluation_jobs
    op.create_index(
        "ix_backend_evaluation_jobs_job_id", "backend_evaluation_jobs", ["id"]
    )
    op.create_index(
        "ix_backend_evaluation_jobs_submission_id",
        "backend_evaluation_jobs",
        ["submission_id"],
    )
    op.create_index(
        "ix_backend_evaluation_jobs_competition_id",
        "backend_evaluation_jobs",
        ["competition_id"],
    )
    op.create_index(
        "ix_backend_evaluation_jobs_broadcast",
        "backend_evaluation_jobs",
        ["broadcast_time"],
    )
    op.create_index(
        "ix_backend_evaluation_jobs_miner", "backend_evaluation_jobs", ["miner_hotkey"]
    )

    # Create backend_evaluation_results table
    op.create_table(
        "backend_evaluation_results",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("backend_job_id", sa.BigInteger(), nullable=False),
        sa.Column("validator_hotkey", sa.String(length=48), nullable=False),
        sa.Column("miner_hotkey", sa.String(length=48), nullable=False),
        sa.Column("competition_id", sa.String(length=64), nullable=False),
        sa.Column("benchmark", sa.String(length=128), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("success_rate", sa.Float(), nullable=True),
        sa.Column("avg_reward", sa.Float(), nullable=True),
        sa.Column("total_episodes", sa.Integer(), nullable=True),
        sa.Column("logs", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("extra_data", sa.JSON(), nullable=True),
        sa.Column(
            "result_time",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("score >= 0", name="ck_score_non_negative"),
        sa.CheckConstraint(
            "success_rate IS NULL OR (success_rate >= 0 AND success_rate <= 1)",
            name="ck_success_rate_range",
        ),
        sa.CheckConstraint(
            "total_episodes IS NULL OR total_episodes > 0", name="ck_episodes_positive"
        ),
        sa.ForeignKeyConstraint(
            ["backend_job_id"],
            ["backend_evaluation_jobs.id"],
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "backend_job_id",
            "validator_hotkey",
            "benchmark",
            name="uq_job_validator_benchmark",
        ),
    )

    # Create indexes for backend_evaluation_results
    op.create_index(
        "ix_backend_evaluation_results_job_id", "backend_evaluation_results", ["id"]
    )
    op.create_index(
        "ix_backend_evaluation_results_backend_job_id",
        "backend_evaluation_results",
        ["backend_job_id"],
    )
    op.create_index(
        "ix_backend_evaluation_results_validator",
        "backend_evaluation_results",
        ["validator_hotkey"],
    )
    op.create_index(
        "ix_backend_evaluation_results_miner",
        "backend_evaluation_results",
        ["miner_hotkey"],
    )
    op.create_index(
        "ix_backend_evaluation_results_competition",
        "backend_evaluation_results",
        ["competition_id"],
    )
    op.create_index(
        "ix_backend_evaluation_results_benchmark",
        "backend_evaluation_results",
        ["benchmark"],
    )
    op.create_index(
        "ix_backend_evaluation_results_score", "backend_evaluation_results", ["score"]
    )
    op.create_index(
        "ix_backend_evaluation_results_time",
        "backend_evaluation_results",
        ["result_time"],
    )

    # Create validator_connections table
    op.create_table(
        "validator_connections",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("validator_hotkey", sa.String(length=48), nullable=False),
        sa.Column("connection_id", sa.String(length=128), nullable=False),
        sa.Column(
            "first_connected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_connected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "last_heartbeat",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("total_jobs_sent", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "total_results_received", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("total_errors", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("is_connected", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint("total_jobs_sent >= 0", name="ck_jobs_sent_non_negative"),
        sa.CheckConstraint(
            "total_results_received >= 0", name="ck_results_received_non_negative"
        ),
        sa.CheckConstraint("total_errors >= 0", name="ck_errors_non_negative"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("validator_hotkey"),
    )

    # Create indexes for validator_connections
    op.create_index(
        "ix_validator_connections_validator_hotkey",
        "validator_connections",
        ["validator_hotkey"],
    )
    op.create_index(
        "ix_validator_connections_connection_id",
        "validator_connections",
        ["connection_id"],
    )
    op.create_index(
        "ix_validator_connections_heartbeat",
        "validator_connections",
        ["last_heartbeat"],
    )
    op.create_index(
        "ix_validator_connections_connected", "validator_connections", ["is_connected"]
    )

    # Create backend_state table (singleton for service state)
    op.create_table(
        "backend_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "last_seen_block", sa.BigInteger(), nullable=False, server_default="0"
        ),
        sa.Column("last_chain_scan", sa.DateTime(timezone=True), nullable=True),
        sa.Column("service_version", sa.String(length=32), nullable=True),
        sa.Column(
            "service_start_time",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.CheckConstraint(
            "last_seen_block >= 0", name="ck_backend_block_non_negative"
        ),
        sa.CheckConstraint("id = 1", name="ck_backend_state_singleton"),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    # Drop tables in reverse order of creation due to foreign key dependencies
    op.drop_table("backend_state")
    op.drop_table("validator_connections")
    op.drop_table("backend_evaluation_results")
    op.drop_table("backend_evaluation_jobs")
    op.drop_table("miner_submissions")
    op.drop_table("competitions")
