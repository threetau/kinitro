"""Add evaluator_connections table for direct evaluator communication.

Revision ID: 020_add_evaluator_connections
Revises: 019_competition_uploads_nonneg
Create Date: 2025-01-15 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "020_add_evaluator_connections"
down_revision = "019_competition_uploads_nonneg"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "evaluator_connections",
        sa.Column("id", sa.BigInteger(), primary_key=True),
        sa.Column(
            "evaluator_id", sa.String(128), nullable=False, unique=True, index=True
        ),
        sa.Column(
            "api_key_id",
            sa.BigInteger(),
            sa.ForeignKey("api_keys.id"),
            nullable=True,
            index=True,
        ),
        sa.Column(
            "supported_task_types",
            sa.JSON(),
            nullable=False,
            server_default='["rl_rollout"]',
        ),
        sa.Column(
            "max_concurrent_jobs", sa.Integer(), nullable=False, server_default="1"
        ),
        sa.Column(
            "current_job_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("is_connected", sa.Boolean(), nullable=False, server_default="true"),
        sa.Column(
            "last_heartbeat",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "first_connected_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "total_jobs_assigned", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_jobs_completed", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "total_jobs_failed", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("capabilities", sa.JSON(), nullable=True),
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
            onupdate=sa.func.now(),
        ),
        sa.CheckConstraint(
            "current_job_count >= 0", name="ck_evaluator_concurrent_non_negative"
        ),
        sa.CheckConstraint(
            "max_concurrent_jobs > 0", name="ck_evaluator_max_concurrent_positive"
        ),
        sa.CheckConstraint(
            "total_jobs_assigned >= 0", name="ck_evaluator_jobs_assigned_non_negative"
        ),
        sa.CheckConstraint(
            "total_jobs_completed >= 0", name="ck_evaluator_jobs_completed_non_negative"
        ),
        sa.CheckConstraint(
            "total_jobs_failed >= 0", name="ck_evaluator_jobs_failed_non_negative"
        ),
    )
    op.create_index(
        "ix_evaluator_connections_connected", "evaluator_connections", ["is_connected"]
    )
    op.create_index(
        "ix_evaluator_connections_heartbeat",
        "evaluator_connections",
        ["last_heartbeat"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_evaluator_connections_heartbeat", table_name="evaluator_connections"
    )
    op.drop_index(
        "ix_evaluator_connections_connected", table_name="evaluator_connections"
    )
    op.drop_table("evaluator_connections")
