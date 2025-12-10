"""Add per-competition job timeout and per-job timeout field.

Revision ID: 018_competition_job_timeouts
Revises: 017_env_specs_on_results
Create Date: 2025-01-01 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "018_competition_job_timeouts"
down_revision = "017_env_specs_on_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "competitions",
        sa.Column(
            "job_timeout_seconds",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("7200"),
        ),
    )
    op.add_column(
        "backend_evaluation_jobs",
        sa.Column("timeout_seconds", sa.Integer(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("backend_evaluation_jobs", "timeout_seconds")
    op.drop_column("competitions", "job_timeout_seconds")
