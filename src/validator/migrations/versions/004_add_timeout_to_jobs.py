"""Add timeout_seconds to validator evaluation jobs.

Revision ID: 004_add_timeout_to_jobs
Revises: 003_env_specs_on_results
Create Date: 2025-01-01 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "004_add_timeout_to_jobs"
down_revision = "003_env_specs_on_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("validator_evaluation_jobs") as batch_op:
        batch_op.add_column(sa.Column("timeout_seconds", sa.Integer(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("validator_evaluation_jobs") as batch_op:
        batch_op.drop_column("timeout_seconds")
