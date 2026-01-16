"""Add artifact metadata columns to validator evaluation jobs.

Revision ID: 002_artifact_cols
Revises: 001
Create Date: 2025-10-16 05:30:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "002_artifact_cols"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("validator_evaluation_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("artifact_url", sa.String(length=512), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_expires_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_sha256", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_size_bytes", sa.BigInteger(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("validator_evaluation_jobs") as batch_op:
        batch_op.drop_column("artifact_size_bytes")
        batch_op.drop_column("artifact_sha256")
        batch_op.drop_column("artifact_expires_at")
        batch_op.drop_column("artifact_url")
