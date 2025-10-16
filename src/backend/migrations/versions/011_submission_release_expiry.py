"""Add release URL expiry to miner submissions.

Revision ID: 011_submission_release_expiry
Revises: 010_competition_holdout_seconds
Create Date: 2025-10-16 06:20:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "011_submission_release_expiry"
down_revision = "010_competition_holdout_seconds"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "miner_submissions",
        sa.Column(
            "public_artifact_url_expires_at", sa.DateTime(timezone=True), nullable=True
        ),
    )


def downgrade() -> None:
    op.drop_column("miner_submissions", "public_artifact_url_expires_at")
