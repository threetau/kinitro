"""Add submission holdout seconds to competitions.

Revision ID: 010_competition_holdout_seconds
Revises: 009_direct_vault_submission
Create Date: 2025-10-16 06:10:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from backend.constants import DEFAULT_SUBMISSION_HOLDOUT_SECONDS


revision = "010_competition_holdout_seconds"
down_revision = "009_direct_vault_submission"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "competitions",
        sa.Column(
            "submission_holdout_seconds",
            sa.Integer(),
            nullable=False,
            server_default=str(DEFAULT_SUBMISSION_HOLDOUT_SECONDS),
        ),
    )


def downgrade() -> None:
    op.drop_column("competitions", "submission_holdout_seconds")
