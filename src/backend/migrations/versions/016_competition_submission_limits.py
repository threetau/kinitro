"""Add submission size and rate limit columns to competitions.

Revision ID: 016_comp_submission_limits
Revises: 015_episode_unique_per_validator
Create Date: 2024-11-22 00:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "016_comp_submission_limits"
down_revision = "015_episode_unique_per_validator"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "competitions",
        sa.Column("submission_max_size_bytes", sa.BigInteger(), nullable=True),
    )
    op.add_column(
        "competitions",
        sa.Column("submission_upload_window_seconds", sa.Integer(), nullable=True),
    )
    op.add_column(
        "competitions",
        sa.Column("submission_uploads_per_window", sa.Integer(), nullable=True),
    )

    op.create_check_constraint(
        "ck_competition_submission_max_size_positive",
        "competitions",
        "submission_max_size_bytes IS NULL OR submission_max_size_bytes > 0",
    )
    op.create_check_constraint(
        "ck_competition_upload_window_positive",
        "competitions",
        "submission_upload_window_seconds IS NULL OR submission_upload_window_seconds > 0",
    )
    op.create_check_constraint(
        "ck_competition_uploads_per_window_positive",
        "competitions",
        "submission_uploads_per_window IS NULL OR submission_uploads_per_window > 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_competition_uploads_per_window_positive",
        "competitions",
        type_="check",
    )
    op.drop_constraint(
        "ck_competition_upload_window_positive",
        "competitions",
        type_="check",
    )
    op.drop_constraint(
        "ck_competition_submission_max_size_positive",
        "competitions",
        type_="check",
    )
    op.drop_column("competitions", "submission_uploads_per_window")
    op.drop_column("competitions", "submission_upload_window_seconds")
    op.drop_column("competitions", "submission_max_size_bytes")
