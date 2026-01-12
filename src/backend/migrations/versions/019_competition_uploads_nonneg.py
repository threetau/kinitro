"""Allow zero uploads per window in competitions.

Revision ID: 019_competition_uploads_nonneg
Revises: 018_competition_job_timeouts
Create Date: 2025-02-15 00:00:00
"""

from __future__ import annotations

from alembic import op

revision = "019_competition_uploads_nonneg"
down_revision = "018_competition_job_timeouts"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.drop_constraint(
        "ck_competition_uploads_per_window_positive",
        "competitions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_competition_uploads_per_window_positive",
        "competitions",
        "submission_uploads_per_window IS NULL OR submission_uploads_per_window >= 0",
    )


def downgrade() -> None:
    op.drop_constraint(
        "ck_competition_uploads_per_window_positive",
        "competitions",
        type_="check",
    )
    op.create_check_constraint(
        "ck_competition_uploads_per_window_positive",
        "competitions",
        "submission_uploads_per_window IS NULL OR submission_uploads_per_window > 0",
    )
