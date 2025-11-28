"""Add env_specs column to backend evaluation results.

Revision ID: 017_env_specs_on_results
Revises: 016_comp_submission_limits
Create Date: 2025-01-01 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "017_env_specs_on_results"
down_revision = "016_comp_submission_limits"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "backend_evaluation_results",
        sa.Column("env_specs", sa.JSON(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("backend_evaluation_results", "env_specs")
