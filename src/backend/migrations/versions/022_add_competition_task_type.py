"""Add task_type column to competitions for executor dispatch.

Revision ID: 022_add_competition_task_type
Revises: 021_add_evaluator_role
Create Date: 2025-01-15 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "022_add_competition_task_type"
down_revision = "021_add_evaluator_role"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add task_type column for executor dispatch."""
    op.add_column(
        "competitions",
        sa.Column(
            "task_type",
            sa.String(64),
            nullable=False,
            server_default="rl_rollout",
        ),
    )


def downgrade() -> None:
    """Remove task_type column from competitions."""
    op.drop_column("competitions", "task_type")
