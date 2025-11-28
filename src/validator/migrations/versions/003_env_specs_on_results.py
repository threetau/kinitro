"""Store env_specs on validator evaluation results.

Revision ID: 003_env_specs_on_results
Revises: 002_artifact_cols
Create Date: 2025-01-01 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op

revision = "003_env_specs_on_results"
down_revision = "002_artifact_cols"
branch_labels = None
depends_on = None


def upgrade() -> None:
    with op.batch_alter_table("validator_evaluation_results") as batch_op:
        batch_op.add_column(sa.Column("env_specs", sa.JSON(), nullable=True))


def downgrade() -> None:
    with op.batch_alter_table("validator_evaluation_results") as batch_op:
        batch_op.drop_column("env_specs")
