"""Add evaluator role to api_keys constraint.

Revision ID: 021_add_evaluator_role
Revises: 020_add_evaluator_connections
Create Date: 2025-01-15 00:00:00
"""

from __future__ import annotations

from alembic import op

revision = "021_add_evaluator_role"
down_revision = "020_add_evaluator_connections"
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add 'evaluator' to the valid roles for api_keys."""
    op.drop_constraint("ck_api_keys_valid_role", "api_keys", type_="check")
    op.create_check_constraint(
        "ck_api_keys_valid_role",
        "api_keys",
        "role IN ('admin', 'validator', 'evaluator', 'viewer')",
    )


def downgrade() -> None:
    """Remove 'evaluator' from the valid roles for api_keys."""
    op.drop_constraint("ck_api_keys_valid_role", "api_keys", type_="check")
    op.create_check_constraint(
        "ck_api_keys_valid_role",
        "api_keys",
        "role IN ('admin', 'validator', 'viewer')",
    )
