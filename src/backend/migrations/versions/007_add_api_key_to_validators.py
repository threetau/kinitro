"""Add API key reference to validator connections

Revision ID: 007_add_api_key_to_validators
Revises: 006_add_api_keys_table
Create Date: 2025-01-20

"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "007_add_api_key_to_validators"
down_revision: Union[str, None] = "006_add_api_keys_table"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add api_key_id column to validator_connections table."""
    # Add api_key_id column
    op.add_column(
        "validator_connections",
        sa.Column("api_key_id", sa.BigInteger(), nullable=True),
    )

    # Create foreign key constraint
    op.create_foreign_key(
        "fk_validator_connections_api_key",
        "validator_connections",
        "api_keys",
        ["api_key_id"],
        ["id"],
    )

    # Create index for api_key_id
    op.create_index(
        "ix_validator_connections_api_key_id",
        "validator_connections",
        ["api_key_id"],
    )


def downgrade() -> None:
    """Remove api_key_id column from validator_connections table."""
    # Drop index
    op.drop_index("ix_validator_connections_api_key_id", "validator_connections")

    # Drop foreign key constraint
    op.drop_constraint(
        "fk_validator_connections_api_key", "validator_connections", type_="foreignkey"
    )

    # Drop column
    op.drop_column("validator_connections", "api_key_id")
