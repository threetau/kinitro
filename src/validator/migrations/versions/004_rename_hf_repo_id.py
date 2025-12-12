"""Rename hf_repo_id to repo_id

Since we no longer use Hugging Face to store submissions, rename the column
to the more generic 'repo_id'.

Revision ID: 004
Revises: 003_env_specs_on_results
Create Date: 2025-12-09
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "004_rename_hf_repo_id"
down_revision = "003_env_specs_on_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename column in validator_evaluation_jobs table
    op.alter_column(
        "validator_evaluation_jobs",
        "hf_repo_id",
        new_column_name="repo_id",
    )


def downgrade() -> None:
    # Revert column name in validator_evaluation_jobs table
    op.alter_column(
        "validator_evaluation_jobs",
        "repo_id",
        new_column_name="hf_repo_id",
    )
