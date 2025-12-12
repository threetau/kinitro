"""Rename hf_repo_id to repo_id

Since we no longer use Hugging Face to store submissions, rename the column
to the more generic 'repo_id'.

Revision ID: 018
Revises: 017_add_env_specs_to_results
Create Date: 2025-12-09
"""

from alembic import op

# revision identifiers, used by Alembic.
revision = "018_rename_hf_repo_id"
down_revision = "017_add_env_specs_to_results"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Rename column in miner_submissions table
    op.alter_column(
        "miner_submissions",
        "hf_repo_id",
        new_column_name="repo_id",
    )

    # Rename column in backend_evaluation_jobs table
    op.alter_column(
        "backend_evaluation_jobs",
        "hf_repo_id",
        new_column_name="repo_id",
    )


def downgrade() -> None:
    # Revert column name in miner_submissions table
    op.alter_column(
        "miner_submissions",
        "repo_id",
        new_column_name="hf_repo_id",
    )

    # Revert column name in backend_evaluation_jobs table
    op.alter_column(
        "backend_evaluation_jobs",
        "repo_id",
        new_column_name="hf_repo_id",
    )
