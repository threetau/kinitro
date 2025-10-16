"""Add submission upload metadata for direct vault workflow.

Revision ID: 009_direct_vault_submission
Revises: 008_add_validator_hotkeys_to_episode_tables
Create Date: 2025-10-16 04:00:00
"""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa

from sqlmodel.sql.sqltypes import AutoString


# revision identifiers, used by Alembic.
revision = "009_direct_vault_submission"
down_revision = "008_validator_hotkeys_episode"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "submission_uploads",
        sa.Column("submission_id", sa.BigInteger(), nullable=False),
        sa.Column("miner_hotkey", sa.String(length=48), nullable=False, index=True),
        sa.Column("competition_id", sa.String(length=64), nullable=False, index=True),
        sa.Column("version", sa.String(length=32), nullable=False),
        sa.Column("artifact_object_key", sa.String(length=512), nullable=False),
        sa.Column("artifact_sha256", sa.String(length=64), nullable=False),
        sa.Column("artifact_size_bytes", sa.BigInteger(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "READY",
                "PROCESSED",
                "EXPIRED",
                name="submission_upload_status",
            ),
            nullable=False,
            server_default="PENDING",
        ),
        sa.Column("upload_url_expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("uploaded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "holdout_seconds", sa.Integer(), nullable=False, server_default="3600"
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(),
            nullable=False,
            server_default=sa.text("CURRENT_TIMESTAMP"),
        ),
        sa.PrimaryKeyConstraint("submission_id"),
        sa.ForeignKeyConstraint(["competition_id"], ["competitions.id"]),
    )
    op.create_index(
        "ix_submission_uploads_hotkey", "submission_uploads", ["miner_hotkey"]
    )
    op.create_index("ix_submission_uploads_status", "submission_uploads", ["status"])

    with op.batch_alter_table("miner_submissions") as batch_op:
        batch_op.add_column(
            sa.Column("artifact_object_key", sa.String(length=512), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_sha256", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_size_bytes", sa.BigInteger(), nullable=True)
        )
        batch_op.add_column(
            sa.Column("holdout_release_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("released_at", sa.DateTime(timezone=True), nullable=True)
        )
        batch_op.add_column(
            sa.Column("public_artifact_url", sa.String(length=512), nullable=True)
        )
        batch_op.create_index(
            "ix_miner_submissions_holdout_release_at", ["holdout_release_at"]
        )

    with op.batch_alter_table("backend_evaluation_jobs") as batch_op:
        batch_op.add_column(
            sa.Column("artifact_object_key", sa.String(length=512), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_sha256", sa.String(length=64), nullable=True)
        )
        batch_op.add_column(
            sa.Column("artifact_size_bytes", sa.BigInteger(), nullable=True)
        )


def downgrade() -> None:
    with op.batch_alter_table("backend_evaluation_jobs") as batch_op:
        batch_op.drop_column("artifact_size_bytes")
        batch_op.drop_column("artifact_sha256")
        batch_op.drop_column("artifact_object_key")

    with op.batch_alter_table("miner_submissions") as batch_op:
        batch_op.drop_index("ix_miner_submissions_holdout_release_at")
        batch_op.drop_column("public_artifact_url")
        batch_op.drop_column("released_at")
        batch_op.drop_column("holdout_release_at")
        batch_op.drop_column("artifact_size_bytes")
        batch_op.drop_column("artifact_sha256")
        batch_op.drop_column("artifact_object_key")

    op.drop_index("ix_submission_uploads_status", table_name="submission_uploads")
    op.drop_index("ix_submission_uploads_hotkey", table_name="submission_uploads")
    op.drop_table("submission_uploads")
    op.execute("DROP TYPE IF EXISTS submission_upload_status")
