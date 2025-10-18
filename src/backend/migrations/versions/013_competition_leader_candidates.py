"""Add competition leader candidates review table.

Revision ID: 013_comp_leader_candidates
Revises: 011_submission_release_expiry
Create Date: 2025-10-30 00:00:00
"""

from __future__ import annotations

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

ENUM_NAME = "leader_candidate_status"
ENUM_VALUES = ("pending", "approved", "rejected")

revision = "013_comp_leader_candidates"
down_revision = "011_submission_release_expiry"
branch_labels = None
depends_on = None


def _ensure_enum_exists(bind) -> None:
    """Ensure the Postgres enum exists prior to table creation."""

    status_enum = postgresql.ENUM(*ENUM_VALUES, name=ENUM_NAME)
    status_enum.create(bind, checkfirst=True)


def upgrade() -> None:
    bind = op.get_bind()
    _ensure_enum_exists(bind)
    status_enum = postgresql.ENUM(
        *ENUM_VALUES,
        name=ENUM_NAME,
        create_type=False,
    )

    op.create_table(
        "competition_leader_candidates",
        sa.Column("id", sa.BigInteger(), nullable=False),
        sa.Column("competition_id", sa.String(length=64), nullable=False),
        sa.Column("miner_hotkey", sa.String(length=48), nullable=False),
        sa.Column("evaluation_result_id", sa.BigInteger(), nullable=False),
        sa.Column("avg_reward", sa.Float(), nullable=False),
        sa.Column("success_rate", sa.Float(), nullable=True),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("total_episodes", sa.Integer(), nullable=True),
        sa.Column(
            "status",
            status_enum,
            nullable=False,
            server_default="pending",
        ),
        sa.Column("status_reason", sa.Text(), nullable=True),
        sa.Column("reviewed_by_api_key_id", sa.BigInteger(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.ForeignKeyConstraint(["competition_id"], ["competitions.id"]),
        sa.ForeignKeyConstraint(
            ["evaluation_result_id"],
            ["backend_evaluation_results.id"],
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["reviewed_by_api_key_id"],
            ["api_keys.id"],
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "evaluation_result_id", name="uq_leader_candidates_eval_result"
        ),
    )

    op.create_index(
        "ix_leader_candidates_status",
        "competition_leader_candidates",
        ["status"],
    )
    op.create_index(
        "ix_leader_candidates_created_at",
        "competition_leader_candidates",
        ["created_at"],
    )
    op.create_index(
        "ix_leader_candidates_competition",
        "competition_leader_candidates",
        ["competition_id"],
    )
    op.create_index(
        "ix_leader_candidates_miner",
        "competition_leader_candidates",
        ["miner_hotkey"],
    )
    op.create_index(
        "ix_leader_candidates_reviewed",
        "competition_leader_candidates",
        ["reviewed_by_api_key_id"],
    )
    op.create_index(
        "ix_leader_candidates_eval_result",
        "competition_leader_candidates",
        ["evaluation_result_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_leader_candidates_eval_result",
        table_name="competition_leader_candidates",
    )
    op.drop_index(
        "ix_leader_candidates_reviewed",
        table_name="competition_leader_candidates",
    )
    op.drop_index(
        "ix_leader_candidates_miner",
        table_name="competition_leader_candidates",
    )
    op.drop_index(
        "ix_leader_candidates_competition",
        table_name="competition_leader_candidates",
    )
    op.drop_index(
        "ix_leader_candidates_created_at",
        table_name="competition_leader_candidates",
    )
    op.drop_index(
        "ix_leader_candidates_status",
        table_name="competition_leader_candidates",
    )
    op.drop_table("competition_leader_candidates")

    status_enum = postgresql.ENUM(*ENUM_VALUES, name=ENUM_NAME)
    bind = op.get_bind()

    if bind.dialect.name == "postgresql":
        enum_exists = bind.execute(
            sa.text("SELECT 1 FROM pg_type WHERE typname = :name"),
            {"name": ENUM_NAME},
        ).scalar()
        if not enum_exists:
            return

    status_enum.drop(bind, checkfirst=False)
