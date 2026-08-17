"""add session_fighter uniqueness constraints

Deletes any existing duplicate (session_id, fighter_id) and (session_id,
label) rows in ``session_fighters`` before adding the ``uq_session_fighter``
and ``uq_session_label`` constraints, so upgrading a pre-existing database
(created via ``Base.metadata.create_all()`` before these constraints
existed) never fails on data that predates them. For each duplicate group
the lowest ``id`` (the first-created link) is kept.

Revision ID: f3a1c9d2e8b4
Revises: b566b42839c9
Create Date: 2026-08-17 06:55:00.000000
"""
import sqlalchemy as sa
from alembic import op

revision = 'f3a1c9d2e8b4'
down_revision = 'b566b42839c9'
branch_labels = None
depends_on = None


def upgrade() -> None:
    bind = op.get_bind()
    # Two passes: first collapse duplicate (session_id, fighter_id) pairs,
    # then duplicate (session_id, label) pairs -- each keeping the
    # earliest-created row, so both target constraints hold afterwards.
    bind.execute(
        sa.text(
            "DELETE FROM session_fighters WHERE id NOT IN "
            "(SELECT MIN(id) FROM session_fighters GROUP BY session_id, fighter_id)"
        )
    )
    bind.execute(
        sa.text(
            "DELETE FROM session_fighters WHERE id NOT IN "
            "(SELECT MIN(id) FROM session_fighters GROUP BY session_id, label)"
        )
    )
    with op.batch_alter_table("session_fighters") as batch_op:
        batch_op.create_unique_constraint("uq_session_fighter", ["session_id", "fighter_id"])
        batch_op.create_unique_constraint("uq_session_label", ["session_id", "label"])


def downgrade() -> None:
    with op.batch_alter_table("session_fighters") as batch_op:
        batch_op.drop_constraint("uq_session_label", type_="unique")
        batch_op.drop_constraint("uq_session_fighter", type_="unique")
