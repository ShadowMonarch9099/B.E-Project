"""Sakshya-Bench: questions and their annotations (plan section 8).

Revision ID: 0002
Revises: 0001
Create Date: 2026-10-03
"""
import sqlalchemy as sa
from alembic import op

from app.db.models import JSONType

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "bench_questions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("text", sa.Text, nullable=False),
        # same_language / cross_language / romanized / unanswerable (plan section 8.2)
        sa.Column("slice", sa.String(24), nullable=False),
        sa.Column("lang", sa.String(8)),  # the question's language: en / hi / mr
        sa.Column("script", sa.String(16)),  # devanagari / latin / mixed
        sa.Column("answerable", sa.Boolean, nullable=False, server_default=sa.true()),
        # the document the question was written about (its gold document when answerable)
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="SET NULL"), index=True),
        sa.Column("status", sa.String(16), nullable=False, server_default="verified"),  # draft / verified
        sa.Column("needs_second", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_by", sa.String(64), nullable=False),
        sa.Column("notes", sa.Text),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_table(
        "bench_annotations",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("question_id", sa.String(36), sa.ForeignKey("bench_questions.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("annotator", sa.String(64), nullable=False),
        sa.Column("role", sa.String(16), nullable=False),  # primary / second
        sa.Column("gold_doc_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="SET NULL")),
        sa.Column("gold_page", sa.Integer),
        sa.Column("span_start", sa.Integer),  # offsets into the page's text_raw
        sa.Column("span_end", sa.Integer),
        sa.Column("span_text", sa.Text),
        sa.Column("bbox", JSONType),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("question_id", "role"),
    )


def downgrade() -> None:
    op.drop_table("bench_annotations")
    op.drop_table("bench_questions")
