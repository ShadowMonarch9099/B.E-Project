"""Initial schema: documents, pages, chunks, embeddings, conversations, messages, query logs.

Revision ID: 0001
Revises:
Create Date: 2026-10-03
"""
import sqlalchemy as sa
from alembic import op

from app.db.models import EmbeddingVector, JSONType

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def _ts(name: str, nullable: bool = False) -> sa.Column:
    return sa.Column(name, sa.DateTime(timezone=True), nullable=nullable)


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("sha256", sa.String(64), nullable=False, unique=True, index=True),
        sa.Column("size_bytes", sa.BigInteger, nullable=False),
        sa.Column("label", sa.String(32), nullable=False),
        sa.Column("kind", sa.String(16)),
        sa.Column("detection", JSONType),
        sa.Column("original_path", sa.String(512), nullable=False),
        sa.Column("viewer_path", sa.String(512)),
        sa.Column("anchor", sa.String(16), nullable=False, server_default="page"),
        sa.Column("status", sa.String(16), nullable=False, server_default="queued", index=True),
        sa.Column("stage", sa.String(32)),
        sa.Column("progress_done", sa.Integer, nullable=False, server_default="0"),
        sa.Column("progress_total", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text),
        sa.Column("page_count", sa.Integer),
        sa.Column("notes", JSONType),
        sa.Column("stats", JSONType),
        _ts("created_at"), _ts("updated_at"), _ts("indexed_at", nullable=True),
    )

    op.create_table(
        "pages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("page_no", sa.Integer, nullable=False),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("ocr_reason", sa.String(32)),
        sa.Column("ocr_conf", sa.Float),
        sa.Column("check_reason", sa.String(32)),
        sa.Column("width", sa.Float),
        sa.Column("height", sa.Float),
        sa.Column("text_raw", sa.Text, nullable=False, server_default=""),
        sa.Column("text_norm", sa.Text, nullable=False, server_default=""),
        sa.Column("layout", JSONType),
        sa.Column("word_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("lang", sa.String(16)),
        sa.Column("script", sa.String(16)),
        sa.UniqueConstraint("doc_id", "page_no"),
    )

    op.create_table(
        "chunks",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("page_id", sa.Integer, sa.ForeignKey("pages.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("page_no", sa.Integer, nullable=False),
        sa.Column("para_idx", sa.Integer, nullable=False),
        sa.Column("chunk_idx", sa.Integer, nullable=False),
        sa.Column("seq", sa.Integer, nullable=False),
        sa.Column("char_start", sa.Integer, nullable=False),
        sa.Column("char_end", sa.Integer, nullable=False),
        sa.Column("bbox", JSONType),
        sa.Column("text_raw", sa.Text, nullable=False),
        sa.Column("text_norm", sa.Text, nullable=False),
        sa.Column("text_en", sa.Text),
        sa.Column("text_en_model", sa.String(128)),
        sa.Column("lang", sa.String(16)),
        sa.Column("lang_scores", JSONType),
        sa.Column("script", sa.String(16)),
        sa.Column("source", sa.String(16), nullable=False),
        sa.Column("ocr_conf", sa.Float),
        sa.Column("token_count", sa.Integer, nullable=False, server_default="0"),
    )

    op.create_table(
        "embeddings",
        sa.Column("chunk_id", sa.Integer, sa.ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("model", sa.String(128), primary_key=True),
        sa.Column("view", sa.String(16), primary_key=True),
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("dim", sa.Integer, nullable=False),
        sa.Column("emb", EmbeddingVector()),
    )

    op.create_table(
        "conversations",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("doc_id", sa.String(36), sa.ForeignKey("documents.id", ondelete="CASCADE"), index=True),
        sa.Column("title", sa.String(256)),
        _ts("created_at"),
    )

    op.create_table(
        "query_logs",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        _ts("created_at"),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="SET NULL")),
        sa.Column("doc_ids", JSONType),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("views", JSONType),
        sa.Column("features", JSONType),
        sa.Column("route", sa.String(16)),
        sa.Column("route_reason", sa.Text),
        sa.Column("route_scores", JSONType),
        sa.Column("candidates", JSONType),
        sa.Column("reranked", JSONType),
        sa.Column("gate", JSONType),
        sa.Column("abstained", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("llm_called", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("llm", JSONType),
        sa.Column("verification", JSONType),
        sa.Column("fidelity", JSONType),
        sa.Column("latency_ms", JSONType),
        sa.Column("config", JSONType),
        sa.Column("error", sa.Text),
    )

    op.create_table(
        "messages",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("conversation_id", sa.String(36), sa.ForeignKey("conversations.id", ondelete="CASCADE"),
                  nullable=False, index=True),
        sa.Column("role", sa.String(16), nullable=False),
        sa.Column("content", sa.Text, nullable=False, server_default=""),
        sa.Column("status", sa.String(16), nullable=False, server_default="complete"),
        sa.Column("payload", JSONType),
        sa.Column("query_log_id", sa.Integer, sa.ForeignKey("query_logs.id", ondelete="SET NULL")),
        _ts("created_at"),
    )


def downgrade() -> None:
    for table in ("messages", "query_logs", "conversations", "embeddings", "chunks", "pages", "documents"):
        op.drop_table(table)
