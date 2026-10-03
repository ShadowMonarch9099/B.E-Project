"""Database schema (plan sections 6.2 and 6.8).

PostgreSQL 16 + pgvector in normal use. The same models also run on SQLite so the unit
tests need no database server; only the dense search (app/index/dense.py) differs.

Chunk records carry every field from plan section 6.2 step 7. The E5 vectors live in
`embeddings`, one row per (chunk, model, view), so E5-base and E5-large (a budget variable
in the plan) and the English shadow view can sit side by side, each with its own HNSW index.
"""

import uuid
from datetime import datetime, timezone

import numpy as np
from sqlalchemy import (JSON, BigInteger, Boolean, DateTime, Float, ForeignKey, Integer,
                        LargeBinary, String, Text, UniqueConstraint)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import TypeDecorator

JSONType = JSON().with_variant(JSONB(), "postgresql")


def new_id() -> str:
    return str(uuid.uuid4())


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class EmbeddingVector(TypeDecorator):
    """pgvector `vector` on PostgreSQL; float32 bytes on SQLite (tests only)."""

    impl = LargeBinary
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            from pgvector.sqlalchemy import Vector

            return dialect.type_descriptor(Vector())
        return dialect.type_descriptor(LargeBinary())

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        arr = np.asarray(value, dtype=np.float32)
        return arr if dialect.name == "postgresql" else arr.tobytes()

    def process_result_value(self, value, dialect):
        if value is None:
            return None
        if dialect.name == "postgresql":
            return np.asarray(value, dtype=np.float32)
        return np.frombuffer(value, dtype=np.float32)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(512))
    sha256: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    size_bytes: Mapped[int] = mapped_column(BigInteger)
    label: Mapped[str] = mapped_column(String(32))  # detected content type, e.g. "pdf"
    # native / scanned / mixed / legacy (PDF text layer), or docx / image
    kind: Mapped[str | None] = mapped_column(String(16))
    detection: Mapped[dict | None] = mapped_column(JSONType)  # the /api/detect result
    original_path: Mapped[str] = mapped_column(String(512))  # relative to UPLOAD_DIR
    viewer_path: Mapped[str | None] = mapped_column(String(512))  # the PDF the viewer opens
    # "page": citations point to PDF pages. "paragraph": a DOCX that could not be
    # converted to PDF, so citations point to paragraphs instead.
    anchor: Mapped[str] = mapped_column(String(16), default="page")
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    stage: Mapped[str | None] = mapped_column(String(32))
    progress_done: Mapped[int] = mapped_column(Integer, default=0)
    progress_total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text)
    page_count: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[list] = mapped_column(JSONType, default=list)  # user-facing notes
    stats: Mapped[dict] = mapped_column(JSONType, default=dict)  # language mix, OCR share, ...
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    pages: Mapped[list["Page"]] = relationship(back_populates="document", cascade="all, delete-orphan",
                                               passive_deletes=True, order_by="Page.page_no")


class Page(Base):
    __tablename__ = "pages"
    __table_args__ = (UniqueConstraint("doc_id", "page_no"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(16))  # native / ocr / docx
    ocr_reason: Mapped[str | None] = mapped_column(String(32))  # no_text_layer / legacy_font
    ocr_conf: Mapped[float | None] = mapped_column(Float)  # mean word confidence, 0-1
    check_reason: Mapped[str | None] = mapped_column(String(32))  # low_confidence / no_text
    width: Mapped[float | None] = mapped_column(Float)  # PDF points
    height: Mapped[float | None] = mapped_column(Float)
    text_raw: Mapped[str] = mapped_column(Text, default="")
    text_norm: Mapped[str] = mapped_column(Text, default="")
    # Paragraphs -> lines -> words, each word {"t": text, "b": [x0, y0, x1, y1] in PDF points
    # (top-left origin), "c": OCR confidence 0-100 or null, "s": start offset in text_raw}.
    layout: Mapped[list] = mapped_column(JSONType, default=list)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    lang: Mapped[str | None] = mapped_column(String(16))
    script: Mapped[str | None] = mapped_column(String(16))

    document: Mapped[Document] = relationship(back_populates="pages")


class Chunk(Base):
    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    page_id: Mapped[int] = mapped_column(ForeignKey("pages.id", ondelete="CASCADE"), index=True)
    page_no: Mapped[int] = mapped_column(Integer)
    para_idx: Mapped[int] = mapped_column(Integer)  # first paragraph of the chunk on its page
    chunk_idx: Mapped[int] = mapped_column(Integer)  # order of the chunk on its page
    seq: Mapped[int] = mapped_column(Integer)  # order of the chunk in the document
    char_start: Mapped[int] = mapped_column(Integer)  # offsets into Page.text_raw
    char_end: Mapped[int] = mapped_column(Integer)
    bbox: Mapped[list] = mapped_column(JSONType, default=list)  # line boxes, PDF points
    text_raw: Mapped[str] = mapped_column(Text)
    text_norm: Mapped[str] = mapped_column(Text)
    text_en: Mapped[str | None] = mapped_column(Text)
    text_en_model: Mapped[str | None] = mapped_column(String(128))
    lang: Mapped[str | None] = mapped_column(String(16))
    lang_scores: Mapped[dict | None] = mapped_column(JSONType)
    script: Mapped[str | None] = mapped_column(String(16))
    source: Mapped[str] = mapped_column(String(16))  # native / ocr / docx
    ocr_conf: Mapped[float | None] = mapped_column(Float)  # 0-1
    token_count: Mapped[int] = mapped_column(Integer, default=0)


class Embedding(Base):
    __tablename__ = "embeddings"

    chunk_id: Mapped[int] = mapped_column(ForeignKey("chunks.id", ondelete="CASCADE"), primary_key=True)
    model: Mapped[str] = mapped_column(String(128), primary_key=True)
    view: Mapped[str] = mapped_column(String(16), primary_key=True)  # "passage" or "en"
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    dim: Mapped[int] = mapped_column(Integer)
    emb = mapped_column(EmbeddingVector())


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    doc_id: Mapped[str | None] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    title: Mapped[str | None] = mapped_column(String(256))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    messages: Mapped[list["Message"]] = relationship(back_populates="conversation", cascade="all, delete-orphan",
                                                     passive_deletes=True, order_by="Message.id")


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[str] = mapped_column(ForeignKey("conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user / assistant
    content: Mapped[str] = mapped_column(Text, default="")
    # complete / abstained / cut_short / refused / error
    status: Mapped[str] = mapped_column(String(16), default="complete")
    payload: Mapped[dict] = mapped_column(JSONType, default=dict)  # citations, sentences, route, ...
    query_log_id: Mapped[int | None] = mapped_column(ForeignKey("query_logs.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    conversation: Mapped[Conversation] = relationship(back_populates="messages")


class QueryLog(Base):
    """One row per question: the router's features, route, scores and outcome.

    These rows are the training data for the future learned router (plan section 6.5)
    and the raw material for the evaluation (section 9).
    """

    __tablename__ = "query_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    conversation_id: Mapped[str | None] = mapped_column(ForeignKey("conversations.id", ondelete="SET NULL"))
    doc_ids: Mapped[list] = mapped_column(JSONType, default=list)
    question: Mapped[str] = mapped_column(Text)
    views: Mapped[dict] = mapped_column(JSONType, default=dict)
    features: Mapped[dict] = mapped_column(JSONType, default=dict)
    route: Mapped[str | None] = mapped_column(String(16))
    route_reason: Mapped[str | None] = mapped_column(Text)
    route_scores: Mapped[dict] = mapped_column(JSONType, default=dict)
    candidates: Mapped[dict] = mapped_column(JSONType, default=dict)  # top hits per route
    reranked: Mapped[list] = mapped_column(JSONType, default=list)
    gate: Mapped[dict] = mapped_column(JSONType, default=dict)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)
    llm_called: Mapped[bool] = mapped_column(Boolean, default=False)
    llm: Mapped[dict] = mapped_column(JSONType, default=dict)  # model, effort, usage, stop_reason
    verification: Mapped[dict] = mapped_column(JSONType, default=dict)
    fidelity: Mapped[dict] = mapped_column(JSONType, default=dict)
    latency_ms: Mapped[dict] = mapped_column(JSONType, default=dict)
    config: Mapped[dict] = mapped_column(JSONType, default=dict)
    error: Mapped[str | None] = mapped_column(Text)
