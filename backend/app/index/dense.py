"""Dense index: multilingual-E5 vectors in pgvector with an HNSW index (plan section 6.2 step 8).

Vectors are stored per (chunk, model, view): view "passage" is the chunk as written, view
"en" its English translation (the translate route's shadow index, plan section 6.4). Each
(model, view) pair gets its own partial HNSW index on PostgreSQL, built over
`emb::vector(dim)`, so E5-base (768) and E5-large (1024) can live in the same table.

On SQLite (tests only) the search is exact, in NumPy.
"""

import logging
import re
from collections.abc import Callable

import numpy as np
from sqlalchemy import bindparam, delete, select, text

from ..db import Chunk, Embedding, is_postgres, session_scope
from ..ml.embedder import get_embedder

log = logging.getLogger("sakshya.dense")

_SAFE = re.compile(r"^[\w./-]+$")
EMBED_BATCH = 32


def _check(model: str, view: str) -> None:
    if not (_SAFE.match(model) and _SAFE.match(view)):
        raise ValueError("unexpected model or view name")


def index_name(model: str, view: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", model.lower()).strip("_")
    return f"ix_emb_hnsw_{slug[-36:]}_{view}"


def ensure_hnsw(model: str, view: str, dim: int) -> None:
    """Create the partial HNSW index for this model and view if it does not exist yet."""
    if not is_postgres():
        return
    _check(model, view)
    sql = (f"CREATE INDEX IF NOT EXISTS {index_name(model, view)} ON embeddings "
           f"USING hnsw ((emb::vector({int(dim)})) vector_cosine_ops) "
           f"WHERE model = '{model}' AND view = '{view}'")
    with session_scope() as s:
        s.execute(text(sql))


def embed_chunks(doc_id: str, view: str = "passage", texts_of: Callable[[Chunk], str | None] | None = None,
                 on_progress: Callable[[int, int], None] | None = None) -> int:
    """Embed every chunk of a document for `view`; returns how many vectors were stored."""
    embedder = get_embedder()
    texts_of = texts_of or (lambda c: c.text_norm)
    with session_scope() as s:
        s.execute(delete(Embedding).where(Embedding.doc_id == doc_id, Embedding.model == embedder.name,
                                          Embedding.view == view))
        chunks = s.scalars(select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.seq)).all()
        items = [(c.id, texts_of(c)) for c in chunks]
    items = [(cid, t) for cid, t in items if t]
    done = 0
    for i in range(0, len(items), EMBED_BATCH):
        batch = items[i:i + EMBED_BATCH]
        vecs = embedder.embed_passages([t for _, t in batch])
        with session_scope() as s:
            s.add_all(Embedding(chunk_id=cid, model=embedder.name, view=view, doc_id=doc_id,
                                dim=embedder.dim, emb=v) for (cid, _), v in zip(batch, vecs))
        done += len(batch)
        if on_progress:
            on_progress(done, len(items))
    ensure_hnsw(embedder.name, view, embedder.dim)
    return done


def has_embeddings(doc_id: str, view: str = "passage") -> bool:
    name = get_embedder().name
    with session_scope() as s:
        return s.scalar(select(Embedding.chunk_id).where(Embedding.doc_id == doc_id, Embedding.model == name,
                                                         Embedding.view == view).limit(1)) is not None


def search(query_vec: np.ndarray, doc_ids: list[str] | None = None, k: int = 40,
           view: str = "passage") -> list[tuple[int, float]]:
    """(chunk id, cosine similarity) of the k nearest chunks, within `doc_ids` if given."""
    embedder = get_embedder()
    model, dim = embedder.name, embedder.dim
    _check(model, view)
    q = np.asarray(query_vec, dtype=np.float32)
    if is_postgres():
        vec = "[" + ",".join(f"{x:.7f}" for x in q.tolist()) + "]"
        where = f"model = '{model}' AND view = '{view}'"
        params: dict = {"q": vec, "k": k}
        if doc_ids is not None:
            where += " AND doc_id IN :docs"
            params["docs"] = list(doc_ids)
        sql = text(f"SELECT chunk_id, 1 - ((emb::vector({dim})) <=> CAST(:q AS vector({dim}))) AS score "
                   f"FROM embeddings WHERE {where} "
                   f"ORDER BY (emb::vector({dim})) <=> CAST(:q AS vector({dim})) LIMIT :k")
        if doc_ids is not None:
            sql = sql.bindparams(bindparam("docs", expanding=True))
        with session_scope() as s:
            s.execute(text(f"SET LOCAL hnsw.ef_search = {max(40, 2 * k)}"))
            s.execute(text("SET LOCAL hnsw.iterative_scan = relaxed_order"))  # filtered searches
            rows = s.execute(sql, params).all()
        return [(int(r.chunk_id), float(r.score)) for r in rows]

    with session_scope() as s:
        stmt = select(Embedding.chunk_id, Embedding.emb).where(Embedding.model == model, Embedding.view == view)
        if doc_ids is not None:
            stmt = stmt.where(Embedding.doc_id.in_(list(doc_ids)))
        rows = s.execute(stmt).all()
    if not rows:
        return []
    mat = np.stack([r.emb for r in rows])
    scores = mat @ q
    top = np.argsort(-scores)[:k]
    return [(int(rows[i].chunk_id), float(scores[i])) for i in top]
