"""Search the indexes directly (for checking retrieval; the chatbot uses app/retrieval)."""

from fastapi import APIRouter, Depends, Query

from ..index import dense
from ..index.lexical import get_lexical_index
from ..ingest.normalize import normalize
from ..ml.embedder import get_embedder
from ..retrieval.chunks import load_chunks
from .documents import require_db

router = APIRouter(prefix="/api/search", tags=["search"])


@router.get("", dependencies=[Depends(require_db)])
def search(q: str = Query(..., min_length=1, max_length=1000), doc_id: list[str] | None = Query(None),
           k: int = Query(10, ge=1, le=100)) -> dict:
    """Top hits from each index: BM25 words, BM25 character n-grams, and E5 dense."""
    norm = normalize(q)
    lexical = get_lexical_index().search(norm, doc_id, k)
    qvec = get_embedder().embed_queries([norm])[0]
    hits = {"word": [(h.chunk_id, h.score) for h in lexical["word"]],
            "char": [(h.chunk_id, h.score) for h in lexical["char"]],
            "dense": dense.search(qvec, doc_id, k)}
    info = load_chunks([cid for lst in hits.values() for cid, _ in lst])
    return {"query": q, "normalized": norm,
            "hits": {name: [{"score": round(score, 4), **info[cid].as_dict()} for cid, score in lst if cid in info]
                     for name, lst in hits.items()}}
