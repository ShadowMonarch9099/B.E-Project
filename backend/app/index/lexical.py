"""Lexical index: BM25 over IndicNLP tokens and character n-grams (plan section 6.2 step 8).

Two BM25 indexes (the `bm25s` library) over every indexed chunk's normalised text:
- words: IndicNLP tokens, lower-cased, punctuation dropped;
- characters: 3-5-grams of each word with boundary marks ("#घर", "घरक", ...), which still
  match when OCR has broken a word, or when a word is inflected ("योजनेसाठी" vs "योजना").

PostgreSQL full-text search is not used: it ships no Marathi configuration. The indexes
are built in memory from the database and rebuilt when the set of indexed chunks changes.
"""

import logging
import threading
import time
from dataclasses import dataclass

import numpy as np
from sqlalchemy import func, select

from ..db import Chunk, Document, session_scope

log = logging.getLogger("sakshya.lexical")

N_MIN, N_MAX = 3, 5


def word_tokens(text: str) -> list[str]:
    from indicnlp.tokenize.indic_tokenize import trivial_tokenize

    return [t for t in trivial_tokenize(text.lower(), "hi") if any(ch.isalnum() for ch in t)]


def char_ngrams(words: list[str]) -> list[str]:
    grams: list[str] = []
    for w in words:
        marked = f"#{w}#"
        if len(marked) <= N_MIN:
            grams.append(marked)
            continue
        for n in range(N_MIN, N_MAX + 1):
            grams.extend(marked[i:i + n] for i in range(len(marked) - n + 1))
    return grams


@dataclass
class Hit:
    chunk_id: int
    score: float


@dataclass
class _Built:
    signature: tuple
    chunk_ids: np.ndarray
    doc_ids: np.ndarray
    word: object
    char: object
    seconds: float


class LexicalIndex:
    def __init__(self) -> None:
        self._built: _Built | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _signature(s) -> tuple:
        row = s.execute(select(func.count(Chunk.id), func.max(Chunk.id), func.sum(Chunk.id))
                        .join(Document, Document.id == Chunk.doc_id)
                        .where(Document.status == "indexed")).one()
        return tuple(int(v or 0) for v in row)

    def _build(self, s, signature: tuple) -> _Built | None:
        import bm25s

        t0 = time.perf_counter()
        rows = s.execute(select(Chunk.id, Chunk.doc_id, Chunk.text_norm)
                         .join(Document, Document.id == Chunk.doc_id)
                         .where(Document.status == "indexed").order_by(Chunk.id)).all()
        if not rows:
            return None
        words = [word_tokens(r.text_norm) for r in rows]
        word_index = bm25s.BM25()
        word_index.index(words, show_progress=False)
        char_index = bm25s.BM25()
        char_index.index([char_ngrams(w) for w in words], show_progress=False)
        built = _Built(signature, np.array([r.id for r in rows]), np.array([r.doc_id for r in rows], dtype=object),
                       word_index, char_index, time.perf_counter() - t0)
        log.info("Lexical index built over %d chunks in %.1fs", len(rows), built.seconds)
        return built

    def current(self) -> _Built | None:
        with session_scope() as s:
            signature = self._signature(s)
            with self._lock:
                if self._built is None or self._built.signature != signature:
                    self._built = self._build(s, signature)
                return self._built

    def search(self, text: str, doc_ids: list[str] | None = None, k: int = 40) -> dict[str, list[Hit]]:
        """Top-k chunks by word BM25 and by character-n-gram BM25, within `doc_ids` if given."""
        built = self.current()
        if built is None:
            return {"word": [], "char": []}
        mask = None
        if doc_ids is not None:
            mask = np.isin(built.doc_ids, list(doc_ids)).astype(np.float32)
        words = word_tokens(text)
        out = {}
        for name, index, query in (("word", built.word, words), ("char", built.char, char_ngrams(words))):
            if not query:
                out[name] = []
                continue
            scores = index.get_scores(query, weight_mask=mask)
            top = np.argsort(-scores)[:k]
            out[name] = [Hit(int(built.chunk_ids[i]), float(scores[i])) for i in top if scores[i] > 0]
        return out


_index = LexicalIndex()


def get_lexical_index() -> LexicalIndex:
    return _index
