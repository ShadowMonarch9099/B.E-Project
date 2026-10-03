"""Known-item queries: an automatic sanity check, NOT Sakshya-Bench.

A sentence is copied from a page of an indexed document, and retrieval must find that page.
With --translate the sentence is machine-translated to English first (NLLB-200/IndicTrans2),
which exercises the cross-language path. These queries are not questions a person would ask,
and copied sentences favour lexical matching, so the numbers only check that the harness,
the indexes and the routes work end to end. They are never reported as benchmark results.
"""

import random
import re

from sqlalchemy import select

from app.db import Chunk, Document, session_scope
from app.ingest.chunking import sentence_spans

MIN_WORDS, MAX_WORDS = 8, 30


def _candidate_sentences(text: str) -> list[str]:
    out = []
    for start, end in sentence_spans(text):
        s = " ".join(text[start:end].split())
        n = len(s.split())
        if MIN_WORDS <= n <= MAX_WORDS and sum(ch.isalpha() for ch in s) >= 0.6 * len(s.replace(" ", "")):
            out.append(s)
    return out


def sample(docs: dict[str, dict], n: int, seed: int = 11, doc_type: str | None = None,
           translate: bool = False) -> list[dict]:
    """n known-item queries, spread over documents: one sentence each from random chunks.
    docs: data.documents(); doc_type: "native" / "scanned" / None for both."""
    rng = random.Random(seed)
    by_id = {d["doc_id"]: (sha, d) for sha, d in docs.items() if d["manifest_id"]
             and (doc_type is None or d["doc_type"] == doc_type)}
    with session_scope() as s:
        rows = s.execute(select(Chunk.doc_id, Chunk.page_no, Chunk.text_raw, Chunk.lang)
                         .join(Document, Document.id == Chunk.doc_id)
                         .where(Document.status == "indexed", Chunk.doc_id.in_(list(by_id)))).all()
    per_doc: dict[str, list] = {}
    for r in rows:
        per_doc.setdefault(r.doc_id, []).append(r)
    order = sorted(per_doc)
    rng.shuffle(order)
    out, used = [], set()
    attempts = 0
    while len(out) < n and order and attempts < n * 20:
        attempts += 1
        doc_id = order[attempts % len(order)]
        r = rng.choice(per_doc[doc_id])
        sentences = [x for x in _candidate_sentences(r.text_raw) if x not in used]
        if not sentences:
            continue
        sent = rng.choice(sentences)
        used.add(sent)
        sha, d = by_id[doc_id]
        out.append({"id": f"ki-{len(out) + 1:03d}", "question": sent, "slice": "known_item", "lang": r.lang,
                    "script": None, "answerable": True, "status": "verified", "split": d["split"],
                    "document": d["manifest_id"], "sha256": sha,
                    "gold": {"sha256": sha, "page": r.page_no}, "second": None})
    if translate:
        from app.ml.translate import get_translator

        tr = get_translator()
        kept = []
        for q in out:
            if q["lang"] not in ("hi", "mr") or not re.search(r"[ऀ-ॿ]", q["question"]):
                continue
            english = (tr.to_english([q["question"]], q["lang"])[0] or "").strip()
            if english:
                kept.append({**q, "source_text": q["question"], "question": english, "slice": "known_item_mt",
                             "lang": "en", "translated_by": tr.label(q["lang"])})
        out = kept
    return out
