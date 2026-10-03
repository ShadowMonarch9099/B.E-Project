"""Rule-based router (a stand-in for the plan's learned router N1, section 6.5).

It uses the cheap features the plan lists for the learned router and chooses direct,
translate or hybrid with transparent, ordered rules; each decision carries the rule that
fired, in words, for the "why this route" panel. Every question's features, the chosen route
and the per-route probe scores go to query_logs: they are the training data for the learned
router (decision log, 2026-10-03).

Features (section 6.5): query script, romanized flag, code-mix ratio, length, has a number
or ID, question type, IndicLID confidence, the language mix of the searched documents, their
share of scanned pages and mean OCR confidence, the OCR confidence of the top dense hits,
whether the English shadow index is ready, and each route's best reranker score on its top
three first-stage candidates (the "probe": comparable scores across routes, and the gap
between them).
"""

import re
from dataclasses import asdict, dataclass, field

from sqlalchemy import func, select

from ..db import Chunk, Document, Embedding, Page, session_scope
from ..ml.embedder import get_embedder
from .query import QueryAnalysis

# Question words in English, Hindi and Marathi (Devanagari and romanized) -> question type.
QUESTION_TYPES = [
    ("numeric", r"\b(how much|how many|amount|rate|limit|kiti|kitna|kitne|kitni)\b|किती|कितना|कितने|कितनी|किंमत|रक्कम"),
    ("date", r"\b(when|date|kevha|kab)\b|केव्हा|कधी|कब|तारीख|दिनांक"),
    ("person", r"\b(who|whom|whose|kon|kaun)\b|कोण|कौन|कोणाला|किसे"),
    ("place", r"\b(where|kuthe|kahan|kaha)\b|कुठे|कोठे|कहाँ|कहां"),
    ("procedure", r"\b(how to|how do|how can|kasa|kase|kaise|process|procedure)\b|कसे|कसा|कैसे|प्रक्रिया"),
    ("list", r"\b(which|what are|list|kontya|konte)\b|कोणत्या|कोणते|कौन से"),
    ("reason", r"\b(why|kyon|kyun)\b|का\?|क्यों"),
]

PROBE_GAP = 0.30  # another route's evidence must beat the rule's choice by this much to override it


@dataclass
class Decision:
    route: str
    rule: str  # short id of the rule that fired
    reason: str  # the rule in words, for the UI
    features: dict = field(default_factory=dict)
    probes: dict = field(default_factory=dict)  # route -> {"best": score, "top": [...]}
    override: str | None = None  # set when the probe gap overrode the rule

    def as_dict(self) -> dict:
        return asdict(self)


def question_type(text: str) -> str:
    t = text.lower()
    for name, pattern in QUESTION_TYPES:
        if re.search(pattern, t):
            return name
    return "other"


def collection_features(doc_ids: list[str] | None) -> dict:
    """Language mix, scanned share and OCR confidence of the documents being searched."""
    with session_scope() as s:
        stmt = select(Chunk.lang, func.sum(Chunk.token_count)).join(Document, Document.id == Chunk.doc_id) \
            .where(Document.status == "indexed")
        if doc_ids is not None:
            stmt = stmt.where(Chunk.doc_id.in_(doc_ids))
        mix = {lang or "unknown": int(n or 0) for lang, n in s.execute(stmt.group_by(Chunk.lang)).all()}
        pstmt = select(Page.source, Page.ocr_conf).join(Document, Document.id == Page.doc_id) \
            .where(Document.status == "indexed")
        if doc_ids is not None:
            pstmt = pstmt.where(Page.doc_id.in_(doc_ids))
        pages = s.execute(pstmt).all()
        estmt = select(func.count()).select_from(Embedding).where(Embedding.view == "en",
                                                                  Embedding.model == get_embedder().name)
        cstmt = select(func.count(Chunk.id))
        if doc_ids is not None:
            estmt = estmt.where(Embedding.doc_id.in_(doc_ids))
            cstmt = cstmt.where(Chunk.doc_id.in_(doc_ids))
        shadow, chunks = s.scalar(estmt) or 0, s.scalar(cstmt) or 0
    total = sum(mix.values()) or 1
    share = {k: round(v / total, 3) for k, v in mix.items()}
    confs = [p.ocr_conf for p in pages if p.ocr_conf is not None]
    return {
        "doc_lang_mix": share,
        "doc_main_lang": max(share, key=share.get) if share else None,
        "doc_english_share": share.get("en", 0.0),
        "scanned_share": round(sum(1 for p in pages if p.source == "ocr") / len(pages), 3) if pages else 0.0,
        "doc_mean_ocr_conf": round(sum(confs) / len(confs), 3) if confs else None,
        "shadow_ready": chunks > 0 and shadow >= chunks,
    }


def features(analysis: QueryAnalysis, doc_ids: list[str] | None, top_dense_ocr: list[float | None]) -> dict:
    words = analysis.as_typed.split()
    ocr = [c for c in top_dense_ocr if c is not None]
    return {
        "script": analysis.script,
        "lang": analysis.lang,
        "lang_confidence": analysis.lang_confidence,
        "romanized": analysis.romanized,
        "code_mix": analysis.code_mix,
        "length_words": len(words),
        "has_number_or_id": bool(analysis.numbers),
        "question_type": question_type(analysis.as_typed + " " + (analysis.devanagari or "")),
        "views": sorted(analysis.views()),
        "top_dense_ocr_conf": round(sum(ocr) / len(ocr), 3) if ocr else None,
        "top_dense_scanned_share": round(len(ocr) / len(top_dense_ocr), 3) if top_dense_ocr else 0.0,
        **collection_features(doc_ids),
    }


def choose(f: dict, probes: dict) -> Decision:
    """Ordered rules over the features; then a probe check that can override a rule's choice."""
    lang, main = f.get("lang"), f.get("doc_main_lang")
    cross_lingual = lang in ("en", "hi", "mr") and main in ("en", "hi", "mr") and lang != main
    if f.get("has_number_or_id"):
        d = Decision("hybrid", "numbers", "The question has a number or ID, which must match exactly: hybrid "
                     "search adds BM25 and exact matching to the embeddings.")
    elif f.get("romanized") or (f.get("code_mix") or 0) >= 0.2:
        d = Decision("hybrid", "romanized", "The question is romanized or code-mixed: hybrid search uses its "
                     "Devanagari transliteration and character n-grams, which tolerate spelling variants.")
    elif (f.get("top_dense_ocr_conf") or 1.0) < 0.7:
        d = Decision("hybrid", "low_ocr", "The best matches are on low-confidence scans: hybrid search adds "
                     "character n-grams, which still match words that OCR has damaged.")
    elif cross_lingual and f.get("shadow_ready"):
        d = Decision("translate", "cross_lingual", f"The question ({lang}) and the documents (mostly {main}) "
                     "are in different languages: the translate route compares them in English.")
    elif not cross_lingual and lang in ("hi", "mr"):
        d = Decision("direct", "same_language", "The question and the documents are in the same language and the "
                     "question has no IDs: multilingual embeddings match it by meaning.")
    else:
        d = Decision("hybrid", "default", "No rule is decisive: hybrid search combines meaning and exact words.")
    d.features, d.probes = f, probes

    best = {r: p.get("best") for r, p in probes.items() if p.get("best") is not None}
    if d.route in best and best:
        winner = max(best, key=best.get)
        if winner != d.route and best[winner] - best[d.route] >= PROBE_GAP:
            d.override = d.route
            d.reason += (f" However, the {winner} route found clearly better evidence (reranker {best[winner]:.2f} "
                         f"vs {best[d.route]:.2f}), so it is used instead.")
            d.route, d.rule = winner, f"{d.rule}+probe_gap"
    return d
