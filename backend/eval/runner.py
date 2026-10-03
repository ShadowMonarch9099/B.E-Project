"""Run systems over questions at retrieval level (no answer model, no cost).

Each question is retrieved once per system. The question analysis (language ID,
transliteration, translation) is computed once per set of analysis options, and reranker and
query-embedding scores are cached across systems, so a run of every system costs little more
than the slowest one. Latencies are taken from the first system run on each question (Sakshya
when it is included), before anything for that question is cached.
"""

import time
from contextlib import contextmanager

from app.retrieval import hybrid
from app.retrieval.hybrid import GATE_TAU, TOP_K, TOP_N, retrieve
from app.retrieval.query import MODELS_ON, analyse_query
from app.retrieval.router import question_type

from . import data, metrics
from .systems import System

RANKED_KEEP = 20  # distinct pages kept per system and question, for inspection


class _CachedReranker:
    def __init__(self, inner) -> None:
        self.inner, self.cache = inner, {}

    def score(self, query: str, passages: list[str]) -> list[float]:
        missing = list(dict.fromkeys(p for p in passages if (query, p) not in self.cache))
        if missing:
            for p, sc in zip(missing, self.inner.score(query, missing)):
                self.cache[(query, p)] = sc
        return [self.cache[(query, p)] for p in passages]

    def __getattr__(self, name):
        return getattr(self.inner, name)


class _CachedEmbedder:
    def __init__(self, inner) -> None:
        self.inner, self.cache = inner, {}

    def embed_queries(self, texts: list[str]):
        import numpy as np

        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            for t, v in zip(missing, self.inner.embed_queries(missing)):
                self.cache[t] = v
        return np.stack([self.cache[t] for t in texts])

    def __getattr__(self, name):
        return getattr(self.inner, name)


@contextmanager
def cached_models():
    get_r, get_e = hybrid.get_reranker, hybrid.get_embedder
    reranker, embedder = _CachedReranker(get_r()), _CachedEmbedder(get_e())
    hybrid.get_reranker, hybrid.get_embedder = (lambda: reranker), (lambda: embedder)
    try:
        yield
    finally:
        hybrid.get_reranker, hybrid.get_embedder = get_r, get_e


def _pages(r, candidates) -> list[tuple[str, int]]:
    return [(r.chunks[c.chunk_id].doc_id, r.chunks[c.chunk_id].page_no) for c in candidates if c.chunk_id in r.chunks]


def run(questions: list[dict], systems: list[System], scope: str = "document", progress=print) -> dict:
    """scope "document": each question searches its own document (as a conversation in the
    app does); "collection": every indexed Sakshya-Bench document."""
    docs = data.documents()
    collection = sorted(d["doc_id"] for d in docs.values() if d["manifest_id"])
    out, analyses = [], {}
    t_start = time.perf_counter()
    with cached_models():
        for qi, q in enumerate(questions, start=1):
            gold = data.gold_pages(q, docs)
            base = docs.get(q.get("sha256") or "") or (docs.get((q.get("gold") or {}).get("sha256") or ""))
            gold_doc = docs.get((q.get("gold") or {}).get("sha256") or "")
            row = {"id": q["id"], "question": q["question"], "slice": q.get("slice"), "lang": q.get("lang"),
                   "script": q.get("script"), "answerable": bool(q.get("answerable")), "split": q.get("split"),
                   "qtype": question_type(q["question"]),
                   "doc_type": (gold_doc or base or {}).get("doc_type"),
                   "document": (base or {}).get("manifest_id"),
                   "gold": sorted([list(p) for p in gold]), "systems": {}}
            if scope == "document" and base is None:
                row["skipped"] = "its document is not indexed in the app"
                out.append(row)
                continue
            if row["answerable"] and not gold:
                row["skipped"] = "answerable but its gold page is not in an indexed document"
                out.append(row)
                continue
            doc_ids = [base["doc_id"]] if scope == "document" else collection
            for si, system in enumerate(systems):
                opts = system.options(MODELS_ON)
                key = (q["question"], repr(opts))
                first = key not in analyses
                if first:
                    analyses[key] = analyse_query(q["question"], opts)
                r = retrieve(q["question"], doc_ids, route=system.route, probe=system.route is None,
                             analysis=analyses[key], ocr_aware=system.ocr_aware, use_reranker=system.reranker,
                             tau=GATE_TAU if system.gate else 0.0)
                ranked = metrics.distinct_pages(_pages(r, r.candidates))
                top = set(_pages(r, r.top))
                rec = {"route": r.route, "rule": r.decision.rule if r.decision else None,
                       "override": bool(r.decision and r.decision.override),
                       "answered": r.gate.passed, "gate_score": round(r.gate.score, 4),
                       "evidence_in_top": bool(gold & top) if gold else None,
                       "ranked": [list(p) for p in ranked[:RANKED_KEEP]]}
                if gold:
                    rec.update({k: round(v, 4) for k, v in metrics.retrieval_scores(ranked, gold).items()})
                if si == 0:
                    rec["latency_ms"] = {**r.latency_ms, **({} if first else {"analysis_cached": 1.0})}
                row["systems"][system.name] = rec
            out.append(row)
            if progress and (qi % 10 == 0 or qi == len(questions)):
                progress(f"  {qi}/{len(questions)} questions, {time.perf_counter() - t_start:.0f}s")
    return {"questions": out, "scope": scope, "collection_size": len(collection),
            "config": {"top_k": TOP_K, "top_n": TOP_N, "tau": GATE_TAU, "weights": hybrid.WEIGHTS,
                       "ocr_low": hybrid.OCR_LOW, "query_models": MODELS_ON},
            "seconds": round(time.perf_counter() - t_start, 1)}
