"""Retrieval routes, fusion, reranking and the evidence gate (plan sections 6.4 and 6.7 step 1).

hybrid:  E5 dense + BM25 words + BM25 character n-grams (+ exact number/ID matches), fused
         with weighted reciprocal rank fusion (RRF), top 40 -> cross-encoder -> top 6.
direct:  E5 dense only, top 40 -> cross-encoder -> top 6.

Every route searches with all of the question's views (plan section 6.3, app/retrieval/
query.py): as typed, the Devanagari transliteration of a romanized question, and the English
translation of a Hindi/Marathi one. Dense search runs once per view and keeps each chunk's
best score; BM25 searches the views together; the reranker scores each passage against the
question as typed and its Devanagari view and keeps the higher score.

The evidence gate abstains, without calling the answer model, when the best reranker
score is below tau (SAKSHYA_GATE_TAU). tau is a hand-set threshold for now: the plan's
calibrated gate (section 6.7) is future work, and tau will be re-chosen from the
risk-coverage curve on the Sakshya-Bench dev split (decision log, 2026-10-03).
"""

import os
import time
from dataclasses import dataclass, field

from ..index import dense
from ..index.lexical import get_lexical_index
from ..ml.embedder import get_embedder
from ..ml.reranker import get_reranker
from .chunks import ChunkInfo, load_chunks
from .numbers import exact_matches
from .query import AnalysisOptions, QueryAnalysis, analyse_query

RRF_K = 60
TOP_K = int(os.environ.get("SAKSHYA_TOP_K", "40"))
TOP_N = int(os.environ.get("SAKSHYA_TOP_N", "6"))
WEIGHTS = {"dense": 1.0, "word": 1.0, "char": 0.6, "exact": 0.8}
# Measured with tools/gate_probe.py on 24 hand-written questions over two of the team's
# sample GRs. With the as-typed question only (phase 4): answerable 0.077-1.000, unanswerable
# 0.002-0.063, so tau was 0.07. With the transliterated and translated views (phase 6), the
# romanized answerable questions rose (0.077 -> 0.992) but so did some romanized off-topic ones
# ("RUSA 3.0 ...", a near-miss on a GR about RUSA 1.0/2.0: 0.063 -> 0.401): answerable
# 0.171-1.000, unanswerable 0.002-0.401. tau = 0.15 answers 14/14 and abstains on 9/10.
# A small sample; to be re-chosen from the risk-coverage curve on the Sakshya-Bench dev split.
GATE_TAU = float(os.environ.get("SAKSHYA_GATE_TAU", "0.15"))
NEAREST_PAGES = 3


@dataclass
class Candidate:
    chunk_id: int
    fused: float = 0.0
    ranks: dict[str, int] = field(default_factory=dict)  # 1-based rank in each list it appears in
    scores: dict[str, float] = field(default_factory=dict)
    rerank: float | None = None

    def as_log(self) -> dict:
        return {"chunk_id": self.chunk_id, "fused": round(self.fused, 5), "ranks": self.ranks,
                "scores": {k: round(v, 4) for k, v in self.scores.items()},
                "rerank": None if self.rerank is None else round(self.rerank, 4)}


@dataclass
class GateDecision:
    passed: bool
    score: float
    tau: float
    reason: str

    def as_dict(self) -> dict:
        return {"passed": self.passed, "score": round(self.score, 4), "tau": self.tau, "reason": self.reason}


@dataclass
class Retrieval:
    question: str
    normalized: str
    route: str
    lists: dict[str, list[tuple[int, float]]]  # per-source hits before fusion
    candidates: list[Candidate]  # fused top-k, reranked order
    top: list[Candidate]  # top-n after reranking
    gate: GateDecision
    chunks: dict[int, ChunkInfo]
    latency_ms: dict[str, float]
    analysis: QueryAnalysis | None = None

    def nearest_pages(self, n: int = NEAREST_PAGES) -> list[dict]:
        """The best distinct pages, for an abstention."""
        seen, out = set(), []
        for c in self.candidates:
            info = self.chunks.get(c.chunk_id)
            if info is None or (info.doc_id, info.page_no) in seen:
                continue
            seen.add((info.doc_id, info.page_no))
            out.append({**info.as_dict(with_text=False), "score": round(c.rerank or 0.0, 4),
                        "snippet": info.text[:240]})
            if len(out) == n:
                break
        return out

    def log(self) -> dict:
        return {"lists": {k: [[cid, round(s, 4)] for cid, s in v] for k, v in self.lists.items()},
                "candidates": [c.as_log() for c in self.candidates]}


def fuse(lists: dict[str, list[tuple[int, float]]], weights: dict[str, float]) -> list[Candidate]:
    """Weighted reciprocal rank fusion: sum over lists of weight / (60 + rank)."""
    out: dict[int, Candidate] = {}
    for name, hits in lists.items():
        w = weights.get(name, 0.0)
        for rank, (cid, score) in enumerate(hits, start=1):
            c = out.setdefault(cid, Candidate(cid))
            c.ranks[name] = rank
            c.scores[name] = score
            c.fused += w / (RRF_K + rank)
    return sorted(out.values(), key=lambda c: -c.fused)


def evidence_gate(top: list[Candidate], tau: float = GATE_TAU) -> GateDecision:
    best = max((c.rerank or 0.0 for c in top), default=0.0)
    if not top:
        return GateDecision(False, 0.0, tau, "nothing was retrieved")
    if best < tau:
        return GateDecision(False, best, tau, f"best passage scores {best:.2f}, below the threshold {tau:.2f}")
    return GateDecision(True, best, tau, f"best passage scores {best:.2f} (threshold {tau:.2f})")


def dense_lists(analysis: QueryAnalysis, doc_ids: list[str] | None, k: int, view: str = "passage",
                timings: dict | None = None) -> list[tuple[int, float]]:
    """Dense search with every view of the question; each chunk keeps its best score."""
    timings = {} if timings is None else timings
    texts = list(analysis.views().values())
    t = time.perf_counter()
    vecs = get_embedder().embed_queries(texts)
    timings["embed"] = timings.get("embed", 0.0) + (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    best: dict[int, float] = {}
    for vec in vecs:
        for cid, score in dense.search(vec, doc_ids, k, view=view):
            best[cid] = max(score, best.get(cid, -1.0))
    timings["dense"] = timings.get("dense", 0.0) + (time.perf_counter() - t) * 1000
    return sorted(best.items(), key=lambda kv: -kv[1])[:k]


def first_stage(analysis: QueryAnalysis, doc_ids: list[str] | None, route: str = "hybrid",
                k: int = TOP_K, timings: dict | None = None) -> dict[str, list[tuple[int, float]]]:
    """The candidate lists a route draws from, before fusion."""
    timings = {} if timings is None else timings
    lists = {"dense": dense_lists(analysis, doc_ids, k, timings=timings)}
    if route == "hybrid":
        t = time.perf_counter()
        lexical = get_lexical_index().search(" ".join(analysis.views().values()), doc_ids, k)
        lists["word"] = [(h.chunk_id, h.score) for h in lexical["word"]]
        lists["char"] = [(h.chunk_id, h.score) for h in lexical["char"]]
        if analysis.numbers:
            lists["exact"] = exact_matches(analysis.numbers, doc_ids, k)
        timings["lexical"] = (time.perf_counter() - t) * 1000
    return lists


def rerank(question_views: list[str], candidates: list[Candidate], chunks: dict[int, ChunkInfo]) -> None:
    """Score every candidate with the cross-encoder; keep the best score over the views."""
    if not candidates:
        return
    passages = [chunks[c.chunk_id].text for c in candidates]
    best = [0.0] * len(candidates)
    for view in question_views:
        for i, score in enumerate(get_reranker().score(view, passages)):
            best[i] = max(best[i], score)
    for c, score in zip(candidates, best):
        c.rerank = score
    candidates.sort(key=lambda c: -(c.rerank or 0.0))


def retrieve(question: str, doc_ids: list[str] | None, route: str = "hybrid", k: int = TOP_K,
             n: int = TOP_N, tau: float = GATE_TAU, weights: dict[str, float] | None = None,
             analysis: QueryAnalysis | None = None, options: AnalysisOptions | None = None) -> Retrieval:
    if route not in {"hybrid", "direct"}:
        raise ValueError(f"unknown route {route}")
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    if analysis is None:
        analysis = analyse_query(question, options)
    timings.update(analysis.latency_ms)
    lists = first_stage(analysis, doc_ids, route, k, timings)
    candidates = fuse(lists, weights or WEIGHTS)[:k]
    chunks = load_chunks([c.chunk_id for c in candidates])
    candidates = [c for c in candidates if c.chunk_id in chunks]

    t = time.perf_counter()
    rerank_views = [question] + ([analysis.devanagari] if analysis.devanagari else [])
    rerank(rerank_views, candidates, chunks)
    timings["rerank"] = (time.perf_counter() - t) * 1000
    top = candidates[:n]
    gate = evidence_gate(top, tau)
    timings["retrieval_total"] = (time.perf_counter() - t0) * 1000
    return Retrieval(question, analysis.normalized, route, lists, candidates, top, gate, chunks,
                     {key: round(v, 1) for key, v in timings.items()}, analysis)
